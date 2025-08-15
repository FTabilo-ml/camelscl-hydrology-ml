# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ID_COL = "basin_id"
DATE_COL = "date"
TARGET_CANDIDATES = ["y_log1p","y","discharge_mm"]

def _pick_target(cols) -> str:
    s = set(cols)
    for c in TARGET_CANDIDATES:
        if c in s:
            return c
    raise ValueError("No target en archivo (y_log1p|y|discharge_mm).")

def _nse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    den = np.sum((y_true - np.mean(y_true))**2)
    num = np.sum((y_true - y_pred)**2)
    return float(1 - num/den) if den > 0 else float('nan')

def main():
    ap = argparse.ArgumentParser(description="Añade Persistencia/NSE al metrics_per_basin.csv (todas las cuencas).")
    ap.add_argument("--data_dir", type=str, required=True, help="data/processed/features_parts")
    ap.add_argument("--metrics_csv", type=str, required=True, help="models/metrics_per_basin.csv (input)")
    ap.add_argument("--split_date", type=str, default="2009-01-01")
    ap.add_argument("--out_csv", type=str, required=True, help="ruta de salida, p.ej. models/metrics_per_basin_with_persistence.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    split_date = pd.Timestamp(args.split_date)

    m = pd.read_csv(args.metrics_csv)
    m["basin_id"] = m["basin_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    pers_vals = []
    for bid in m["basin_id"]:
        fp = data_dir / f"features_{bid}.parquet"
        if not fp.exists():
            pers_vals.append((bid, float("nan"), 0))
            continue
        df = pd.read_parquet(fp)
        if DATE_COL not in df.columns: 
            pers_vals.append((bid, float("nan"), 0)); continue
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        tgt = _pick_target(list(df.columns))
        df = df.sort_values(DATE_COL)
        df["y_shift1"] = df[tgt].shift(1)
        te = df[df[DATE_COL] >= split_date].dropna(subset=[tgt,"y_shift1"])
        if te.empty:
            pers_vals.append((bid, float("nan"), 0))
        else:
            nse = _nse(te[tgt].to_numpy(), te["y_shift1"].to_numpy())
            pers_vals.append((bid, float(nse), int(len(te))))

    pers_df = pd.DataFrame(pers_vals, columns=["basin_id","persistence_NSE","persistence_N"])
    out = pd.merge(m, pers_df, on="basin_id", how="left")
    # Gating sugerido global si persistencia>0 y R2<0
    if "test_R2" in out.columns:
        out["suggested_serving"] = np.where((out["test_R2"] < 0) & (out["persistence_NSE"] > 0), "global", "local")
    out.to_csv(args.out_csv, index=False)
    print(f"✔ {args.out_csv}")

if __name__ == "__main__":
    main()
