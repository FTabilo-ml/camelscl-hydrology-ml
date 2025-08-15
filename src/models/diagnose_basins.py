# -*- coding: utf-8 -*-
# Diagnóstico por cuenca (CAMELS-CL): varianza del target, NaNs y baseline de persistencia.
# Salidas: summary.csv, review_list.csv, missing_top15/<basin_id>.csv

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import re
import numpy as np
import pandas as pd

# ---------------- Métricas hidrológicas ----------------
def nse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    den = np.sum((y_true - y_true.mean())**2)
    num = np.sum((y_true - y_pred)**2)
    return float(1 - num/den) if den > 0 else float('nan')

def nse_log1p(y_true, y_pred) -> float:
    yt = np.log1p(np.maximum(y_true, 0))
    yp = np.log1p(np.maximum(y_pred, 0))
    den = np.sum((yt - yt.mean())**2)
    num = np.sum((yt - yp)**2)
    return float(1 - num/den) if den > 0 else float('nan')

def kge(y_true, y_pred) -> float:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    if yt.size == 0 or yp.size == 0:
        return float('nan')
    syt = float(np.std(yt))
    syp = float(np.std(yp))
    if syt == 0 or syp == 0:
        return float('nan')
    r = float(np.corrcoef(yt, yp)[0, 1])
    alpha = syp / syt
    mean_yt = float(np.mean(yt))
    mean_yp = float(np.mean(yp))
    beta  = mean_yp / mean_yt if mean_yt != 0 else float('nan')
    return float(1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))

# ---------------- Utilidades ----------------
EXCLUDE_COLS = {"basin_id", "date", "y", "y_log1p", "discharge_mm"}

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def choose_feature_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if is_numeric_series(df[c]):
            cols.append(c)
    return cols

def persistence_baseline(y: pd.Series) -> pd.Series:
    # Predicción persistente: y_hat[t] = y[t-1]
    return y.shift(1)

def build_review_list(metrics_csv: Path, n_bottom: int = 20, extra_threshold: float = 0.0) -> pd.DataFrame:
    m = pd.read_csv(metrics_csv)
    if 'basin_id' not in m.columns or 'test_R2' not in m.columns:
        raise ValueError("metrics_per_basin.csv debe incluir columnas 'basin_id' y 'test_R2'.")
    m = m.copy()
    m['basin_id'] = m['basin_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    bottom = m.sort_values('test_R2', ascending=True).head(n_bottom)
    low = m[m['test_R2'] <= extra_threshold]
    review = pd.concat([bottom, low], axis=0).drop_duplicates(subset=['basin_id'])
    keep_cols = [c for c in ['basin_id','rows','test_R2','test_RMSE','train_R2','train_RMSE'] if c in review.columns]
    return review[keep_cols]

def _stats(y: pd.Series) -> Dict[str, float]:
    yv = y.to_numpy(dtype=float)
    if yv.size == 0:
        return {"n":0,"mean":np.nan,"std":np.nan,"var":np.nan,"min":np.nan,"max":np.nan,"rng":np.nan}
    return {
        "n": int(yv.size),
        "mean": float(np.mean(yv)),
        "std": float(np.std(yv)),
        "var": float(np.var(yv)),
        "min": float(np.min(yv)),
        "max": float(np.max(yv)),
        "rng": float(np.max(yv)-np.min(yv)),
    }

def analyze_basin(fp: Path, split_date: pd.Timestamp, target_col: str = "y_log1p") -> Tuple[Dict, pd.DataFrame]:
    df = pd.read_parquet(fp)
    need = {'date','basin_id', target_col}
    if not need.issubset(df.columns):
        missing = list(need - set(df.columns))
        raise ValueError(f"{fp.name}: faltan columnas requeridas: {missing}")
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'basin_id', target_col]).sort_values('date')

    tr = df[df['date'] < split_date]
    te = df[df['date'] >= split_date]

    st_tr = _stats(tr[target_col])
    st_te = _stats(te[target_col])

    # Baseline de persistencia (sobre TEST)
    y_hat_p = persistence_baseline(te[target_col])
    valid = (~y_hat_p.isna()) & (~te[target_col].isna())
    if int(valid.sum()) > 5:
        nse_p  = nse(te[target_col][valid], y_hat_p[valid])
        lnse_p = nse_log1p(np.expm1(te[target_col][valid]), np.expm1(y_hat_p[valid]))
        kge_p  = kge(np.expm1(te[target_col][valid]), np.expm1(y_hat_p[valid]))
    else:
        nse_p = lnse_p = kge_p = float('nan')

    # NaNs en TEST (antes de imputar): top 15
    feat_cols = choose_feature_cols(df)
    te_feat = te[feat_cols] if feat_cols else pd.DataFrame(index=te.index)
    nan_rates = te_feat.isna().mean().sort_values(ascending=False).head(15).reset_index()
    nan_rates.columns = ['column','nan_rate']

    # Señal de varianza casi nula en TEST (umbral absoluto y relativo)
    std_te = st_te["std"]; mean_te = st_te["mean"]
    near_zero_abs = (not pd.isna(std_te)) and (std_te < 1e-3)
    near_zero_rel = (not pd.isna(mean_te)) and ((std_te / (abs(mean_te) + 1e-6)) < 0.01) if not pd.isna(std_te) else False
    near_zero = bool(near_zero_abs or near_zero_rel)

    summary = {
        "basin_id": str(df['basin_id'].iloc[0]),
        "n_train": st_tr["n"], "n_test": st_te["n"],
        "mean_train": st_tr["mean"], "std_train": st_tr["std"], "var_train": st_tr["var"],
        "mean_test": st_te["mean"], "std_test": st_te["std"], "var_test": st_te["var"],
        "range_test": st_te["rng"],
        "near_zero_var_test": near_zero,
        "persistence_NSE": float(nse_p),
        "persistence_logNSE": float(lnse_p),
        "persistence_KGE": float(kge_p),
        "avg_nan_rate_test": float(te_feat.isna().mean().mean()) if not te_feat.empty else 0.0,
    }
    return summary, nan_rates

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Diagnóstico por cuenca (CAMELS-CL)")
    ap.add_argument("--data_dir", type=str, required=True, help="Ruta a data/processed/features_parts")
    ap.add_argument("--metrics_csv", type=str, required=True, help="Ruta a models/metrics_per_basin.csv")
    ap.add_argument("--split_date", type=str, default="2009-01-01")
    ap.add_argument("--n_bottom", type=int, default=20, help="N cuencas con peor R² a diagnosticar")
    ap.add_argument("--extra_threshold", type=float, default=0.0, help="Incluye todas las cuencas con R² <= umbral")
    ap.add_argument("--out_dir", type=str, required=True, help="Carpeta de salida para diagnostics")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "missing_top15").mkdir(parents=True, exist_ok=True)

    split_date = pd.Timestamp(args.split_date)

    review = build_review_list(Path(args.metrics_csv), n_bottom=args.n_bottom, extra_threshold=args.extra_threshold)
    review.to_csv(out_dir / "review_list.csv", index=False)

    summaries: List[Dict] = []
    for _, row in review.iterrows():
        basin_id = str(row["basin_id"]).replace(".0","")
        fp = data_dir / f"features_{basin_id}.parquet"
        if not fp.exists():
            print(f"[WARN] No existe {fp.name} — salto")
            continue
        try:
            summary, nan_top = analyze_basin(fp, split_date, target_col="y_log1p")
            summaries.append(summary)
            nan_top.to_csv(out_dir / "missing_top15" / f"{basin_id}.csv", index=False)
            print(f"[OK] {basin_id}")
        except Exception as e:
            print(f"[ERR] {basin_id}: {e}")

    if summaries:
        pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
        print(f"✔ Guardado {out_dir/'summary.csv'}")
    else:
        print("No se generaron resúmenes (verifica rutas y datos).")

if __name__ == "__main__":
    main()
