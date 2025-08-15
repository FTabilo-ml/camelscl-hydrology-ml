# src/build_master.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\felip\camelscl-hydrology-ml")
PRO  = ROOT / "data" / "processed"
OUT  = PRO / "master_table.parquet"

# Variables mínimas v1
NEEDED = {
    "streamflow_mm.parquet":      ("discharge_mm",     ["basin_id","date"]),
    "precip_cr2met.parquet":      ("prcp_mm_cr2met",   ["basin_id","date"]),
    "tmin_cr2met.parquet":        ("tmin_c",           ["basin_id","date"]),
    "tmax_cr2met.parquet":        ("tmax_c",           ["basin_id","date"]),
    "pet_hargreaves.parquet":     ("pet_mm_hargreaves",["basin_id","date"]),
    # opcional:
    # "swe.parquet":              ("swe_mm",           ["basin_id","date"]),
}

def _read(name: str) -> pd.DataFrame:
    p = PRO / name
    if not p.exists():
        raise FileNotFoundError(f"Falta {name} en {PRO}")
    df = pd.read_parquet(p)
    # sanity
    if "basin_id" in df.columns:
        df["basin_id"] = df["basin_id"].astype(str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def build_master() -> Path:
    # base = target (discharge)
    q = _read("streamflow_mm.parquet")        # basin_id, date, discharge_mm
    df = q.copy()

    # merges por fecha y cuenca
    for fname, (colname, keys) in NEEDED.items():
        if fname == "streamflow_mm.parquet":
            continue
        tmp = _read(fname)
        df = df.merge(tmp, on=["basin_id","date"], how="left")

    # atributos estáticos
    attr_path = PRO / "attributes.parquet"
    if attr_path.exists():
        attrs = pd.read_parquet(attr_path)
        if "basin_id" in attrs.columns:
            attrs["basin_id"] = attrs["basin_id"].astype(str)
        df = df.merge(attrs, on="basin_id", how="left")

    # limpieza mínima
    # elimina filas sin target
    df = df.dropna(subset=["discharge_mm"])

    # (opcional) enforce rangos físicos básicos
    for c in ["prcp_mm_cr2met", "pet_mm_hargreaves"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = pd.NA

    df = df.sort_values(["basin_id","date"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"✔ master_table → {OUT}  rows={len(df):,}  cols={len(df.columns)}")
    return OUT

if __name__ == "__main__":
    build_master()
