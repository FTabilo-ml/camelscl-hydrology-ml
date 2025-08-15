# src/preprocessing/parse_timeseries_wide.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\felip\camelscl-hydrology-ml")
RAW  = ROOT / "data" / "raw"
PRO  = ROOT / "data" / "processed"
PRO.mkdir(parents=True, exist_ok=True)

NA_VALUES = ["-9999", "-9999.0", "NaN", "NA", "", " "]

def parse_wide_timeseries(raw_filename: str, value_name: str, out_name: str, sep: str = "\t", date_col: str | None = None) -> Path:
    """
    Lee un archivo 'ancho' (1a columna = fecha; resto = IDs de cuenca) y lo guarda en formato largo:
    columnas: basin_id, date, <value_name>
    - raw_filename: nombre exacto en data/raw (sin extensión si no la tuviera)
    - value_name: nombre de la variable (ej: 'discharge_mm', 'prcp_mm_cr2met', 'tmin_c', ...)
    - out_name: nombre del parquet de salida en data/processed
    - sep: separador (por defecto TAB)
    - date_col: fuerza el nombre de la columna fecha si no es la primera
    """
    fpath = RAW / raw_filename
    if not fpath.exists():
        # intenta sin extensión o con .txt si aplica
        if (RAW / f"{raw_filename}.txt").exists():
            fpath = RAW / f"{raw_filename}.txt"
        elif (RAW / f"{raw_filename}.csv").exists():
            fpath = RAW / f"{raw_filename}.csv"
        else:
            raise FileNotFoundError(f"No se encontró {raw_filename} en {RAW}")

    df = pd.read_csv(
        fpath, sep=sep, comment="#", encoding="utf-8",
        na_values=NA_VALUES, engine="python"
    )

    # Normalizar headers mínimos
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_") for c in df.columns]

    # Determinar columna fecha
    if date_col is None:
        date_col = df.columns[0]  # heurística: primera columna
    if date_col not in df.columns:
        raise ValueError(f"No encuentro columna de fecha '{date_col}' en {fpath.name}. Columnas: {df.columns.tolist()[:10]}...")

    # Parse fecha
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Melt -> largo
    value_cols = [c for c in df.columns if c != date_col]
    long_df = df.melt(id_vars=date_col, value_vars=value_cols, var_name="basin_id", value_name=value_name)

    # Tipos y orden
    long_df = long_df.rename(columns={date_col: "date"})
    long_df["basin_id"] = long_df["basin_id"].astype(str)
    long_df = long_df.sort_values(["basin_id", "date"]).reset_index(drop=True)

    # Limpieza simple
    # valores negativos imposibles en mm/°C (opcional, aquí solo sentinelas ya están en NaN)
    out_path = PRO / out_name
    long_df.to_parquet(out_path, index=False)
    print(f"✔ {fpath.name} → {out_name}  rows={len(long_df):,}")
    return out_path

if __name__ == "__main__":
    # Ejemplos típicos (ejecuta solo lo que ya tengas en data/raw)
    parse_wide_timeseries("3_CAMELScl_streamflow_mm",  "discharge_mm",     "streamflow_mm.parquet")
    parse_wide_timeseries("4_CAMELScl_precip_cr2met", "prcp_mm_cr2met",   "precip_cr2met.parquet")
    parse_wide_timeseries("8_CAMELScl_tmin_cr2met",   "tmin_c",           "tmin_cr2met.parquet")
    parse_wide_timeseries("9_CAMELScl_tmax_cr2met",   "tmax_c",           "tmax_cr2met.parquet")
    parse_wide_timeseries("10_CAMELScl_tmean_cr2met", "tmean_c",          "tmean_cr2met.parquet")
    parse_wide_timeseries("12_CAMELScl_pet_hargreaves","pet_mm_hargreaves","pet_hargreaves.parquet")
    # opcional
    # parse_wide_timeseries("13_CAMELScl_swe",          "swe_mm",           "swe.parquet")
