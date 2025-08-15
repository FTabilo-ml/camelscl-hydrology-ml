# src/features/build_features.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Any
import math
import pandas as pd
import numpy as np

# === Rutas ===
ROOT = Path(r"C:\Users\felip\camelscl-hydrology-ml")
PRO  = ROOT / "data" / "processed"
PARTS_DIR = PRO / "master_parts"                 # entrada (de build_master_partitioned.py)
ATTR_PATH = PRO / "attributes.parquet"          # atributos estáticos por cuenca
OUT_DIR   = PRO / "features_parts"              # salida (una cuenca por archivo)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Configuración de features ===
# lags (en días) por variable meteorológica
LAGS: Dict[str, List[int]] = {
    "prcp_mm_cr2met":      [1, 2, 3, 7, 14, 30],
    "tmin_c":              [1, 3, 7],
    "tmax_c":              [1, 3, 7],
    "tmean_c":             [1, 3, 7],      # si existe; si no, se crea desde tmin/tmax
    "pet_mm_hargreaves":   [1, 3, 7, 14],
    "swe_mm":              [1, 3, 7],      # si existe
    # Lags de Q (target) — ver bandera USE_Q_LAGS abajo
    "discharge_mm":        [1, 3, 7],
}

# ventanas para acumulados/medias móviles (en días)
ROLL_SUM_WINDOWS = [3, 7, 14, 30]
ROLL_MEAN_WINDOWS = [7, 30]

# Bandera: usar lags de caudal (Q) como features.
# OJO: esto mejora mucho, pero para validaciones “puras” puedes poner False.
USE_Q_LAGS = True

# Si quieres crear tmean_c = (tmin+tmax)/2 cuando no exista
CREATE_TMEAN_IF_ABSENT = True

# Columnas básicas esperadas
DATE_COL = "date"
ID_COL   = "basin_id"
TARGET   = "discharge_mm"

# Nombre de salida unificada (opcional)
WRITE_BIG_PARQUET = False                 # True para escribir master_features.parquet
BIG_PARQUET_PATH  = PRO / "master_features.parquet"


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def _add_safe_rolls(s: pd.Series, windows: Iterable[int], how: str = "sum") -> Dict[str, pd.Series]:
    """
    Crea acumulados/medias móviles *sin leakage*: se hace shift(1) antes de la ventana.
    how: 'sum' o 'mean'
    """
    out: Dict[str, pd.Series] = {}
    s_shift = s.shift(1)  # clave para no mirar el día actual
    for w in windows:
        if how == "sum":
            out[f"{s.name}_sum_{w}d"] = s_shift.rolling(window=w, min_periods=1).sum()
        else:
            out[f"{s.name}_mean_{w}d"] = s_shift.rolling(window=w, min_periods=1).mean()
    return out


def _add_lags(df: pd.DataFrame, col: str, lags: Iterable[int]) -> Dict[str, pd.Series]:
    """
    Lags clásicos col_lag{k}; seguro contra leakage porque usa shift(k).
    """
    out: Dict[str, pd.Series] = {}
    if col not in df.columns:
        return out
    for k in lags:
        out[f"{col}_lag{k}"] = df[col].shift(k)
    return out


def _degree_days_above(temp: pd.Series, threshold: float) -> pd.Series:
    # Devuelve Series (no ndarray), preserva índice y nombre
    # Nota: .clip(lower=0) es de pandas y mantiene tipo/metadata
    return (temp.astype(float) - threshold).clip(lower=0)



def _build_features_for_basin(df_b: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """
    df_b: datos de UNA cuenca (ya filtrados) con columnas:
          basin_id, date, discharge_mm, prcp_mm_cr2met, tmin_c, tmax_c, tmean_c?, pet_mm_hargreaves, swe_mm?
    attrs: atributos estáticos para esa cuenca (una fila por basin_id)
    """
    df = df_b.sort_values(DATE_COL).reset_index(drop=True).copy()

    # Asegura tmean si no está
    if CREATE_TMEAN_IF_ABSENT and "tmean_c" not in df.columns:
        if "tmin_c" in df.columns and "tmax_c" in df.columns:
            df["tmean_c"] = (df["tmin_c"] + df["tmax_c"]) / 2.0

    # Asegura no-negatividad básica
    for c in ["prcp_mm_cr2met", "pet_mm_hargreaves", "swe_mm"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # ===== LAGS =====
    lag_cols: Dict[str, pd.Series] = {}
    for var, ks in LAGS.items():
        if var == TARGET and not USE_Q_LAGS:
            continue
        if var in df.columns:
            lag_cols.update(_add_lags(df, var, ks))
    for name, series in lag_cols.items():
        df[name] = series

    # ===== ROLLING SUMS / MEANS (no leakage con shift(1)) =====
    if "prcp_mm_cr2met" in df.columns:
        sums = _add_safe_rolls(df["prcp_mm_cr2met"], ROLL_SUM_WINDOWS, how="sum")
        means = _add_safe_rolls(df["prcp_mm_cr2met"], ROLL_MEAN_WINDOWS, how="mean")
        for name, series in {**sums, **means}.items():
            df[name] = series

    if "pet_mm_hargreaves" in df.columns:
        # déficit hídrico = PET - PRCP y sus acumulados
        deficit = (df["pet_mm_hargreaves"] - df.get("prcp_mm_cr2met", 0.0)).rename("deficit")
        df["deficit"] = deficit
        sums = _add_safe_rolls(deficit, [3, 7, 14], how="sum")
        for name, series in sums.items():
            df[name] = series

    # Degree-days para fusión nival (usar tmean)
    if "tmean_c" in df.columns:
        dd0 = _degree_days_above(df["tmean_c"], 0.0).rename("dd_above0")
        dd5 = _degree_days_above(df["tmean_c"], 5.0).rename("dd_above5")
        dd0_sums = _add_safe_rolls(dd0, [3, 7], how="sum")
        dd5_sums = _add_safe_rolls(dd5, [3, 7], how="sum")
        df["dd_above0"] = dd0
        df["dd_above5"] = dd5
        for name, series in {**dd0_sums, **dd5_sums}.items():
            df[name] = series

    # ===== Join con atributos estáticos =====
    # (mantén basin_id como string para evitar problemas de merge)
    df[ID_COL] = df[ID_COL].astype(str)
    attrs = attrs.copy()
    attrs[ID_COL] = attrs[ID_COL].astype(str)
    df = df.merge(attrs, on=ID_COL, how="left")

    # ===== Target transform (opcional: dejamos ambas) =====
    if TARGET in df.columns:
        df["y"] = df[TARGET].astype(float)
        # versión log1p — más estable; el modelo puede entrenar en y_log1p
        df["y_log1p"] = np.log1p(df["y"].clip(lower=0))


    # Elimina filas iniciales con NaNs por lags/rollings si quieres un dataset “listo”
    # (si prefieres dejarlas y que el trainer las filtre, comenta la línea de abajo)
    # df = df.dropna(subset=["y"] + [c for c in df.columns if c.endswith(("_lag1","_sum_3d","_mean_7d"))], how="any")

    return df


def main(write_big: bool = WRITE_BIG_PARQUET):
    # Cargar atributos estáticos una sola vez
    if not ATTR_PATH.exists():
        raise FileNotFoundError(f"No existe {ATTR_PATH}")
    attrs = pd.read_parquet(ATTR_PATH)
    attrs = _ensure_types(attrs)

    # Recorrer todos los master_{basin}.parquet
    files = sorted(PARTS_DIR.glob("master_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay partes en {PARTS_DIR}")

    written = 0
    for i, fp in enumerate(files, 1):
        df_b = pd.read_parquet(fp)
        df_b = _ensure_types(df_b)

        # mínimo de columnas requeridas
        if TARGET not in df_b.columns or DATE_COL not in df_b.columns or ID_COL not in df_b.columns:
            # saltar partes corruptas
            continue

        # construir features por cuenca
        feat_b = _build_features_for_basin(df_b, attrs)

        basin_id = str(feat_b[ID_COL].iloc[0])
        outp = OUT_DIR / f"features_{basin_id}.parquet"
        feat_b.to_parquet(outp, index=False)
        written += 1

        if i % 25 == 0:
            print(f"  {i}/{len(files)} cuencas procesadas… (última: {outp.name}, rows={len(feat_b):,})")

    print(f"✔ Features por cuenca escritos en {OUT_DIR}  (cuencas={written})")

    if write_big:
        # Unir todo — cuidado con RAM; sólo usa esto si quieres un archivo grande único
        # Alternativa: leer incrementalmente y append a un dataset de parquet particionado (pyarrow.dataset)
        parts = sorted(OUT_DIR.glob("features_*.parquet"))
        dfs = []
        for j, p in enumerate(parts, 1):
            dfs.append(pd.read_parquet(p))
            if j % 50 == 0:
                print(f"  uniendo {j}/{len(parts)}…")
        big = pd.concat(dfs, ignore_index=True)
        big.to_parquet(BIG_PARQUET_PATH, index=False)
        print(f"✔ master_features.parquet → {BIG_PARQUET_PATH} rows={len(big):,}, cols={len(big.columns)}")


if __name__ == "__main__":
    # Ejecuta:  python -m src.features.build_features
    main()
