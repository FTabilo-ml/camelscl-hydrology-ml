# src/preprocessing/parse_attributes_transposed.py
from __future__ import annotations
from pathlib import Path
import io
import re
import pandas as pd

ROOT = Path(r"C:\Users\felip\camelscl-hydrology-ml")
RAW  = ROOT / "data" / "raw"
PRO  = ROOT / "data" / "processed"
PRO.mkdir(parents=True, exist_ok=True)

def _read_text_with_fallback(path: Path) -> str:
    # intenta UTF-8 y luego latin-1, ignorando errores aislados
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    # último recurso: abrir binario y decodificar ignorando
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def _preclean_attributes_text(txt: str) -> str:
    # normaliza saltos de línea
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    # elimina BOM si existe
    txt = txt.lstrip("\ufeff")
    # remueve comillas dobles que rompen el parser (caso reportado)
    txt = txt.replace('"', "")
    # compacta tabs múltiples adyacentes (por seguridad)
    txt = re.sub(r"\t+", "\t", txt)
    # elimina líneas totalmente vacías o con solo espacios/tabs
    txt = "\n".join([ln for ln in txt.split("\n") if ln.strip() != ""])
    return txt

def parse_attributes_transposed(raw_filename: str = "1_CAMELScl_attributes", sep: str = "\t") -> Path:
    """
    Convierte la matriz transpuesta de atributos CAMELS-CL (fila 0 = encabezados con 'gauge_id', col 0 = nombre del atributo)
    a una tabla por cuenca: basin_id + columnas de atributos.
    """
    fpath = RAW / raw_filename
    if not fpath.exists():
        # intenta variantes comunes
        if (RAW / f"{raw_filename}.txt").exists():
            fpath = RAW / f"{raw_filename}.txt"
        elif (RAW / f"{raw_filename}.csv").exists():
            fpath = RAW / f"{raw_filename}.csv"
        else:
            raise FileNotFoundError(f"No se encontró {raw_filename} en {RAW}")

    # --- Prelimpieza robusta del archivo ---
    raw_text = _read_text_with_fallback(fpath)
    cleaned = _preclean_attributes_text(raw_text)

    # Leer SIN header; la fila 0 es el header real
    df_raw = pd.read_csv(
        io.StringIO(cleaned),
        sep=sep,
        comment="#",
        header=None,
        engine="python",
        dtype=str,        # mantenemos como string y convertimos luego
    )

    if df_raw.empty:
        raise ValueError(f"{fpath.name} está vacío tras limpieza.")

    # la primera fila son los encabezados: gauge_id, <ids de cuenca...>
    header = (
        df_raw.iloc[0]
        .astype(str)
        .str.strip()
    )
    data = df_raw.iloc[1:].copy()
    data.columns = header

    if "gauge_id" not in data.columns:
        # a veces aparece como 'gaugeid' o similar; intenta detectar
        candidates = [c for c in data.columns if c.lower().replace("_","") in {"gaugeid","gauge","id"}]
        if candidates:
            data = data.rename(columns={candidates[0]: "gauge_id"})
        else:
            raise ValueError(f"No se encontró columna 'gauge_id'. Encabezados detectados: {list(data.columns)[:10]}")

    # Melt -> (attribute, basin_id, value)
    long_attr = data.melt(id_vars=["gauge_id"], var_name="basin_id", value_name="value")
    long_attr = long_attr.rename(columns={"gauge_id": "attribute"})
    long_attr["basin_id"] = long_attr["basin_id"].astype(str)

    # Pivot -> una fila por cuenca
    wide_attr = long_attr.pivot_table(index="basin_id", columns="attribute", values="value", aggfunc="first")

    # Normaliza nombres de columnas de atributos
    def _norm(s: str) -> str:
        s = s.strip().lower()
        for a, b in ((" ", "_"), ("-", "_"), (".", "_"), ("/", "_")):
            s = s.replace(a, b)
        return s
    wide_attr.columns = [_norm(str(c)) for c in wide_attr.columns]

    wide_attr = wide_attr.reset_index()

    # Conversión numérica tolerante donde aplique
    for c in wide_attr.columns:
        if c == "basin_id":
            continue
        # intenta convertir a numérico; deja NaN si no se puede
        wide_attr[c] = pd.to_numeric(wide_attr[c], errors="coerce")

    out_path = PRO / "attributes.parquet"
    wide_attr.to_parquet(out_path, index=False)
    print(f"✔ {fpath.name} → attributes.parquet  shape={wide_attr.shape}")
    return out_path

if __name__ == "__main__":
    parse_attributes_transposed()
