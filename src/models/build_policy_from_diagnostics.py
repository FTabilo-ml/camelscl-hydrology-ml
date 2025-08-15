# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import json
import re
import numpy as np
import pandas as pd

# --------- Config mínima (coherente con tu training) ----------
EXCLUDE_COLS = {"basin_id", "date", "y", "y_log1p", "discharge_mm"}
SPLIT_DATE_DEFAULT = "2009-01-01"

# Patrones para sugerir imputación = 0 (lags / acumulados / medias / DD / déficit / SWE)
ZERO_PATTERNS = [
    r"_lag\d+$",
    r"_sum_\d+d$",
    r"_mean_\d+d$",
    r"^dd_above(0|5)(?:_sum_(3|7)d)?$",
    r"^deficit(?:_sum_(3|7|14)d)?$",
    r"^swe_mm(_lag(1|3|7))?$",
]
ZERO_REGEX = [re.compile(p) for p in ZERO_PATTERNS]

def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns if c not in EXCLUDE_COLS and _is_num(df[str(c)])]

def _is_zero_impute_feature(col: str) -> bool:
    return any(p.search(col) for p in ZERO_REGEX)

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# --------- Métricas hidrológicas (para logNSE = NSE en log-espacio) ----------
def nse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    den = np.sum((y_true - np.mean(y_true))**2)
    num = np.sum((y_true - y_pred)**2)
    return float(1 - num/den) if den > 0 else float('nan')

# --------- Carga de insumos ----------
def load_inputs(metrics_csv: Path, diagnostics_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(metrics_csv)
    s = pd.read_csv(diagnostics_dir / "summary.csv")
    # str para basin_id por consistencia
    m["basin_id"] = m["basin_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    s["basin_id"] = s["basin_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return m, s

# --------- 1) Near-zero-var → reporte RMSE/MAE/logNSE ----------
def build_near_zero_report(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    nz = summary[summary.get("near_zero_var_test", False) == True].copy()
    if nz.empty:
        return pd.DataFrame(columns=["basin_id","test_MAE","test_RMSE","logNSE_test","n_test","std_test","range_test"])
    join_cols = ["basin_id","n_test","std_test","range_test"]
    left = nz[[c for c in join_cols if c in nz.columns]].copy()
    right_cols = ["basin_id","test_MAE","test_RMSE","test_R2"]
    right = metrics[[c for c in right_cols if c in metrics.columns]].copy()
    j = pd.merge(left, right, on="basin_id", how="left")
    # IMPORTANTE: tu R² actual se calculó en y_log1p → equivale a NSE en log-espacio
    if "test_R2" in j.columns:
        j["logNSE_test"] = j["test_R2"]
    else:
        j["logNSE_test"] = np.nan
    keep = ["basin_id","test_MAE","test_RMSE","logNSE_test","n_test","std_test","range_test"]
    j = j[[c for c in keep if c in j.columns]]
    return j.sort_values("logNSE_test", ascending=True, na_position="last")

# --------- 2) High-NaN por cuenca: exclusiones e imputación sugerida ----------
def analyze_nan_by_basin(data_dir: Path, basin_id: str, split_date: pd.Timestamp) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Devuelve:
      exclude_cols: columnas con >80% NaN en TEST
      impute_advice: lista de (col, 'zero'|'median') para cols con 40–80% NaN
    """
    fp = data_dir / f"features_{basin_id}.parquet"
    if not fp.exists():
        return [], []
    df = pd.read_parquet(fp)
    if "date" not in df.columns:
        return [], []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    te = df[df["date"] >= split_date]
    if te.empty:
        return [], []
    cols = _feature_cols(df)
    if not cols:
        return [], []
    rates = te[cols].isna().mean()  # proporción NaN (Series index puede ser Hashable)
    # Fuerza a str para cumplir con List[str] y evitar Hashable
    exclude_cols: List[str] = [str(c) for c, r in rates.items() if float(r) >= 0.80]
    mid_cols: List[str] = [str(c) for c, r in rates.items() if 0.40 <= float(r) < 0.80]
    impute_advice: List[Tuple[str, str]] = [(c, "zero" if _is_zero_impute_feature(c) else "median") for c in mid_cols]
    return exclude_cols, impute_advice

def build_high_nan_policies(data_dir: Path, joined: pd.DataFrame, split_date: pd.Timestamp) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, str]]]]:
    """
    joined: tabla merge metrics + summary (ver build_policy())
    """
    feature_exclude: Dict[str, List[str]] = {}
    feature_impute_check: Dict[str, List[Dict[str, str]]] = {}

    # high-NaN por regla: promedio ≥ 30% en TEST
    if "avg_nan_rate_test" in joined.columns:
        cand = joined[joined["avg_nan_rate_test"] >= 0.30]
    else:
        cand = joined.iloc[0:0]

    for _, r in cand.iterrows():
        bid = str(r["basin_id"])
        excl, adv = analyze_nan_by_basin(data_dir, bid, split_date)
        if excl:
            feature_exclude[bid] = list(excl)
        if adv:
            feature_impute_check[bid] = [{"column": str(c), "suggestion": str(s)} for (c, s) in adv]
    return feature_exclude, feature_impute_check

# --------- 3) Overrides de modelo (comparar con persistencia) ----------
def choose_model_overrides(joined: pd.DataFrame) -> Dict[str, str]:
    """
    Regla: si test_R2 < 0 y persistence_NSE > 0 → usar modelo global
    """
    out: Dict[str, str] = {}
    cols = set(joined.columns)
    if {"test_R2","persistence_NSE","basin_id"}.issubset(cols):
        subset = joined[(joined["test_R2"] < 0) & (joined["persistence_NSE"] > 0)]
        for _, r in subset.iterrows():
            out[str(r["basin_id"])] = "global"
    return out

# --------- 4) Construcción de política principal ----------
def build_policy(metrics: pd.DataFrame, summary: pd.DataFrame, data_dir: Path, split_date: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    left_cols = [c for c in ["basin_id","rows","test_R2","test_RMSE","test_MAE"] if c in metrics.columns]
    right_cols = [c for c in ["basin_id","n_test","std_test","range_test","near_zero_var_test","avg_nan_rate_test","persistence_NSE"] if c in summary.columns]

    joined = pd.merge(
        metrics[left_cols],
        summary[right_cols],
        on="basin_id", how="left"
    )

    # 1) near-zero-var → metric_mode="rmse"
    metric_mode_overrides: Dict[str, str] = {
        str(r["basin_id"]): "rmse"
        for _, r in summary[summary.get("near_zero_var_test", False) == True].iterrows()
    }

    # 2) high-NaN → exclusiones e imputación sugerida
    feature_exclude, feature_impute_check = build_high_nan_policies(data_dir, joined, split_date)

    # 3) overrides de modelo (persistencia vs local)
    model_overrides = choose_model_overrides(joined)

    policy: Dict[str, Any] = {
        "rules": {
            "metric_mode.near_zero_var_test": "rmse",
            "model_override": "use_global_if(test_R2<0 and persistence_NSE>0)",
            "high_nan_threshold": 0.30,
            "exclude_threshold": 0.80,
            "impute_check_threshold": [0.40, 0.80],
            "impute_suggestion": {
                "zero": "lags/sum/mean/deficit/swe",
                "median": "resto (mediana del TRAIN por cuenca)"
            }
        },
        "metric_mode_overrides": metric_mode_overrides,      # {basin_id: "rmse"}
        "model_overrides": model_overrides,                  # {basin_id: "global"}
        "feature_exclude": feature_exclude,                  # {basin_id: [cols...]}
        "feature_impute_check": feature_impute_check         # {basin_id: [{column,suggestion}, ...]}
    }

    return joined, policy

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Construye política de gating y limpieza por cuenca a partir de diagnostics y métricas.")
    ap.add_argument("--data_dir", type=str, required=True, help="data/processed/features_parts")
    ap.add_argument("--metrics_csv", type=str, required=True, help="models/metrics_per_basin.csv")
    ap.add_argument("--diagnostics_dir", type=str, required=True, help="models/diagnostics (con summary.csv)")
    ap.add_argument("--out_dir", type=str, required=True, help="carpeta de salida (p.ej., models/diagnostics)")
    ap.add_argument("--split_date", type=str, default=SPLIT_DATE_DEFAULT)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    diagnostics_dir = Path(args.diagnostics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_date = pd.Timestamp(args.split_date)

    metrics, summary = load_inputs(Path(args.metrics_csv), diagnostics_dir)

    # Reporte específico para near-zero-var (RMSE/MAE/logNSE)
    near_zero_report = build_near_zero_report(metrics, summary)
    near_zero_report.to_csv(out_dir / "near_zero_report.csv", index=False)

    # Política completa
    joined, policy = build_policy(metrics, summary, data_dir, split_date)
    joined.to_csv(out_dir / "diagnostics_joined.csv", index=False)
    (out_dir / "gating_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    print(f"✔ near_zero_report.csv → {out_dir / 'near_zero_report.csv'}")
    print(f"✔ diagnostics_joined.csv → {out_dir / 'diagnostics_joined.csv'}")
    print(f"✔ gating_policy.json → {out_dir / 'gating_policy.json'}")

if __name__ == "__main__":
    main()
