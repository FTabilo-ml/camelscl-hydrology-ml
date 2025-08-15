# -*- coding: utf-8 -*-
# src/models/train_hybrid_models.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any
import os
import json
import re

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------- Rutas ----------
ROOT      = Path(r"C:\Users\felip\camelscl-hydrology-ml")
PRO       = ROOT / "data" / "processed"
FEAT_DIR  = PRO / "features_parts"                 # salida de build_features.py
MODELS    = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
BASIN_DIR = MODELS / "basins"
BASIN_DIR.mkdir(parents=True, exist_ok=True)

# Ruta de política (opcional) generada por build_policy_from_diagnostics.py
POLICY_DIR  = MODELS / "diagnostics"
POLICY_PATH = POLICY_DIR / "gating_policy.json"

# ---------- Config ----------
SPLIT_DATE        = pd.Timestamp("2009-01-01")  # train < 2009, test >= 2009
MIN_ROWS_BASIN    = 900                         # umbral para cuenca "grande" (ajustable)
TARGET_COLS       = ["y_log1p", "y", "discharge_mm"]  # prioridad de target
ID_COL            = "basin_id"
DATE_COL          = "date"

# Núcleo de features: patrones (regex simples) para reducir dimensionalidad
CORE_FEATURE_PATTERNS = [
    r"^prcp_mm_cr2met(_lag(1|2|3|7|14|30)|_sum_(3|7|14|30)d|_mean_(7|30)d)$",
    r"^pet_mm_hargreaves(_lag(1|3|7|14))$",
    r"^deficit(_sum_(3|7|14)d)?$",
    r"^tmean_c(_lag(1|3|7))$",
    r"^tmin_c(_lag(1|3|7))$",
    r"^tmax_c(_lag(1|3|7))$",
    r"^dd_above(0|5)(_sum_(3|7)d)?$",
    r"^swe_mm(_lag(1|3|7))$",
    r"^(elev|area|slope|lat|lon|aridity|glacier|forest|soil|geol).*",
]

# Muestreo controlado para el GLOBAL
CAP_TRAIN_PER_BASIN = 1500     # máx filas por cuenca en TRAIN (global)
CAP_TEST_PER_BASIN  = 500      # máx filas por cuenca en TEST  (global)
MAX_GLOBAL_TRAIN    = 600_000  # tope absoluto de filas train (evitar RAM alta)
MAX_GLOBAL_TEST     = 200_000  # tope absoluto de filas test

# ---------- Política opcional (gating / exclusiones) ----------
def _load_policy() -> Dict[str, Any]:
    if POLICY_PATH.exists():
        try:
            return json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _feature_exclude_map_from_policy(pol: Dict[str, Any]) -> Dict[str, List[str]]:
    fe = pol.get("feature_exclude", {}) if pol else {}
    # normaliza a str -> List[str]
    return {str(k): [str(c) for c in v] for k, v in fe.items()}

def _metric_mode_overrides_from_policy(pol: Dict[str, Any]) -> Dict[str, str]:
    mm = pol.get("metric_mode_overrides", {}) if pol else {}
    return {str(k): str(v) for k, v in mm.items()}

def _model_overrides_from_policy(pol: Dict[str, Any]) -> Dict[str, str]:
    mo = pol.get("model_overrides", {}) if pol else {}
    return {str(k): str(v) for k, v in mo.items()}

# ---------- XGB params ----------
def xgb_cpu_params() -> dict:
    ver = tuple(int(p) for p in xgb.__version__.split(".")[:2])
    base = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
        # puedes añadir early_stopping_rounds en .fit(...)
    )
    if ver >= (2, 0):
        base.update({"device": "cpu"})
    else:
        base.update({"tree_method": "hist"})
    return base

# -------- Utilidades --------
def _pick_target(cols: Iterable[str]) -> str:
    s = set(cols)
    for c in TARGET_COLS:
        if c in s:
            return c
    raise ValueError("No se encontró ninguna columna de target (y_log1p, y, discharge_mm).")

def _load_all_feature_files() -> List[Path]:
    files = sorted(FEAT_DIR.glob("features_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay archivos en {FEAT_DIR}. Ejecuta primero build_features.py")
    return files

def _read_one(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

def _split_temporal(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=[target, DATE_COL, ID_COL])
    train = df[df[DATE_COL] < SPLIT_DATE].copy()
    test  = df[df[DATE_COL] >= SPLIT_DATE].copy()
    if train.empty or test.empty:
        raise ValueError("El split temporal dejó train o test vacío.")
    return train, test

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, target_name: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # Si target es y_log1p, R2 ≡ NSE en log-espacio (lo reportamos como logNSE)
    logNSE = r2 if target_name == "y_log1p" else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "logNSE": logNSE}

def _basin_size_summary(files: List[Path]) -> pd.DataFrame:
    rows = []
    for fp in files:
        try:
            df = _read_one(fp)
            rows.append({"basin_id": str(df[ID_COL].iloc[0]), "rows": len(df)})
        except Exception:
            continue
    s = pd.DataFrame(rows)
    if s.empty:
        print("No pude calcular resumen de tamaños por cuenca (data vacía).")
        return s
    q = s["rows"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    print("Resumen tamaños por cuenca (filas brutas, antes de dropna):",
          {str(k): int(v) for k, v in q.items()})
    return s

def _infer_common_numeric_features(files: List[Path], target: str, sample_files: int = 50) -> List[str]:
    feats_common: set[str] | None = None
    used = 0
    for fp in files[:sample_files]:
        df = _read_one(fp)
        exclude = {ID_COL, DATE_COL, target, "y", "y_log1p", "discharge_mm"}
        feats_num = {c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])}
        if not feats_num:
            continue
        used += 1
        if feats_common is None:
            feats_common = feats_num
        else:
            feats_common &= feats_num
        if feats_common and len(feats_common) < 20 and used > 10:
            break
    if not feats_common:
        raise ValueError("No fue posible inferir un conjunto común de features numéricas.")
    feats = sorted(feats_common)
    print(f"Intersección de features numéricas en {used} cuencas → {len(feats)} columnas")
    return feats

def _select_core_features(cols: List[str]) -> List[str]:
    """Filtra columnas numéricas por patrones hidrológicos clave."""
    keep = []
    for c in cols:
        for pat in CORE_FEATURE_PATTERNS:
            if re.match(pat, c):
                keep.append(c)
                break
    seen = set()
    core = []
    for c in keep:
        if c not in seen:
            core.append(c); seen.add(c)
    return core

# ---- Imputación por reglas ----
ZERO_PATTERNS_IMPUTE = [
    r"_lag\d+$",
    r"_sum_\d+d$",
    r"_mean_\d+d$",
    r"^dd_above(0|5)(?:_sum_(3|7)d)?$",
    r"^deficit(?:_sum_(3|7|14)d)?$",
    r"^swe_mm(_lag(1|3|7))?$",
]
ZERO_REGEX_IMPUTE = [re.compile(p) for p in ZERO_PATTERNS_IMPUTE]

def _impute_rules_fill(df: pd.DataFrame, cols: List[str], ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Imputa in-place:
      - lags/rolling/degree-days/deficit/swe -> 0
      - resto -> mediana (calculada sobre ref si se provee, o sobre df)
    """
    zero_cols, med_cols = [], []
    for c in cols:
        if any(pat.search(c) for pat in ZERO_REGEX_IMPUTE):
            zero_cols.append(c)
        else:
            med_cols.append(c)

    if zero_cols:
        df.loc[:, zero_cols] = df[zero_cols].fillna(0)

    if med_cols:
        _ref = ref if ref is not None else df
        med = _ref[med_cols].median(numeric_only=True)
        df.loc[:, med_cols] = df[med_cols].fillna(med)

    return df
def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    """Crea columnas faltantes como NaN para asegurar esquema fijo."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

def _zero_out_columns(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> None:
    """Setea a 0 columnas 'excluidas' para permitir ancho constante sin usarlas realmente."""
    if cols:
        train.loc[:, cols] = 0.0
        test.loc[:, cols] = 0.0

# ---- Target encoding por cuenca (dos pasadas, memory-safe) ----
def _fit_basin_target_encoding_stream(files: List[Path], target: str):
    """
    Primera pasada: solo TRAIN. Acumula (sum_y, count_y) por basin_id para TE.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for i, fp in enumerate(files, 1):
        df = _read_one(fp)
        try:
            train, _ = _split_temporal(df, target)
        except ValueError:
            continue
        train = train.dropna(subset=[target, ID_COL])
        grp = train.groupby(ID_COL)[target].agg(['sum', 'count']).reset_index()
        for _, row in grp.iterrows():
            b = str(row[ID_COL]); s = float(row['sum']); n = int(row['count'])
            sums[b] = sums.get(b, 0.0) + s
            counts[b] = counts.get(b, 0) + n
        if i % 50 == 0:
            print(f"  TE pass1: {i}/{len(files)} archivos...")
    total_sum = sum(sums.get(b, 0.0) for b in sums)
    total_n   = sum(counts.get(b, 0) for b in counts)
    gmean = (total_sum / total_n) if total_n > 0 else 0.0
    smoothing = 100.0
    mapping = {}
    for b in counts:
        n = counts[b]; m = (sums[b] / n) if n > 0 else gmean
        enc = (n * m + smoothing * gmean) / (n + smoothing)
        mapping[b] = float(enc)
    return mapping, gmean

def _apply_basin_target_encoding(df: pd.DataFrame, mapping: Dict[str, float], global_mean: float, newcol: str = "basin_te"):
    df[newcol] = df[ID_COL].map(mapping).fillna(global_mean)
    return df

# ---------------- Entrenamiento por cuenca grande ----------------
def train_per_basin_models(
    files: List[Path],
    target: str,
    feats_union: List[str],
    feature_exclude_map: Dict[str, List[str]] | None = None,
    metric_mode_overrides: Dict[str, str] | None = None,
    model_overrides: Dict[str, str] | None = None,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    params = xgb_cpu_params()
    records: List[Dict[str, Any]] = []
    model_paths: Dict[str, str] = {}

    skipped_small = 0
    skipped_split = 0
    skipped_no_feats = 0
    skipped_nan = 0

    feature_exclude_map = feature_exclude_map or {}
    metric_mode_overrides = metric_mode_overrides or {}
    model_overrides = model_overrides or {}

    for i, fp in enumerate(files, 1):
        df = _read_one(fp)
        if df.empty or ID_COL not in df.columns:
            skipped_split += 1
            continue

        basin_id = str(df[ID_COL].iloc[0])

        # filtro por tamaño
        if len(df) < MIN_ROWS_BASIN:
            skipped_small += 1
            continue

        # split temporal
        try:
            train, test = _split_temporal(df, target)
        except ValueError:
            skipped_split += 1
            continue

        # seleccionar features presentes en la cuenca (desde feats_union)
        feats = [c for c in feats_union if c in df.columns]

        # aplica exclusiones por cuenca si existen
        drop_cols = set(feature_exclude_map.get(basin_id, []))
        if drop_cols:
            feats = [c for c in feats if c not in drop_cols]

        # si quedan muy pocas columnas, descarta
        if len(feats) < 3:
            skipped_no_feats += 1
            continue

        # Imputación por reglas usando estadísticas del TRAIN
        _impute_rules_fill(train, feats, ref=train)
        _impute_rules_fill(test, feats, ref=train)
        train = train.dropna(subset=[target])
        test  = test.dropna(subset=[target])
        if train.empty or test.empty:
            skipped_nan += 1
            continue

        # matrices
        X_tr = train[feats].to_numpy(dtype=np.float32)
        y_tr = train[target].to_numpy(dtype=np.float32)
        X_te = test[feats].to_numpy(dtype=np.float32)
        y_te = test[target].to_numpy(dtype=np.float32)

        # entrena
        model = xgb.XGBRegressor(**params)
        # puedes activar early stopping si quieres:
        # model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        # métricas
        yhat_tr = model.predict(X_tr)
        yhat_te = model.predict(X_te)
        mtr = _metrics(y_tr, yhat_tr, target_name=target)
        mte = _metrics(y_te, yhat_te, target_name=target)

        # guarda modelo
        outp = BASIN_DIR / f"xgb_basin_{basin_id}.json"
        model.save_model(outp)
        model_paths[basin_id] = str(outp)

        # modo de métrica reportada (por política)
        metric_mode = metric_mode_overrides.get(basin_id, "r2")  # "rmse" o "r2"
        serve_model = model_overrides.get(basin_id, "local")     # "global" o "local"

        records.append({
            "basin_id": basin_id,
            "rows": len(df),
            "train_MAE": mtr["MAE"], "train_RMSE": mtr["RMSE"], "train_R2": mtr["R2"], "train_logNSE": mtr["logNSE"],
            "test_MAE":  mte["MAE"],  "test_RMSE":  mte["RMSE"],  "test_R2":  mte["R2"],  "test_logNSE":  mte["logNSE"],
            "n_features": len(feats),
            "metric_mode": metric_mode,
            "serve_model": serve_model,
        })

        if len(model_paths) % 10 == 0:
            print(f"  entrenadas {len(model_paths)} cuencas (última {basin_id})")

    if records:
        metrics_df = pd.DataFrame.from_records(records)
        # ordena por métrica reportada
        def _reported_metric_row(row):
            return row["test_RMSE"] if row.get("metric_mode", "r2") == "rmse" else row["test_R2"]
        metrics_df["reported_metric"] = metrics_df.apply(_reported_metric_row, axis=1)
        # si es RMSE, ordenar asc; si es R2, desc. Para mezclar, no ordenamos globalmente por reported_metric.
        metrics_df = metrics_df.sort_values(["metric_mode", "reported_metric"], ascending=[True, True])
    else:
        metrics_df = pd.DataFrame(columns=[
            "basin_id","rows","train_MAE","train_RMSE","train_R2","train_logNSE",
            "test_MAE","test_RMSE","test_R2","test_logNSE","n_features","metric_mode","serve_model","reported_metric"
        ])

    print(f"Descartes — small:{skipped_small} split:{skipped_split} no_feats:{skipped_no_feats} nan:{skipped_nan}")
    print(f"Total modelos por cuenca entrenados: {len(model_paths)}")
    return model_paths, metrics_df

# ---------------- Entrenamiento modelo global (fallback) — memory safe ----------------
def train_global_model_streaming(
    files: List[Path],
    target: str,
    feats_union: List[str],
    feature_exclude_map: Dict[str, List[str]] | None = None,
) -> Tuple[str, Dict[str, float], float, Dict[str, Dict[str, float]]]:
    """
    Entrena un modelo global SIN concatenar todo en memoria con esquema FIJO de columnas:
      1) Primera pasada: TE mapping sobre TRAIN.
      2) Segunda pasada: dataset global con columnas constantes (mismo orden) para todas las cuencas.
         - Columnas excluidas por cuenca se mantienen pero se fijan a 0 (equivalente a 'apagarlas').
         - Columnas faltantes se crean y luego se imputan.
    """
    params = xgb_cpu_params()
    feature_exclude_map = feature_exclude_map or {}

    # 1) TE mapping (pasada 1)
    print("Global TE pass1 (stream): calculando encoding por cuenca…")
    te_mapping, te_gmean = _fit_basin_target_encoding_stream(files, target)

    # 2) Construcción de matrices (pasada 2)
    print("Global pass2 (stream): armando dataset con muestreo por cuenca…")
    Xtr_list: List[np.ndarray] = []
    ytr_list: List[np.ndarray] = []
    Xte_list: List[np.ndarray] = []
    yte_list: List[np.ndarray] = []

    # Selección de features core (intersección ∩ core patterns) desde un ejemplo
    df0 = _read_one(files[0])
    exclude = {ID_COL, DATE_COL, target, "y", "y_log1p", "discharge_mm"}
    num_cols = [c for c in df0.columns if c not in exclude and pd.api.types.is_numeric_dtype(df0[c])]
    core_cols = _select_core_features(num_cols)

    # Esquema fijo base (constante para TODAS las cuencas)
    base_feats = sorted(set(core_cols).intersection(set(feats_union)))
    if not base_feats:
        raise RuntimeError("No hay features core válidas para el global.")
    global_cols = list(base_feats)
    if "basin_te" not in global_cols:
        global_cols.append("basin_te")
    print(f"Features core seleccionadas (global, fijas): {len(base_feats)} columnas (+ basin_te)")
    # === NEW: persistir orden exacto de columnas del modelo global ===
    FEATURE_ORDER_PATH = MODELS / "global_feature_order.json"
    try:
        FEATURE_ORDER_PATH.write_text(
            json.dumps({"feature_order": global_cols}, indent=2),
            encoding="utf-8"
        )
        print(f"✔ Guardado orden de features del global → {FEATURE_ORDER_PATH}")
    except Exception as e:
        print(f"[warn] No pude guardar global_feature_order.json: {e}")
# === END NEW ===

    n_tr_total = 0
    n_te_total = 0
    n_features_used: int = len(global_cols)

    for i, fp in enumerate(files, 1):
        df = _read_one(fp)
        if df.empty or ID_COL not in df.columns:
            continue
        basin_id = str(df[ID_COL].iloc[0])

        try:
            train, test = _split_temporal(df, target)
        except ValueError:
            continue

        # aplicar TE (crea columna basin_te con mapping; si no hay mapping, usa media global)
        train = _apply_basin_target_encoding(train, te_mapping, te_gmean, newcol="basin_te")
        test  = _apply_basin_target_encoding(test,  te_mapping, te_gmean, newcol="basin_te")

        # Asegurar esquema FIJO de columnas en train/test
        _ensure_columns(train, global_cols)
        _ensure_columns(test,  global_cols)

        # Columnas a "excluir" en esta cuenca → se fijan a 0 para permitir stacking
        excl_cols = [c for c in feature_exclude_map.get(basin_id, []) if c in global_cols and c != "basin_te"]
        _zero_out_columns(train, test, excl_cols)

        # Imputación por reglas SOLO en columnas no excluidas (basin_te incluida por si acaso)
        impute_cols = [c for c in global_cols if c not in excl_cols]
        _impute_rules_fill(train, impute_cols, ref=train)
        _impute_rules_fill(test,  impute_cols, ref=train)

        # garantizar target válido
        train = train.dropna(subset=[target])
        test  = test.dropna(subset=[target])
        if train.empty or test.empty:
            continue

        # muestreo por cuenca (cada archivo es 1 cuenca)
        if len(train) > CAP_TRAIN_PER_BASIN:
            train = train.sample(n=CAP_TRAIN_PER_BASIN, random_state=42)
        if len(test) > CAP_TEST_PER_BASIN:
            test = test.sample(n=CAP_TEST_PER_BASIN, random_state=42)

        # Acumular matrices (mismo orden de columnas SIEMPRE)
        Xtr_list.append(train[global_cols].to_numpy(dtype=np.float32))
        ytr_list.append(train[target].to_numpy(dtype=np.float32))
        Xte_list.append(test[global_cols].to_numpy(dtype=np.float32))
        yte_list.append(test[target].to_numpy(dtype=np.float32))

        n_tr_total += len(train)
        n_te_total += len(test)

        if (i % 50 == 0) or (n_tr_total >= MAX_GLOBAL_TRAIN) or (n_te_total >= MAX_GLOBAL_TEST):
            print(f"  {i}/{len(files)} archivos — acumulado train={n_tr_total:,} test={n_te_total:,}")
        if n_tr_total >= MAX_GLOBAL_TRAIN and n_te_total >= MAX_GLOBAL_TEST:
            break

    if not Xtr_list:
        raise RuntimeError("No se logró construir dataset global (lista vacía).")

    X_tr = np.vstack(Xtr_list); y_tr = np.concatenate(ytr_list)
    X_te = np.vstack(Xte_list); y_te = np.concatenate(yte_list)

    print(f"Dataset global final: train={X_tr.shape}, test={X_te.shape}")

    # Entrenar
    model = xgb.XGBRegressor(**params)
    # model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)
    mtr = _metrics(y_tr, yhat_tr, target_name=target)
    mte = _metrics(y_te, yhat_te, target_name=target)

    outp = MODELS / "xgb_global.json"
    model.save_model(outp)

    metrics = {"train": mtr, "test": mte, "n_features": n_features_used}
    return str(outp), te_mapping, te_gmean, metrics

# ----------------- MAIN -----------------
def main():
    files = _load_all_feature_files()
    # Determinar target
    sample = _read_one(files[0])
    target = _pick_target(sample.columns)
    print("Target elegido:", target)

    # Resumen de tamaños por cuenca y umbral
    _ = _basin_size_summary(files)
    print(f"Umbral MIN_ROWS_BASIN={MIN_ROWS_BASIN}")

    # Features comunes robustas (para per-basin y base del core)
    feats_union = _infer_common_numeric_features(files, target, sample_files=80)
    print(f"Base de features comunes: {len(feats_union)} columnas")

    # Política (si existe)
    policy = _load_policy()
    feature_exclude_map = _feature_exclude_map_from_policy(policy)
    metric_mode_overrides = _metric_mode_overrides_from_policy(policy)
    model_overrides = _model_overrides_from_policy(policy)

    # 1) Modelos por cuenca grande
    print("Entrenando modelos por cuenca grande…")
    basin_models, per_basin_metrics = train_per_basin_models(
        files, target, feats_union,
        feature_exclude_map=feature_exclude_map,
        metric_mode_overrides=metric_mode_overrides,
        model_overrides=model_overrides,
    )

    # 2) Modelo global (fallback) — streaming + muestreo + core features
    print("Entrenando modelo global (fallback) — streaming…")
    global_model_path, te_mapping, te_global_mean, global_metrics = train_global_model_streaming(
        files, target, feats_union,
        feature_exclude_map=feature_exclude_map,
    )

    # 3) Guardar artefactos
    (MODELS / "basin_models_index.json").write_text(json.dumps(basin_models, indent=2), encoding="utf-8")

    te_artifact = {"mapping": te_mapping, "global_mean": te_global_mean}
    (MODELS / "global_basin_te.json").write_text(json.dumps(te_artifact, indent=2), encoding="utf-8")

    out_metrics = {
        "global": global_metrics,
        "num_basin_models": len(basin_models),
        "split_date": str(SPLIT_DATE.date()),
        "min_rows_basin": MIN_ROWS_BASIN,
        "target": target,
        "caps": {
            "CAP_TRAIN_PER_BASIN": CAP_TRAIN_PER_BASIN,
            "CAP_TEST_PER_BASIN": CAP_TEST_PER_BASIN,
            "MAX_GLOBAL_TRAIN": MAX_GLOBAL_TRAIN,
            "MAX_GLOBAL_TEST": MAX_GLOBAL_TEST,
        },
        # metadatos útiles para serving/QA
        "policy_summary": {
            "has_policy": POLICY_PATH.exists(),
            "feature_exclude_counts": {k: len(v) for k, v in feature_exclude_map.items()},
            "metric_mode_overrides": metric_mode_overrides,
            "model_overrides": model_overrides,
        }
    }
    (MODELS / "metrics_hybrid.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")

    if not per_basin_metrics.empty:
        per_basin_metrics.to_csv(MODELS / "metrics_per_basin.csv", index=False)

    print("✔ Guardado índice de modelos por cuenca →", MODELS / "basin_models_index.json")
    print("✔ Guardado modelo global →", global_model_path)
    print("✔ Guardado target encoding →", MODELS / "global_basin_te.json")
    print("✔ Guardado métricas →", MODELS / "metrics_hybrid.json")
    if not per_basin_metrics.empty:
        print("✔ Métricas por cuenca →", MODELS / "metrics_per_basin.csv")
    if POLICY_PATH.exists():
        print("✔ Política aplicada desde →", POLICY_PATH)
    print("Listo ✅")

if __name__ == "__main__":
    main()
