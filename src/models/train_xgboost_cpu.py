# src/models/train_xgboost_cpu.py
from __future__ import annotations
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT    = Path(r"C:\Users\felip\camelscl-hydrology-ml")
PARTS   = ROOT / "data" / "processed" / "master_parts"
MODELS  = ROOT / "models"; MODELS.mkdir(exist_ok=True)

# Variables candidatas (el script tomará solo las que existan en tu tabla)
CANDIDATE_FEATURES = [
    "prcp_mm_cr2met",
    "tmin_c",
    "tmax_c",
    "tmean_c",            # si existe
    "pet_mm_hargreaves",  # si existe
    "swe_mm",             # si existe
]
TARGET = "discharge_mm"

# Split temporal (defendible en hidrología)
SPLIT_DATE = pd.Timestamp("2009-01-01")  # train < 2009, test >= 2009

# Parámetros XGBoost (CPU)
def cpu_params() -> dict:
    # XGBoost 2.x usa device='cpu'; en 1.x usa tree_method='hist'
    ver = tuple(int(p) for p in xgb.__version__.split(".")[:2])
    base = dict(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
    )
    if ver >= (2, 0):
        base.update({"device": "cpu"})
    else:
        base.update({"tree_method": "hist"})
    return base

def load_all_parts() -> pd.DataFrame:
    files = sorted(PARTS.glob("master_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron partes en {PARTS}")
    dfs = []
    for i, fp in enumerate(files, 1):
        df = pd.read_parquet(fp)
        if "basin_id" in df.columns:
            df["basin_id"] = df["basin_id"].astype(str)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        dfs.append(df)
        if i % 50 == 0:
            print(f"  cargadas {i}/{len(files)} partes...")
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"✔ Datos concatenados: rows={len(all_df):,}, cols={len(all_df.columns)}")
    return all_df

def pick_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not feats:
        raise ValueError("No se encontraron columnas de features esperadas en el maestro.")
    return feats

def temporal_split(df: pd.DataFrame, features: list[str]):
    df = df.dropna(subset=features + [TARGET, "date"])
    train = df[df["date"] < SPLIT_DATE].copy()
    test  = df[df["date"] >= SPLIT_DATE].copy()
    if train.empty or test.empty:
        raise ValueError("Split temporal produjo conjuntos vacíos. Revisa rango de fechas.")
    X_tr = np.asarray(train[features].values, dtype=np.float32)
    y_tr = np.asarray(train[TARGET].values,   dtype=np.float32)
    X_te = np.asarray(test[features].values,  dtype=np.float32)
    y_te = np.asarray(test[TARGET].values,    dtype=np.float32)
    return (X_tr, y_tr, train), (X_te, y_te, test)

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    df = load_all_parts()
    features = pick_features(df)
    print("Features usadas:", features)

    (X_tr, y_tr, train_df), (X_te, y_te, test_df) = temporal_split(df, features)

    params = cpu_params()
    print("Entrenando XGBoost (CPU) con:", params)
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)
    m_tr = metrics(y_tr, yhat_tr)
    m_te = metrics(y_te, yhat_te)

    # Guardar modelo y artefactos
    model_out = MODELS / "xgb_model_cpu.json"
    model.save_model(model_out)

    (MODELS / "metrics_xgb_cpu.json").write_text(
        json.dumps({"train": m_tr, "test": m_te, "features": features}, indent=2),
        encoding="utf-8"
    )

    # Importancias
    booster = model.get_booster()
    imp = booster.get_score(importance_type="gain")
    fmap = {f"f{i}": features[i] for i in range(len(features))}
    imp_named = {fmap.get(k, k): v for k, v in imp.items()}
    pd.Series(imp_named, name="gain").sort_values(ascending=False)\
        .to_csv(MODELS / "feature_importance_xgb_cpu.csv")

    print("✔ Modelo guardado en", model_out)
    print("✔ Métricas:", {"train": m_tr, "test": m_te})
    print("✔ Importancias →", MODELS / "feature_importance_xgb_cpu.csv")

if __name__ == "__main__":
    main()
