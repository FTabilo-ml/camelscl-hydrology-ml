# src/models/train_tabpfn_reg.py
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

ROOT    = Path(r"C:\Users\felip\camelscl-hydrology-ml")
PARTS   = ROOT / "data" / "processed" / "master_parts"
MODELS  = ROOT / "models"; MODELS.mkdir(exist_ok=True)

# ===== Configuración =====
CANDIDATE_FEATURES = [
    "prcp_mm_cr2met",
    "tmin_c",
    "tmax_c",
    "tmean_c",            # si existe
    "pet_mm_hargreaves",  # si existe
    "swe_mm",             # si existe
]
TARGET      = "discharge_mm"
SPLIT_DATE  = pd.Timestamp("2009-01-01")  # train < 2009, test >= 2009

# Submuestreo recomendado para TabPFN (≤10k)
MAX_TOTAL_ROWS      = 10_000
MAX_ROWS_PER_BASIN  = 60

def _load_all_parts() -> pd.DataFrame:
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

def _pick_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not feats:
        raise ValueError("No se encontraron columnas de features esperadas en el maestro.")
    return feats

def _temporal_split(df: pd.DataFrame, features: list[str]):
    df = df.dropna(subset=features + [TARGET, "date"])
    train = df[df["date"] < SPLIT_DATE].copy()
    test  = df[df["date"] >= SPLIT_DATE].copy()
    if train.empty or test.empty:
        raise ValueError("Split temporal produjo conjuntos vacíos. Revisa rango de fechas.")
    return train, test

def _balanced_subsample(train: pd.DataFrame) -> pd.DataFrame:
    grp = train.groupby("basin_id", observed=True, sort=False)
    parts = []
    for _, g in grp:
        if len(g) > MAX_ROWS_PER_BASIN:
            g = g.sample(n=MAX_ROWS_PER_BASIN, random_state=42)
        parts.append(g)
    sub = pd.concat(parts, ignore_index=True)
    if len(sub) > MAX_TOTAL_ROWS:
        sub = sub.sample(n=MAX_TOTAL_ROWS, random_state=42)
    sub = sub.sort_values(["basin_id", "date"]).reset_index(drop=True)
    print(f"✔ Train submuestreado: rows={len(sub):,}, basins={sub['basin_id'].nunique()}")
    return sub

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def _make_tabpfn_regressor(device: str):
    from tabpfn import TabPFNRegressor
    from inspect import signature
    sig = signature(TabPFNRegressor.__init__)
    kwargs = {}
    if "device" in sig.parameters:
        kwargs["device"] = device
    if "n_ensembles" in sig.parameters:
        kwargs["n_ensembles"] = 32
    elif "N_ensemble_configurations" in sig.parameters:
        kwargs["N_ensemble_configurations"] = 32
    # NUEVO: permite exceder límites si la versión lo soporta
    if "ignore_pretraining_limits" in sig.parameters:
        kwargs["ignore_pretraining_limits"] = True
    return TabPFNRegressor(**kwargs)


def _disable_flash_attention_if_cuda():
    # Evita kernels SDPA que crashean en algunas GPUs/drivers
    try:
        import torch
        if torch.cuda.is_available():
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            # Opcional: precisión matmul
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def _train_and_predict(device: str, X_tr, y_tr, X_te):
    from tabpfn import TabPFNRegressor
    reg = _make_tabpfn_regressor(device=device)
    print(f"Entrenando TabPFNRegressor en device='{device}' ...")
    reg.fit(X_tr, y_tr)
    print("Prediciendo…")
    yhat_tr = reg.predict(X_tr)
    yhat_te = reg.predict(X_te)
    return reg, yhat_tr, yhat_te

def main():
    # Asegurar tabpfn instalado
    try:
        import tabpfn  # noqa: F401
    except Exception as e:
        msg = "Instala/actualiza TabPFN: pip install -U tabpfn\nDetalle: " + str(e)
        print(msg)
        (MODELS / "metrics_tabpfn.json").write_text(json.dumps({"error": msg}, indent=2), encoding="utf-8")
        return

    # Detectar dispositivo (PyTorch)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print("Dispositivo TabPFN detectado:", device)

    # Desactivar SDPA acelerado si hay CUDA
    _disable_flash_attention_if_cuda()

    # Cargar datos y preparar
    df = _load_all_parts()
    features = _pick_features(df)
    print("Features usadas:", features)

    train, test = _temporal_split(df, features)
    train_sub   = _balanced_subsample(train)

    X_tr = train_sub[features].to_numpy(dtype=np.float32)
    y_tr = train_sub[TARGET].to_numpy(dtype=np.float32)
    X_te = test[features].to_numpy(dtype=np.float32)
    y_te = test[TARGET].to_numpy(dtype=np.float32)

    # Entrenamiento con fallback GPU->CPU si hay error CUDA
    try:
        reg, yhat_tr, yhat_te = _train_and_predict(device, X_tr, y_tr, X_te)
        used_device = device
    except Exception as e:
        print("⚠ Error en GPU:", e)
        print("→ Reintentando en CPU…")
        reg, yhat_tr, yhat_te = _train_and_predict("cpu", X_tr, y_tr, X_te)
        used_device = "cpu"

    m_tr = _metrics(y_tr, yhat_tr)
    m_te = _metrics(y_te, yhat_te)

    # Guardar artefactos
    saved = False
    try:
        from tabpfn.model_loading import save_fitted_tabpfn_model
        save_fitted_tabpfn_model(reg, MODELS / "tabpfn_reg.fit")
        print("✔ Modelo TabPFN guardado en", MODELS / "tabpfn_reg.fit")
        saved = True
    except Exception:
        pass

    if not saved:
        try:
            import joblib
            joblib.dump(reg, MODELS / "tabpfn_reg.pkl")
            print("✔ Modelo TabPFN guardado (joblib) en", MODELS / "tabpfn_reg.pkl")
        except Exception:
            print("ℹ No se pudo serializar el modelo (helpers/joblib no disponibles).")

    (MODELS / "metrics_tabpfn.json").write_text(
        json.dumps({"train": m_tr, "test": m_te, "features": features, "device": used_device}, indent=2),
        encoding="utf-8"
    )
    print("✔ Métricas TabPFN:", {"train": m_tr, "test": m_te, "device": used_device})

if __name__ == "__main__":
    main()
