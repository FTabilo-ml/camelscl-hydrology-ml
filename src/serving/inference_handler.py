# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List, Tuple
import os
import re
import json
import math
import time
import hashlib
import logging

import numpy as np
import pandas as pd  # opcional (no se usa en rutas, pero útil si extiendes)
import xgboost as xgb
from fastapi import FastAPI, HTTPException

# ---------------- Logging básico ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("camelscl-serving")

# ---------------- Config & paths ----------------
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models")).resolve()

GLOBAL_MODEL_PATH = MODELS_DIR / "xgb_global.json"
GLOBAL_TE_PATH = MODELS_DIR / "global_basin_te.json"            # {"mapping": {...}, "global_mean": ..}
GLOBAL_ORDER_PATH = MODELS_DIR / "global_feature_order.json"    # {"feature_order": [...]}
SERVING_POLICY_PATH = MODELS_DIR / "serving_policy.json"        # {"basins": {bid: {"mode": "...", ...}}}
BASIN_INDEX_PATH = MODELS_DIR / "basin_models_index.json"       # (opcional)
RETRAINED_DIR = MODELS_DIR / "retrained" / "basins_local_lite"  # local-lite por cuenca (jsons)

# ---------------- Imputación: qué va a cero ----------------
ZERO_PATTERNS = [
    r"_lag\d+$",
    r"_sum_\d+d$",
    r"_mean_\d+d$",
    r"^dd_above(0|5)(?:_sum_(3|7)d)?$",
    r"^deficit(?:_sum_(3|7|14)d)?$",
    r"^swe_mm(_lag(1|3|7))?$",
]
ZERO_RE = [re.compile(p) for p in ZERO_PATTERNS]

def _is_zero_feature(col: str) -> bool:
    return any(p.search(col) for p in ZERO_RE)

# ---------------- Utilitarios: hashing / manifest ----------------
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

@lru_cache(maxsize=1)
def load_manifest() -> Dict[str, Any]:
    """Construye un manifest dinámico con hashes y metadatos de artefactos."""
    items: Dict[str, Any] = {}
    for name, path in {
        "global_model": GLOBAL_MODEL_PATH,
        "global_te": GLOBAL_TE_PATH,
        "global_order": GLOBAL_ORDER_PATH,
        "serving_policy": SERVING_POLICY_PATH,
    }.items():
        entry = {"exists": path.exists(), "path": str(path)}
        if path.exists():
            try:
                entry.update({
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                })
            except Exception as e:
                entry.update({"sha256": None, "error": str(e)})
        items[name] = entry
    return {
        "models_dir": str(MODELS_DIR),
        "artifacts": items,
        "generated_at": time.time(),
    }

# ---------------- Lazy loaders ----------------
@lru_cache(maxsize=1)
def load_global_order() -> List[str]:
    obj = json.loads(GLOBAL_ORDER_PATH.read_text(encoding="utf-8"))
    feats = obj.get("feature_order", [])
    if not feats or "basin_te" not in feats:
        raise RuntimeError("global_feature_order.json inválido: falta 'feature_order' o 'basin_te'.")
    return [str(x) for x in feats]

@lru_cache(maxsize=1)
def load_te() -> Dict[str, Any]:
    return json.loads(GLOBAL_TE_PATH.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def load_global_model() -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(GLOBAL_MODEL_PATH))
    return booster

@lru_cache(maxsize=1)
def load_policy() -> Dict[str, Any]:
    if not SERVING_POLICY_PATH.exists():
        # fallback: todo global
        return {"basins": {}}
    return json.loads(SERVING_POLICY_PATH.read_text(encoding="utf-8"))

def _load_local_lite_model(path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster

# ---------------- Feature builders ----------------
def _build_global_vector(basin_id: str, features: Dict[str, Any]) -> Tuple[np.ndarray, List[str], float]:
    """
    Construye vector EXACTO (orden global_feature_order.json).
    Imputación simple: faltantes -> 0; incluye lista de features faltantes y basin_te usado.
    """
    order = load_global_order()  # incluye 'basin_te'
    te = load_te()
    te_map = te.get("mapping", {})
    te_gmean = float(te.get("global_mean", 0.0))

    vec = np.zeros(len(order), dtype=np.float32)
    te_val = float(te_map.get(str(basin_id), te_gmean))
    missing: List[str] = []

    for i, col in enumerate(order):
        if col == "basin_te":
            vec[i] = te_val
            continue
        v = features.get(col, None)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            vec[i] = 0.0  # en producción podrías usar medianas versionadas
            if col != "basin_te":
                missing.append(col)
        else:
            try:
                vec[i] = float(v)
            except Exception:
                vec[i] = 0.0
                if col != "basin_te":
                    missing.append(col)
    return vec.reshape(1, -1), missing, te_val

def _build_local_lite_vector(feature_order: List[str], features: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Construye vector EXACTO para local-lite (orden provisto en policy).
    Imputación simple: faltantes -> 0; retorna lista de faltantes.
    """
    vec = np.zeros(len(feature_order), dtype=np.float32)
    missing: List[str] = []
    for i, col in enumerate(feature_order):
        v = features.get(col, None)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            vec[i] = 0.0
            missing.append(col)
        else:
            try:
                vec[i] = float(v)
            except Exception:
                vec[i] = 0.0
                missing.append(col)
    return vec.reshape(1, -1), missing

# ---------------- Prediction core ----------------
def _predict_booster(booster: xgb.Booster, X: np.ndarray) -> float:
    dmat = xgb.DMatrix(X)
    pred = booster.predict(dmat)
    # XGB devuelve array; tomamos escalar
    return float(np.asarray(pred).ravel()[0])

def predict_one(basin_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplica política de serving y devuelve:
      yhat_log, yhat_mm, model_used, mode, features_missing, basin_te (si global), notes.
    Degradación segura:
      - Si política dice 'local' pero NO tenemos orden de features de ese modelo → usamos GLOBAL.
      - Si 'local_lite' no tiene model_path/feature_order o el archivo no existe → usamos GLOBAL.
    """
    pol = load_policy().get("basins", {})
    entry = pol.get(str(basin_id), {"mode": "global"})
    requested_mode = str(entry.get("mode", "global")).lower()
    mode = requested_mode
    notes: List[str] = []
    features_missing: List[str] = []

    # Local-lite
    if mode == "local_lite":
        model_path = entry.get("model_path", "")
        feat_order = entry.get("feature_order", [])
        if not model_path or not feat_order:
            notes.append("local_lite seleccionado pero falta model_path o feature_order; usando global.")
            mode = "global"
        else:
            path = Path(model_path)
            if not path.is_absolute():
                path = MODELS_DIR / Path(model_path)
            if not path.exists():
                notes.append("local_lite model no encontrado en disco; usando global.")
                mode = "global"
            else:
                booster = _load_local_lite_model(path)
                X, features_missing = _build_local_lite_vector([str(c) for c in feat_order], features)
                yhat_log = _predict_booster(booster, X)
                # log auxiliar
                man = load_manifest()
                logger.info(
                    "predict basin=%s mode=%s used=local_lite missing=%d model_sha=%s policy_sha=%s",
                    basin_id, mode, len(features_missing),
                    man["artifacts"].get("global_model", {}).get("sha256", "NA"),
                    man["artifacts"].get("serving_policy", {}).get("sha256", "NA"),
                )
                return {
                    "basin_id": basin_id,
                    "yhat_log": yhat_log,
                    "yhat_mm": float(np.expm1(yhat_log)),
                    "model_used": "local_lite",
                    "mode": mode,
                    "model_path": str(path),
                    "features_missing": features_missing,
                    "notes": notes,
                }

    # Local full → degradar a global (no tenemos feature_order seguro)
    if mode == "local":
        notes.append("local full seleccionado pero no hay 'feature_order'; usando global.")
        mode = "global"

    # Global (default / degradado)
    booster = load_global_model()
    X, features_missing, te_val = _build_global_vector(basin_id, features)
    yhat_log = _predict_booster(booster, X)
    man = load_manifest()
    logger.info(
        "predict basin=%s mode=%s used=global missing=%d model_sha=%s policy_sha=%s",
        basin_id, mode, len(features_missing),
        man["artifacts"].get("global_model", {}).get("sha256", "NA"),
        man["artifacts"].get("serving_policy", {}).get("sha256", "NA"),
    )
    return {
        "basin_id": basin_id,
        "yhat_log": yhat_log,
        "yhat_mm": float(np.expm1(yhat_log)),
        "model_used": "global",
        "mode": mode,
        "model_path": str(GLOBAL_MODEL_PATH),
        "features_missing": features_missing,
        "basin_te": te_val,
        "notes": notes,
    }

# ---------------- FastAPI ----------------
app = FastAPI(title="CAMELS-CL Hybrid Inference", version="1.1.0")

# Middleware para latencia por request
@app.middleware("http")
async def log_requests(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    logger.info("http path=%s status=%s latency_ms=%.2f", request.url.path, response.status_code, dt_ms)
    return response

# ---- Rutas de salud/metadata ----
@app.get("/healthz")
def healthz():
    ok = GLOBAL_MODEL_PATH.exists() and GLOBAL_TE_PATH.exists() and GLOBAL_ORDER_PATH.exists()
    return {
        "ok": ok,
        "models_dir": str(MODELS_DIR),
        "global_model": str(GLOBAL_MODEL_PATH),
        "global_te": str(GLOBAL_TE_PATH),
        "global_order": str(GLOBAL_ORDER_PATH),
        "serving_policy": str(SERVING_POLICY_PATH),
    }

@app.get("/readyz")
def readyz():
    try:
        _ = load_global_order()
        _ = load_te()
        _ = load_global_model()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/metadata")
def metadata():
    # manifest dinámico con hashes/mtime/size
    return load_manifest()

# ---- Endpoint de predicción ----
from pydantic import BaseModel

class PredictIn(BaseModel):
    basin_id: str
    features: Dict[str, float]  # valores por nombre de feature

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        out = predict_one(inp.basin_id, inp.features)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- Administración: recargar cachés ----
@app.post("/admin/reload")
def admin_reload():
    load_policy.cache_clear()
    load_global_order.cache_clear()
    load_te.cache_clear()
    load_global_model.cache_clear()
    load_manifest.cache_clear()
    return {"ok": True, "reloaded": ["policy", "order", "te", "global_model", "manifest"]}

# al final del archivo
from mangum import Mangum
handler = Mangum(app)
