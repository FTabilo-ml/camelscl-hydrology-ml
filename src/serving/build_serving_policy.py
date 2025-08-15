# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import argparse
import json
import ast
import datetime as dt

import pandas as pd
import numpy as np


def _parse_features_used(val: Any) -> List[str]:
    """
    'features_used' en retrained_metrics.csv suele venir como string de lista.
    Intentamos ast.literal_eval; si falla, devolvemos lista vacía.
    """
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return []


def _to_float(x: Any, default: float = np.nan) -> float:
    """
    Convierte de manera robusta Series/ndarray/list/escalares a float.
    Si no se puede, devuelve 'default' (NaN por defecto).
    """
    try:
        if isinstance(x, pd.Series):
            if x.empty:
                return float(default)
            return float(x.iloc[0])
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x)
            if arr.size == 0:
                return float(default)
            return float(arr.ravel()[0])
        # dict-like con .get?
        if hasattr(x, "item"):
            # numpy scalar
            try:
                return float(x.item())
            except Exception:
                pass
        return float(x)
    except Exception:
        return float(default)


def build_policy(
    metrics_with_persistence_csv: Path,
    retrained_compare_csv: Path,
    retrained_metrics_csv: Path,
    basin_models_index_json: Path,
    retrained_models_dir: Path,
    out_json: Path,
    r2_good_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Reglas de fusión (prioridad alta → baja):
      1) Si metrics_with_persistence.suggested_serving == "global" → GLOBAL.
      2) Si retrained_compare.suggested_serving == "local_lite" y ΔR²>0 → LOCAL_LITE
         (y existe el modelo *_lite.json).
      3) Si existe modelo local “full” y test_R2 >= r2_good_threshold → LOCAL FULL.
      4) Else → GLOBAL.
    """
    # 1) cargar tablas
    mp = pd.read_csv(metrics_with_persistence_csv)
    mp["basin_id"] = mp["basin_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    # Desduplicar por cuenca para que .loc devuelva Series (no DataFrame)
    if "basin_id" in mp.columns:
        mp = mp.sort_values("basin_id").drop_duplicates("basin_id", keep="first")

    rc = pd.read_csv(retrained_compare_csv)
    rc["basin_id"] = rc["basin_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    # Asegurar columnas para delta_R2
    for col in ["test_R2_orig", "test_R2_lite"]:
        if col not in rc.columns:
            rc[col] = np.nan
    if "delta_R2" not in rc.columns:
        rc["delta_R2"] = rc["test_R2_lite"] - rc["test_R2_orig"]
    # Desduplicar
    if "basin_id" in rc.columns:
        rc = rc.sort_values("basin_id").drop_duplicates("basin_id", keep="first")

    rm = pd.read_csv(retrained_metrics_csv)
    rm["basin_id"] = rm["basin_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    # 2) índice de locales full
    if basin_models_index_json.exists():
        local_index = json.loads(basin_models_index_json.read_text(encoding="utf-8"))
        local_index = {str(k): str(v) for k, v in local_index.items()}
    else:
        local_index = {}

    # 3) dict auxiliar: para local_lite, recuperar feature_order por cuenca
    feats_lite_map: Dict[str, List[str]] = {}
    for _, row in rm.iterrows():
        bid = str(row.get("basin_id", ""))
        feats_used = _parse_features_used(row.get("features_used", "[]"))
        if bid and feats_used:
            feats_lite_map[bid] = feats_used

    # 4) construir política
    mp_key = mp.set_index("basin_id") if "basin_id" in mp.columns else pd.DataFrame().set_index(pd.Index([]))
    rc_key = rc.set_index("basin_id") if "basin_id" in rc.columns else pd.DataFrame().set_index(pd.Index([]))

    all_basins = set(mp["basin_id"]).union(set(local_index.keys())).union(set(rc["basin_id"]))
    policy: Dict[str, Any] = {
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "r2_good_threshold": r2_good_threshold,
        "basins": {}
    }

    for bid in sorted(all_basins):
        # default
        mode = "global"

        # Regla 1: persistencia sugiere global
        mp_row = mp_key.loc[bid] if bid in mp_key.index else None
        if mp_row is not None:
            suggested = str(mp_row.get("suggested_serving", "")).lower()
            if suggested == "global":
                policy["basins"][bid] = {"mode": "global"}
                continue

        # Regla 2: local_lite si ΔR2>0 y sugerido local_lite y existe el modelo
        rc_row = rc_key.loc[bid] if bid in rc_key.index else None
        if rc_row is not None:
            sugg = str(rc_row.get("suggested_serving", "")).lower()
            delta_r2 = _to_float(rc_row.get("delta_R2", np.nan))  # <-- FIX: seguro a float
            if sugg == "local_lite" and delta_r2 > 0:
                lite_path = retrained_models_dir / f"xgb_basin_{bid}_lite.json"
                if lite_path.exists():
                    fo = feats_lite_map.get(bid, [])
                    policy["basins"][bid] = {
                        "mode": "local_lite",
                        "model_path": str(lite_path),
                        "feature_order": fo
                    }
                    continue

        # Regla 3: local full si existe modelo y R2>=umbral
        if bid in local_index:
            good_local = False
            if mp_row is not None and "test_R2" in mp_row:
                good_local = _to_float(mp_row.get("test_R2", np.nan)) >= r2_good_threshold  # <-- FIX
            if good_local:
                policy["basins"][bid] = {
                    "mode": "local",
                    "model_path": str(local_index[bid])
                }
                continue

        # Regla 4: global por defecto
        policy["basins"][bid] = {"mode": "global"}

    out_json.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return policy


def main():
    ap = argparse.ArgumentParser(description="Fusiona política de serving (global/local/local_lite) por cuenca.")
    ap.add_argument("--metrics_with_persistence", type=str, required=True,
                    help="models/metrics_per_basin_with_persistence.csv")
    ap.add_argument("--retrained_compare", type=str, required=True,
                    help="models/retrained/retrained_compare.csv")
    ap.add_argument("--retrained_metrics", type=str, required=True,
                    help="models/retrained/retrained_metrics.csv")
    ap.add_argument("--basin_models_index", type=str, required=True,
                    help="models/basin_models_index.json")
    ap.add_argument("--retrained_models_dir", type=str, required=True,
                    help="models/retrained/basins_local_lite")
    ap.add_argument("--out_json", type=str, required=True,
                    help="models/serving_policy.json")
    ap.add_argument("--r2_good_threshold", type=float, default=0.0,
                    help="Umbral para considerar un local full como 'bueno'. Default 0.0")

    args = ap.parse_args()
    policy = build_policy(
        Path(args.metrics_with_persistence),
        Path(args.retrained_compare),
        Path(args.retrained_metrics),
        Path(args.basin_models_index),
        Path(args.retrained_models_dir),
        Path(args.out_json),
        r2_good_threshold=float(args.r2_good_threshold),
    )
    print(f"✔ serving_policy.json → {args.out_json} ({len(policy['basins'])} cuencas)")


if __name__ == "__main__":
    main()
