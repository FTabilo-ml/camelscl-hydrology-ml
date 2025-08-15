# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Any, Union, Optional
import argparse
import json
import os
import re

import numpy as np
from numpy.typing import NDArray, ArrayLike
import pandas as pd
import xgboost as xgb

# -------------------- Constantes --------------------
ID_COL = "basin_id"
DATE_COL = "date"
TARGET_CANDIDATES = ["y_log1p", "y", "discharge_mm"]

ZERO_PATTERNS = [
    r"_lag\d+$",
    r"_sum_\d+d$",
    r"_mean_\d+d$",
    r"^dd_above(0|5)(?:_sum_(3|7)d)?$",
    r"^deficit(?:_sum_(3|7|14)d)?$",
    r"^swe_mm(_lag(1|3|7))?$",
]
ZERO_RE = [re.compile(p) for p in ZERO_PATTERNS]

# Diccionario flexible para filas de salida
RowOut = Dict[str, Union[float, int, str, List[float], List[str]]]


# -------------------- Utils de modelo --------------------
def _xgb_params() -> dict:
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
    )
    if ver >= (2, 0):
        base.update({"device": "cpu"})
    else:
        base.update({"tree_method": "hist"})
    return base


def _pick_target(cols: List[str]) -> str:
    s = set(cols)
    for c in TARGET_CANDIDATES:
        if c in s:
            return c
    raise ValueError("No se encontró target en el parquet (y_log1p|y|discharge_mm)")


def _impute_rules_fill(
    df: pd.DataFrame,
    cols: List[str],
    ref: Optional[pd.DataFrame] = None,
) -> None:
    zero_cols: List[str] = []
    med_cols: List[str] = []
    for c in cols:
        if any(p.search(c) for p in ZERO_RE):
            zero_cols.append(c)
        else:
            med_cols.append(c)

    if zero_cols:
        df.loc[:, zero_cols] = df[zero_cols].fillna(0)

    if med_cols:
        r = ref if ref is not None else df
        med = r[med_cols].median(numeric_only=True)
        df.loc[:, med_cols] = df[med_cols].fillna(med)


def _split(df: pd.DataFrame, target: str, split_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=[target, DATE_COL]).copy()
    if ID_COL in df.columns:
        df = df.dropna(subset=[ID_COL])
    tr = df[df[DATE_COL] < split_date].copy()
    te = df[df[DATE_COL] >= split_date].copy()
    if tr.empty or te.empty:
        raise ValueError("Split temporal dejó train o test vacío.")
    return tr, te


def _nse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt: NDArray[np.float64] = np.asarray(y_true, dtype=np.float64)
    yp: NDArray[np.float64] = np.asarray(y_pred, dtype=np.float64)
    den = np.sum((yt - yt.mean()) ** 2)
    num = np.sum((yt - yp) ** 2)
    return float(1.0 - (num / den)) if den > 0 else float("nan")


def _persistence_on_test(df: pd.DataFrame, target: str, split_date: pd.Timestamp) -> Tuple[float, int]:
    df = df.sort_values(DATE_COL).copy()
    df["y_shift1"] = df[target].shift(1)
    te = df[df[DATE_COL] >= split_date].copy()
    te = te.dropna(subset=[target, "y_shift1"])
    if te.empty:
        return float("nan"), 0
    nse = _nse(te[target].to_numpy(), te["y_shift1"].to_numpy())
    return nse, int(len(te))


def _load_policy_map(policy_json: Path) -> Dict[str, List[str]]:
    if not policy_json.exists():
        return {}
    try:
        pol = json.loads(policy_json.read_text(encoding="utf-8"))
        fe = pol.get("feature_exclude", {}) or {}
        return {str(k): [str(c) for c in v] for k, v in fe.items()}
    except Exception:
        return {}


def _load_global_feature_order(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    feats = [str(x) for x in obj.get("feature_order", [])]
    if not feats:
        raise RuntimeError("global_feature_order.json no tiene 'feature_order'")
    return feats


def _load_global_importance(
    global_model_path: Path,
    feature_order: List[str],
    importance_type: str = "gain",
) -> List[Tuple[str, float]]:
    booster = xgb.Booster()
    booster.load_model(global_model_path)
    raw_imp = booster.get_score(importance_type=importance_type)
    imp: Dict[str, float] = {k: float(v[0] if isinstance(v, list) else v) for k, v in raw_imp.items()}
    mapped: List[Tuple[str, float]] = []
    for k, v in imp.items():
        try:
            idx = int(k[1:])  # 'f12' -> 12
            fname = feature_order[idx] if 0 <= idx < len(feature_order) else k
        except Exception:
            fname = k
        mapped.append((fname, float(v)))
    mapped.sort(key=lambda x: x[1], reverse=True)
    return mapped


# -------------------- Entrenamiento --------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Reentrena Bottom-N cuencas en modo local-lite con top-K features del global.")
    ap.add_argument("--data_dir", type=str, required=True, help="data/processed/features_parts")
    ap.add_argument("--metrics_csv", type=str, required=True, help="models/metrics_per_basin.csv (para escoger Bottom-N)")
    ap.add_argument("--global_model", type=str, required=True, help="models/xgb_global.json")
    ap.add_argument("--global_feature_order", type=str, required=True, help="models/global_feature_order.json")
    ap.add_argument("--policy_json", type=str, default="", help="models/diagnostics/gating_policy.json (opcional)")
    ap.add_argument("--split_date", type=str, default="2009-01-01")
    ap.add_argument("--n_bottom", type=int, default=20)
    ap.add_argument("--top_k", type=int, default=60)
    ap.add_argument("--out_dir", type=str, required=True, help="carpeta salida, p. ej. models/retrained")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_date = pd.Timestamp(args.split_date)

    # 1) seleccionar bottom-N
    m = pd.read_csv(args.metrics_csv)
    m["basin_id"] = m["basin_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "test_R2" not in m.columns:
        raise RuntimeError("metrics_per_basin.csv no tiene test_R2.")
    bottom = m.sort_values("test_R2", ascending=True).head(args.n_bottom).copy()

    # 2) cargar global: orden de features y su importancia
    feature_order = _load_global_feature_order(Path(args.global_feature_order))
    # (no usamos basin_te en local-lite)
    imp = _load_global_importance(Path(args.global_model), feature_order)
    top_imp = [(c, g) for (c, g) in imp if c != "basin_te"][: args.top_k]
    topK: List[str] = [c for (c, _) in top_imp]

    (out_dir / "top_features_global.json").write_text(
        json.dumps({"topK": topK, "importance_type": "gain"}, indent=2),
        encoding="utf-8",
    )

    # 3) exclusiones por cuenca (opcional)
    feature_exclude_map = _load_policy_map(Path(args.policy_json)) if args.policy_json else {}

    # 4) loop de reentrenamiento
    params = _xgb_params()
    rows_out: List[RowOut] = []

    for _, r in bottom.iterrows():
        bid = str(r["basin_id"])
        fpath = data_dir / f"features_{bid}.parquet"
        if not fpath.exists():
            print(f"[skip] no encontrado {fpath}")
            continue

        df = pd.read_parquet(fpath)
        if DATE_COL not in df.columns:
            print(f"[skip] {bid} sin columna date")
            continue
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        if ID_COL in df.columns:
            df[ID_COL] = df[ID_COL].astype(str)

        target = _pick_target(list(df.columns))
        try:
            tr, te = _split(df, target, split_date)
        except ValueError:
            print(f"[skip] {bid} split vacío")
            continue

        # columnas candidatas = topK presentes
        feats = [c for c in topK if c in df.columns]
        # aplicar exclusiones por cuenca
        drop_cols = set(feature_exclude_map.get(bid, []))
        if drop_cols:
            feats = [c for c in feats if c not in drop_cols]

        if len(feats) < 3:
            print(f"[skip] {bid} <3 features tras exclusiones.")
            continue

        # imputación por reglas (mediana en TRAIN para las no-zero)
        _impute_rules_fill(tr, feats, ref=tr)
        _impute_rules_fill(te, feats, ref=tr)

        tr = tr.dropna(subset=[target])
        te = te.dropna(subset=[target])
        if tr.empty or te.empty:
            print(f"[skip] {bid} vacío tras imputación/target.")
            continue

        # ---- Conversión segura a arreglos NumPy de float ----
        Xtr: NDArray[np.float32] = np.asarray(tr[feats].to_numpy(), dtype=np.float32)
        ytr: NDArray[np.float32] = np.asarray(tr[target].to_numpy(), dtype=np.float32)
        Xte: NDArray[np.float32] = np.asarray(te[feats].to_numpy(), dtype=np.float32)
        yte: NDArray[np.float32] = np.asarray(te[target].to_numpy(), dtype=np.float32)

        model = xgb.XGBRegressor(**params)
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

        yhat_tr: NDArray[np.float64] = np.asarray(model.predict(Xtr), dtype=np.float64)
        yhat_te: NDArray[np.float64] = np.asarray(model.predict(Xte), dtype=np.float64)

        # métricas en espacio del target
        def _metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
            yt: NDArray[np.float64] = np.asarray(y_true, dtype=np.float64)
            yp: NDArray[np.float64] = np.asarray(y_pred, dtype=np.float64)
            mae = float(np.mean(np.abs(yt - yp)))
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            logNSE = r2 if target == "y_log1p" else float("nan")
            return {"MAE": mae, "RMSE": rmse, "R2": r2, "logNSE": logNSE}

        mtr = _metrics(ytr, yhat_tr)
        mte = _metrics(yte, yhat_te)

        # baseline: Persistencia en TEST
        pers_nse, n_pers = _persistence_on_test(df[[DATE_COL, target]].copy(), target, split_date)

        # guardar modelo
        out_model = out_dir / "basins_local_lite"
        out_model.mkdir(parents=True, exist_ok=True)
        outp = out_model / f"xgb_basin_{bid}_lite.json"
        model.save_model(outp)

        rows_out.append(
            {
                "basin_id": bid,
                "n_features": len(feats),
                "features_used": feats,  # List[str] permitido por RowOut
                "train_MAE": mtr["MAE"],
                "train_RMSE": mtr["RMSE"],
                "train_R2": mtr["R2"],
                "train_logNSE": mtr["logNSE"],
                "test_MAE": mte["MAE"],
                "test_RMSE": mte["RMSE"],
                "test_R2": mte["R2"],
                "test_logNSE": mte["logNSE"],
                "persistence_NSE": pers_nse,
                "persistence_N": n_pers,
                "suggested_serving": "global"
                if (mte["R2"] < 0 and (pers_nse if not np.isnan(pers_nse) else -1.0) > 0)
                else "local_lite",
                "model_path": str(outp),
            }
        )

        print(f"[ok] {bid} reentrenada local-lite con {len(feats)} features → {outp}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_dir / "retrained_metrics.csv", index=False)

    # comparativa contra original
    comp = pd.merge(
        m[["basin_id", "test_R2", "test_RMSE", "test_MAE"]],
        out_df[["basin_id", "test_R2", "test_RMSE", "test_MAE", "persistence_NSE", "suggested_serving"]],
        on="basin_id",
        suffixes=("_orig", "_lite"),
        how="right",
    )
    comp.to_csv(out_dir / "retrained_compare.csv", index=False)

    print(f"✔ retrained_metrics.csv → {out_dir / 'retrained_metrics.csv'}")
    print(f"✔ retrained_compare.csv → {out_dir / 'retrained_compare.csv'}")


if __name__ == "__main__":
    main()
