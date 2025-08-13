"""Evaluate the trained linear regression model."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DATA_PROCESSED, FEATURE_COLS, MODELS_DIR, TARGET_COL


def main() -> None:
    """Compute metrics on the processed dataset.

    If the model or dataset are missing, the script exits without error.
    """
    data_path = DATA_PROCESSED / "daily_samples.parquet"
    model_path = MODELS_DIR / "linear_model.json"
    if not data_path.exists() or not model_path.exists():
        print("Missing data or model, skipping evaluation.")
        return

    df = pd.read_parquet(data_path)
    X = df[FEATURE_COLS]
    y_true = df[TARGET_COL]

    with open(model_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    y_pred = X @ model_info["coef"] + model_info["intercept"]

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
