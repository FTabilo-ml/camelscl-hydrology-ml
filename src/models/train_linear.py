"""Train a simple linear regression model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import DATA_PROCESSED, FEATURE_COLS, MODELS_DIR, TARGET_COL


def main() -> None:
    """Train LinearRegression on the processed dataset.

    If the expected ``daily_samples.parquet`` file is missing, the script
    exits gracefully without training.
    """
    data_path = DATA_PROCESSED / "daily_samples.parquet"
    if not data_path.exists():
        print("Processed dataset not found, skipping training.")
        return

    df = pd.read_parquet(data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    model = LinearRegression().fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "linear_model.json"
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump({
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_,
            "features": FEATURE_COLS,
        }, f, indent=2)


if __name__ == "__main__":
    main()
