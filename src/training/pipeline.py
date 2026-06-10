"""MVP training pipeline: preprocess → time split → LR + CatBoost."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import basic_preprocess
from src.data.split import time_split
from src.models.catboost_model import train_catboost
from src.models.logistic import train_logistic_regression
from src.training.metrics import print_metrics
from src.training.prepare import (
    get_categorical_columns,
    get_feature_columns,
    get_numeric_columns,
)


def run_training(
    merged_path: str | Path = "data/merged/train_merged.parquet",
    *,
    train_size: float = 0.8,
    models_dir: str | Path = "models",
) -> dict:
    """Load merged train data, split by time, train LR and CatBoost baselines.

    Args:
        merged_path: Path to train merged Parquet file.
        train_size: Fraction of rows for training (chronological split).
        models_dir: Directory to save trained models.

    Returns:
        Dict with validation metrics for each model.
    """
    merged_path = Path(merged_path)
    models_dir = Path(models_dir)

    print(f"Loading {merged_path} ...")
    df = pd.read_parquet(merged_path)
    df = basic_preprocess(df)

    train_df, val_df = time_split(df, train_size=train_size)
    print(f"train: {train_df.shape}, val: {val_df.shape}")

    feature_cols = get_feature_columns(train_df)
    categorical_cols = get_categorical_columns(train_df, feature_cols)
    numeric_cols = get_numeric_columns(train_df, feature_cols)

    print(f"features: {len(feature_cols)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")

    _, lr_metrics = train_logistic_regression(
        train_df,
        val_df,
        numeric_cols,
        model_path=models_dir / "logistic_regression.joblib",
    )
    print_metrics("Logistic Regression (numeric features)", lr_metrics)

    _, cb_metrics = train_catboost(
        train_df,
        val_df,
        feature_cols,
        categorical_cols,
        model_path=models_dir / "catboost.cbm",
    )
    print_metrics("CatBoost (numeric + categorical)", cb_metrics)

    results = {
        "logistic_regression": lr_metrics,
        "catboost": cb_metrics,
    }
    return results


if __name__ == "__main__":
    run_training()
