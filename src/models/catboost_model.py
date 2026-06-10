"""CatBoost baseline (numeric + categorical features)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from src.training.metrics import evaluate_probs


def train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    *,
    target: str = "isFraud",
    model_path: str | Path = "models/catboost.cbm",
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 50,
) -> tuple[CatBoostClassifier, dict]:
    """Train CatBoost with early stopping and evaluate on validation data."""
    X_train = train_df[feature_cols]
    y_train = train_df[target].values
    X_val = val_df[feature_cols]
    y_val = val_df[target].values

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        early_stopping_rounds=early_stopping_rounds,
        random_seed=42,
        verbose=100,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols or None,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    val_prob = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_probs(y_val, val_prob)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    return model, metrics
