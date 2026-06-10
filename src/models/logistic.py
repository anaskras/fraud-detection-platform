"""Logistic regression baseline (numeric features only)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.metrics import evaluate_probs


def train_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    target: str = "isFraud",
    model_path: str | Path = "models/logistic_regression.joblib",
) -> tuple[Pipeline, dict]:
    """Train a scaled logistic regression model and evaluate on validation data."""
    X_train = train_df[numeric_cols]
    y_train = train_df[target].values
    X_val = val_df[numeric_cols]
    y_val = val_df[target].values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    # n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_probs(y_val, val_prob)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": numeric_cols}, model_path)

    return model, metrics
