"""Feature preparation for training."""

from __future__ import annotations

import pandas as pd

TARGET = "isFraud"
ID_COL = "TransactionID"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature column names (exclude ID and target)."""
    exclude = {ID_COL, TARGET}
    return [col for col in df.columns if col not in exclude]


def _is_numeric_feature(series: pd.Series) -> bool:
    """True for numeric and boolean columns usable by logistic regression."""
    return pd.api.types.is_numeric_dtype(series)


def get_categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return categorical feature names for CatBoost."""
    return [col for col in feature_cols if df[col].dtype == "object" or str(df[col].dtype) == "string"]


def get_numeric_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return numeric feature names for logistic regression."""
    return [col for col in feature_cols if _is_numeric_feature(df[col])]
