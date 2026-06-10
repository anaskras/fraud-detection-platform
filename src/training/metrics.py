"""Offline evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute ROC-AUC, PR-AUC, and precision/recall at a threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
    }


def print_metrics(name: str, metrics: dict) -> None:
    """Print metrics dict in a readable format."""
    print(f"\n{name}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} @ threshold={metrics['threshold']}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
