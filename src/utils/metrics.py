"""Evaluation metrics for LTV classification."""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    thresholds: List[float] = None,
) -> Dict:
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.5]

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }

    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        if pred.sum() == 0:
            continue
        metrics[f"precision@{thr}"] = float(precision_score(y_true, pred, zero_division=0))
        metrics[f"recall@{thr}"] = float(recall_score(y_true, pred, zero_division=0))
        metrics[f"f1@{thr}"] = float(f1_score(y_true, pred, zero_division=0))

    return metrics


def precision_at_k(y_true: pd.Series | np.ndarray, proba: np.ndarray, k: int) -> float:
    idx = np.argsort(proba)[::-1][:k]
    if hasattr(y_true, "iloc"):
        return float(y_true.iloc[idx].mean())
    return float(y_true[idx].mean())


def compute_lift_table(
    y_true: pd.Series, proba: np.ndarray, ks: List[int] = None
) -> pd.DataFrame:
    if ks is None:
        ks = [100, 500, 1000, 2000, 5000]
    base_rate = y_true.mean()
    rows = []
    for k in ks:
        p_at_k = precision_at_k(y_true, proba, k)
        rows.append(
            {
                "k": k,
                "precision_at_k": p_at_k,
                "lift": p_at_k / base_rate if base_rate > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)
