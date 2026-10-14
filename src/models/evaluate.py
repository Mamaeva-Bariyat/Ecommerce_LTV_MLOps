"""Evaluation helpers (mostly used inside train.py)."""
from typing import Dict

import numpy as np
import pandas as pd

from src.utils.metrics import compute_classification_metrics, compute_lift_table


def evaluate_oof(
    y_true: pd.Series,
    oof_proba: np.ndarray,
) -> Dict:
    metrics = compute_classification_metrics(y_true, oof_proba)
    lift = compute_lift_table(y_true, oof_proba)
    return {"metrics": metrics, "lift": lift}
