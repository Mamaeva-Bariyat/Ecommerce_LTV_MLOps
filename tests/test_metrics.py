"""Basic unit tests for metrics."""
import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import compute_classification_metrics, precision_at_k


def test_precision_at_k():
    y = pd.Series([0, 0, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.3, 0.8])
    assert precision_at_k(y, proba, k=2) == 1.0
    assert precision_at_k(y, proba, k=5) == 0.4


def test_compute_metrics():
    y = np.array([0, 0, 1, 1, 0])
    proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3])
    m = compute_classification_metrics(y, proba)
    assert "roc_auc" in m
    assert "pr_auc" in m
    assert 0 <= m["roc_auc"] <= 1
