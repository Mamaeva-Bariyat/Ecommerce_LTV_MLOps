"""Inference helpers."""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.io import load_config, load_model


def predict_ltv(
    df: pd.DataFrame,
    model_path: str | Path = "models/ltv_ensemble.joblib",
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Add pred_proba, pred_ltv and ltv_decile columns to the input dataframe.
    Expects the same feature columns that were used at training time.
    """
    config = load_config(config_path)
    artifact = load_model(model_path)

    feature_cols = artifact["features"]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    X = df[feature_cols].astype(float)

    proba = (
        artifact["xgb"].predict_proba(X)[:, 1]
        + artifact["lgbm"].predict_proba(X)[:, 1]
        + artifact["cat"].predict_proba(X)[:, 1]
    ) / 3

    avg_return = artifact.get("avg_return_amount", config["data"]["avg_return_amount"])
    n_deciles = config["segmentation"]["n_deciles"]

    result = df.copy()
    result["pred_proba"] = proba
    result["pred_ltv"] = proba * avg_return
    result["ltv_decile"] = (
        pd.qcut(result["pred_ltv"], q=n_deciles, labels=False, duplicates="drop") + 1
    )
    return result
