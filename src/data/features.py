"""
Feature helpers used at training and inference time.
"""
from typing import List, Tuple

import pandas as pd

from src.utils.io import get_feature_columns


def build_features(df: pd.DataFrame, config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    At inference the caller must supply the same engineered columns
    that were produced by preprocess.build_customer_features.
    This function is a light pass-through.
    """
    return df


def get_training_features(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_columns(df, config)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y = df[config["data"]["target_col"]]
    return X, y
