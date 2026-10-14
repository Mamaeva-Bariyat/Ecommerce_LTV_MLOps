"""Shared inference logic for the API."""
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.models.predict import predict_ltv
from src.utils.io import load_config, load_model


def get_model_artifact(model_path: str | Path = "models/ltv_ensemble.joblib") -> Dict[str, Any]:
    return load_model(model_path)


def score_dataframe(df: pd.DataFrame, model_path: str | Path = "models/ltv_ensemble.joblib") -> pd.DataFrame:
    return predict_ltv(df, model_path=model_path)
