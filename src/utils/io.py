"""I/O helpers: config, data, models."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(config: Dict[str, Any]) -> pd.DataFrame:
    features_path = Path(config["paths"]["features_file"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}. "
            "Run preprocessing first or place features_.csv in data/processed/"
        )
    df = pd.read_csv(features_path)
    return df


def save_model(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"Model saved → {path}")


def load_model(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def get_feature_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    exclude = set(config["data"]["exclude_cols"])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        # Only numeric features (bool/int/float)
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
