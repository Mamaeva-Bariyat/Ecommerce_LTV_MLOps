"""Config loading test."""
from pathlib import Path

from src.utils.io import load_config


def test_load_config():
    cfg = load_config("config/config.yaml")
    assert "project" in cfg
    assert "model" in cfg
    assert "paths" in cfg
    assert cfg["project"]["name"] == "Ecommerce_LTV_MLOps"
