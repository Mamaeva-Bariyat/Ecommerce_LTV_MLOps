"""Download Brazilian E-Commerce (Olist) dataset from Kaggle."""
import os
import zipfile
from pathlib import Path

from src.utils.io import load_config


def download_olist_data(config_path: str = "config/config.yaml") -> None:
    """
    Requires:
    - kaggle package installed
    - ~/.kaggle/kaggle.json configured
    """
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = config["data"]["kaggle_dataset"]
    print(f"Downloading {dataset} → {raw_dir}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=str(raw_dir), unzip=True)
        print("Download & unzip completed.")
    except Exception as e:
        print(
            f"Kaggle download failed: {e}\n"
            "Manual steps:\n"
            "1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n"
            "2. Download and place CSVs into data/raw/"
        )
        raise


if __name__ == "__main__":
    download_olist_data()
