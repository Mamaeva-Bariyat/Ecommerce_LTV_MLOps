"""End-to-end training pipeline."""
import argparse
from pathlib import Path

from src.data.download import download_olist_data
from src.data.preprocess import run_preprocessing
from src.models.train import train_ensemble
from src.utils.io import load_config


def run_pipeline(
    config_path: str = "config/config.yaml",
    download: bool = False,
    skip_train: bool = False,
) -> None:
    config = load_config(config_path)
    print(f"=== {config['project']['name']} v{config['project']['version']} ===\n")

    if download:
        print("1. Downloading data...")
        download_olist_data(config_path)
    else:
        print("1. Skipping download (use --download to fetch from Kaggle)")

    print("2. Building features from raw Olist data...")
    run_preprocessing(config_path)

    if skip_train:
        print("3. Skipping training")
        return

    print("3. Training ensemble...")
    train_ensemble(config_path)
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTV training pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--download", action="store_true", help="Download Olist data from Kaggle")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.config, download=args.download, skip_train=args.skip_train)
