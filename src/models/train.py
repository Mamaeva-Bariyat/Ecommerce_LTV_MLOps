"""Train ensemble of XGBoost + LightGBM + CatBoost for LTV classification."""
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.data.features import get_training_features
from src.utils.io import load_config, load_features, save_model, get_feature_columns
from src.utils.metrics import compute_classification_metrics, compute_lift_table


def _build_models(config: Dict, scale_pos_weight: float) -> Dict[str, Any]:
    cfg = config["model"]
    xgb = XGBClassifier(
        n_estimators=cfg["xgb"]["n_estimators"],
        max_depth=cfg["xgb"]["max_depth"],
        learning_rate=cfg["xgb"]["learning_rate"],
        scale_pos_weight=scale_pos_weight,
        subsample=cfg["xgb"]["subsample"],
        colsample_bytree=cfg["xgb"]["colsample_bytree"],
        random_state=config["project"]["seed"],
        eval_metric=cfg["xgb"]["eval_metric"],
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=cfg["lgbm"]["n_estimators"],
        max_depth=cfg["lgbm"]["max_depth"],
        learning_rate=cfg["lgbm"]["learning_rate"],
        is_unbalance=cfg["lgbm"]["is_unbalance"],
        subsample=cfg["lgbm"]["subsample"],
        colsample_bytree=cfg["lgbm"]["colsample_bytree"],
        random_state=config["project"]["seed"],
        verbose=-1,
        n_jobs=-1,
    )
    cat = CatBoostClassifier(
        iterations=cfg["catboost"]["iterations"],
        depth=cfg["catboost"]["depth"],
        learning_rate=cfg["catboost"]["learning_rate"],
        auto_class_weights=cfg["catboost"]["auto_class_weights"],
        random_state=config["project"]["seed"],
        verbose=0,
    )
    return {"xgb": xgb, "lgbm": lgbm, "cat": cat}


def train_ensemble(
    config_path: str = "config/config.yaml",
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
    config = load_config(config_path)
    df = load_features(config)

    X, y = get_training_features(df, config)
    feature_cols = list(X.columns)

    scale_pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    print(f"Samples: {len(X):,} | Positives: {y.sum()} ({y.mean():.2%})")
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    skf = StratifiedKFold(
        n_splits=config["model"]["n_folds"],
        shuffle=True,
        random_state=config["project"]["seed"],
    )

    oof_proba = np.zeros(len(X))
    fold_metrics = []
    last_models = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        models = _build_models(config, scale_pos_weight)
        for name, model in models.items():
            model.fit(X_tr, y_tr)

        proba = (
            models["xgb"].predict_proba(X_val)[:, 1]
            + models["lgbm"].predict_proba(X_val)[:, 1]
            + models["cat"].predict_proba(X_val)[:, 1]
        ) / 3

        oof_proba[val_idx] = proba
        m = compute_classification_metrics(y_val, proba)
        m["fold"] = fold
        fold_metrics.append(m)
        print(f"Fold {fold} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f}")

        if fold == config["model"]["n_folds"]:
            last_models = models

    metrics_df = pd.DataFrame(fold_metrics)
    print("\n=== Cross-validation summary ===")
    print(f"Mean ROC-AUC: {metrics_df['roc_auc'].mean():.4f} ± {metrics_df['roc_auc'].std():.4f}")
    print(f"Mean PR-AUC:  {metrics_df['pr_auc'].mean():.4f} ± {metrics_df['pr_auc'].std():.4f}")

    # Final OOF metrics + lift
    oof_metrics = compute_classification_metrics(y, oof_proba)
    lift_df = compute_lift_table(y, oof_proba)
    print("\n=== OOF metrics ===")
    print(oof_metrics)
    print("\n=== Lift table ===")
    print(lift_df.to_string(index=False))

    # Average return amount
    avg_return = config["data"]["avg_return_amount"]
    if config["data"]["ltv_col"] in df.columns:
        real_avg = df.loc[df[config["data"]["ltv_col"]] > 0, config["data"]["ltv_col"]].mean()
        if not np.isnan(real_avg):
            avg_return = float(real_avg)
            print(f"Using empirical avg_return_amount = {avg_return:.2f}")

    # Package for serving
    artifact = {
        "xgb": last_models["xgb"],
        "lgbm": last_models["lgbm"],
        "cat": last_models["cat"],
        "features": feature_cols,
        "avg_return_amount": avg_return,
        "oof_metrics": oof_metrics,
        "cv_metrics": metrics_df.to_dict(orient="records"),
        "lift_table": lift_df.to_dict(orient="records"),
        "config_snapshot": {
            "seed": config["project"]["seed"],
            "n_folds": config["model"]["n_folds"],
        },
    }

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(artifact, models_dir / "ltv_ensemble.joblib")

    # Also save predictions for analysis
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = df[[config["data"]["id_col"]]].copy() if config["data"]["id_col"] in df.columns else pd.DataFrame()
    out["pred_proba"] = oof_proba
    out["pred_ltv"] = oof_proba * avg_return
    out["ltv_decile"] = (
        pd.qcut(out["pred_ltv"], q=config["segmentation"]["n_deciles"], labels=False, duplicates="drop") + 1
    )
    out.to_csv(reports_dir / "oof_predictions.csv", index=False)
    metrics_df.to_csv(reports_dir / "cv_metrics.csv", index=False)
    lift_df.to_csv(reports_dir / "lift_table.csv", index=False)

    print(f"\nArtifacts saved to {models_dir} and {reports_dir}")
    return artifact, metrics_df, oof_proba


if __name__ == "__main__":
    train_ensemble()
