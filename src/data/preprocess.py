"""
Full preprocessing pipeline for Olist Brazilian E-Commerce dataset.

Builds customer-level feature table with:
- will_return (binary target: customer made >1 order)
- future_ltv (sum of payment value of subsequent orders)
- RFM-style and order/review/payment features based on FIRST order
  (to avoid target leakage)

Output: data/processed/features_.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_config


RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def _load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    missing = []
    data = {}
    for key, fname in RAW_FILES.items():
        path = raw_dir / fname
        if not path.exists():
            missing.append(fname)
            continue
        data[key] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError(
            f"Missing files in {raw_dir}: {missing}\n"
            "Run: python -m src.data.download"
        )
    return data


def _prepare_orders(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"].copy()
    customers = data["customers"][
        ["customer_id", "customer_unique_id", "customer_state", "customer_city"]
    ].copy()

    orders = orders.merge(customers, on="customer_id", how="left")

    valid_status = {"delivered", "shipped", "invoiced", "processing", "approved"}
    orders = orders[orders["order_status"].isin(valid_status)].copy()

    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0

    return orders


def _aggregate_payments(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pay = data["order_payments"].copy()
    agg = (
        pay.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            n_payments=("payment_sequential", "count"),
            payment_type_mode=(
                "payment_type",
                lambda s: s.mode().iloc[0] if len(s) else "unknown",
            ),
        )
        .reset_index()
    )
    return agg


def _aggregate_items(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    items = data["order_items"].copy()
    products = data.get("products")
    if products is not None:
        items = items.merge(
            products[
                [
                    "product_id",
                    "product_category_name",
                    "product_weight_g",
                    "product_photos_qty",
                ]
            ],
            on="product_id",
            how="left",
        )
        cat_tr = data.get("category_translation")
        if cat_tr is not None:
            items = items.merge(cat_tr, on="product_category_name", how="left")
            items["product_category"] = items["product_category_name_english"].fillna(
                items["product_category_name"]
            )
        else:
            items["product_category"] = items["product_category_name"]

    agg = (
        items.groupby("order_id")
        .agg(
            n_items=("order_item_id", "count"),
            n_sellers=("seller_id", "nunique"),
            price_sum=("price", "sum"),
            freight_sum=("freight_value", "sum"),
            weight_sum=("product_weight_g", "sum"),
            photos_mean=("product_photos_qty", "mean"),
            main_category=(
                "product_category",
                lambda s: s.mode().iloc[0] if len(s.dropna()) else "unknown",
            ),
        )
        .reset_index()
    )
    return agg


def _aggregate_reviews(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rev = data["order_reviews"].copy()
    agg = (
        rev.groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            n_reviews=("review_id", "count"),
            has_comment=(
                "review_comment_message",
                lambda s: int(s.notna().any()),
            ),
        )
        .reset_index()
    )
    return agg


def build_customer_features(raw_dir: Path) -> pd.DataFrame:
    data = _load_raw(raw_dir)
    orders = _prepare_orders(data)
    payments = _aggregate_payments(data)
    items = _aggregate_items(data)
    reviews = _aggregate_reviews(data)

    df = (
        orders.merge(payments, on="order_id", how="left")
        .merge(items, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
    )

    df = df.sort_values(["customer_unique_id", "order_purchase_timestamp"])

    df["order_rank"] = df.groupby("customer_unique_id").cumcount() + 1

    # Target from 2nd+ orders
    future = (
        df[df["order_rank"] > 1]
        .groupby("customer_unique_id")
        .agg(
            future_ltv=("payment_value", "sum"),
            n_future_orders=("order_id", "count"),
        )
        .reset_index()
    )

    # Features from FIRST order only (no leakage)
    first = df[df["order_rank"] == 1].copy()
    first = first.merge(future, on="customer_unique_id", how="left")
    first["future_ltv"] = first["future_ltv"].fillna(0.0)
    first["n_future_orders"] = first["n_future_orders"].fillna(0).astype(int)
    first["will_return"] = (first["n_future_orders"] > 0).astype(int)

    max_date = df["order_purchase_timestamp"].max()
    first["recency_days"] = (max_date - first["order_purchase_timestamp"]).dt.days

    first["monetary"] = first["payment_value"].fillna(first["price_sum"]).fillna(0.0)
    first["freight_ratio"] = first["freight_sum"] / (first["monetary"] + 1e-6)
    first["avg_item_price"] = first["price_sum"] / first["n_items"].clip(lower=1)
    first["is_late"] = (first["delivery_delay_days"] > 0).astype(float)
    first["delivery_delay_days"] = first["delivery_delay_days"].fillna(0.0)

    # Payment type dummies
    pay_dummies = pd.get_dummies(
        first["payment_type_mode"].fillna("unknown"), prefix="pay"
    )
    keep_pay = [
        c
        for c in pay_dummies.columns
        if c
        in (
            "pay_credit_card",
            "pay_boleto",
            "pay_voucher",
            "pay_debit_card",
            "pay_unknown",
        )
    ]
    if not keep_pay:
        keep_pay = list(pay_dummies.columns)[:4]
    first = pd.concat([first, pay_dummies[keep_pay]], axis=1)

    # State dummies (top 10 + OTHER)
    top_states = first["customer_state"].value_counts().nlargest(10).index.tolist()
    first["state_top"] = first["customer_state"].where(
        first["customer_state"].isin(top_states), other="OTHER"
    )
    state_dummies = pd.get_dummies(first["state_top"], prefix="state")
    first = pd.concat([first, state_dummies], axis=1)

    # Category dummies (top 12 + other)
    top_cats = first["main_category"].value_counts().nlargest(12).index.tolist()
    first["cat_top"] = first["main_category"].where(
        first["main_category"].isin(top_cats), other="other"
    )
    cat_dummies = pd.get_dummies(first["cat_top"], prefix="cat")
    first = pd.concat([first, cat_dummies], axis=1)

    id_col = "customer_unique_id"
    target_cols = ["will_return", "future_ltv"]

    base_feats = [
        "recency_days",
        "monetary",
        "n_items",
        "n_sellers",
        "price_sum",
        "freight_sum",
        "freight_ratio",
        "avg_item_price",
        "payment_installments",
        "n_payments",
        "weight_sum",
        "photos_mean",
        "review_score",
        "n_reviews",
        "has_comment",
        "delivery_delay_days",
        "is_late",
    ]
    # Exclude helper string columns state_top / cat_top (they are not dummies)
    dummy_feats = [
        c
        for c in first.columns
        if c.startswith(("pay_", "state_", "cat_"))
        and c not in ("state_top", "cat_top")
    ]
    feature_cols = [c for c in base_feats + dummy_feats if c in first.columns]
    # Keep only numeric columns
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(first[c])
    ]

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(first[c]):
            first[c] = first[c].fillna(
                first[c].median() if first[c].notna().any() else 0.0
            )
        else:
            first[c] = first[c].fillna(0)

    out = first[[id_col] + target_cols + feature_cols].copy()
    out = out.drop_duplicates(subset=[id_col])

    print(f"Customers: {len(out):,}")
    print(
        f"will_return = 1: {out['will_return'].sum():,} ({out['will_return'].mean():.2%})"
    )
    print(f"Features: {len(feature_cols)}")
    ret_mean = out.loc[out["will_return"] == 1, "future_ltv"].mean()
    print(f"Mean future_ltv (returners): {ret_mean:.2f}")

    return out


def run_preprocessing(config_path: str = "config/config.yaml") -> Path:
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_file = Path(config["paths"]["features_file"])

    print("Building customer features from Olist raw data...")
    features_df = build_customer_features(raw_dir)

    features_df.to_csv(features_file, index=False)
    print(f"Saved → {features_file}")
    return features_file


if __name__ == "__main__":
    run_preprocessing()
