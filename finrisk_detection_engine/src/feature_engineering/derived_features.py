from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DERIVED_FEATURE_NAMES: List[str] = [
    "amount_vs_customer_avg_ratio",
    "txn_frequency_ratio_24h_7d",
    "failed_payment_ratio_7d",
    "product_diversity_ratio",
    "merchant_diversity_ratio",
    "account_instrument_age_ratio",
    "combined_risk_score",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).fillna(fill)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"amount_usd_equivalent", "customer_avg_amount_30d"}.issubset(df.columns):
        df["amount_vs_customer_avg_ratio"] = _safe_ratio(
            df["amount_usd_equivalent"], df["customer_avg_amount_30d"], fill=1.0
        )

    if {"transaction_frequency_24h", "transaction_frequency_7d"}.issubset(df.columns):
        df["txn_frequency_ratio_24h_7d"] = _safe_ratio(
            df["transaction_frequency_24h"], df["transaction_frequency_7d"], fill=0.0
        )

    if {"customer_failed_payments_7d", "customer_txn_count_30d"}.issubset(df.columns):
        df["failed_payment_ratio_7d"] = _safe_ratio(
            df["customer_failed_payments_7d"], df["customer_txn_count_30d"], fill=0.0
        )

    if {"unique_products", "item_count"}.issubset(df.columns):
        df["product_diversity_ratio"] = _safe_ratio(df["unique_products"], df["item_count"], fill=1.0)

    if {"unique_merchants", "item_count"}.issubset(df.columns):
        df["merchant_diversity_ratio"] = _safe_ratio(df["unique_merchants"], df["item_count"], fill=1.0)

    if {"account_age_days", "instrument_age_days"}.issubset(df.columns):
        df["account_instrument_age_ratio"] = _safe_ratio(
            df["account_age_days"], df["instrument_age_days"], fill=1.0
        )

    risk_cols = [c for c in ["ip_risk_score", "amount_risk_score", "device_risk_score"] if c in df.columns]
    if risk_cols:
        df["combined_risk_score"] = df[risk_cols].mean(axis=1)

    logger.info("Added derived features.")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "amount_usd_equivalent": [100.0, 500.0],
        "customer_avg_amount_30d": [50.0, 0.0],
        "transaction_frequency_24h": [2, 0],
        "transaction_frequency_7d": [5, 0],
        "customer_failed_payments_7d": [1, 0],
        "customer_txn_count_30d": [4, 0],
        "unique_products": [2, 1],
        "item_count": [4, 1],
        "unique_merchants": [1, 1],
        "account_age_days": [200, 10],
        "instrument_age_days": [50, 0],
        "ip_risk_score": [0.2, 0.5],
        "amount_risk_score": [0.3, 0.6],
        "device_risk_score": [0.1, 0.4],
    })
    print(add_derived_features(dummy))
