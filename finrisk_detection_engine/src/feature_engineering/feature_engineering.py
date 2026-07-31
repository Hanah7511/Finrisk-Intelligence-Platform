from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.feature_engineering.report import FeatureEngineeringReport
from src.feature_engineering.customer_history_features import (
    compute_customer_history_features,
    HISTORY_FEATURE_NAMES,
)
from src.feature_engineering.derived_features import add_derived_features, DERIVED_FEATURE_NAMES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
    """Runs on the full, raw, time-sorted dataset — must happen before the train/test
    split and before any preprocessing encoding/log-transform (see customer_history_features
    docstring for why)."""
    report = FeatureEngineeringReport()
    report.rows_in = len(df)

    df = compute_customer_history_features(df)
    report.history_features_recomputed = HISTORY_FEATURE_NAMES

    df = add_derived_features(df)
    report.derived_features_added = DERIVED_FEATURE_NAMES

    report.rows_out = len(df)
    return df, report


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 40
    dummy = pd.DataFrame({
        "customer_id": rng.choice(["C1", "C2", "C3"], n),
        "payment_timestamp": pd.date_range("2026-01-01", periods=n, freq="8h"),
        "device_id": rng.choice(["DEV_1", "DEV_2", "DEV_3"], n),
        "counterparty_country": rng.choice(["India", "Singapore", "Germany"], n),
        "amount_usd_equivalent": rng.exponential(200, n),
        "payment_status": rng.choice(["success", "failed"], n, p=[0.85, 0.15]),
        "unique_merchants": rng.integers(1, 4, n),
        "unique_products": rng.integers(1, 5, n),
        "item_count": rng.integers(1, 6, n),
        "account_age_days": rng.integers(10, 900, n),
        "instrument_age_days": rng.integers(1, 500, n),
        "ip_risk_score": rng.uniform(0, 1, n),
        "amount_risk_score": rng.uniform(0, 1, n),
        "device_risk_score": rng.uniform(0, 1, n),
    })

    engineered, report = run_feature_engineering(dummy)
    report.summary()
    print(engineered[[
        "customer_id", "payment_timestamp", "customer_txn_count_30d",
        "customer_avg_amount_30d", "transaction_frequency_24h",
        "new_device_flag", "amount_vs_customer_avg_ratio",
    ]].head(15))
