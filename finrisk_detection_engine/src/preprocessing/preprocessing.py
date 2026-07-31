from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.preprocessing.report import PreprocessingReport
from src.preprocessing.split import time_based_split
from src.preprocessing.missing_values import (
    drop_missing_timestamps,
    fill_missing_values,
    NUMERIC_IMPUTE_COLS,
    CATEGORICAL_IMPUTE_COLS,
)
from src.preprocessing.encoding import (
    label_encode_ordinal,
    one_hot_encode,
    frequency_encode,
    ORDINAL_LEVELS,
    ONEHOT_COLS,
    FREQUENCY_COLS,
)
from src.preprocessing.outlier_handling import log_transform_skewed, LOG_TRANSFORM_COLS
from src.preprocessing.timestamp_features import add_timestamp_features
from src.feature_engineering.feature_engineering import run_feature_engineering

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, PreprocessingReport]:
    report = PreprocessingReport()
    report.rows_before_split = len(df)

    cleaned = drop_missing_timestamps(df)
    report.rows_dropped_missing_timestamp = report.rows_before_split - len(cleaned)

    engineered, fe_report = run_feature_engineering(cleaned)
    report.history_features_recomputed = fe_report.history_features_recomputed
    report.derived_features_added = fe_report.derived_features_added

    train_df, test_df = time_based_split(engineered)
    report.train_rows = len(train_df)
    report.test_rows = len(test_df)

    train_df, test_df = fill_missing_values(train_df, test_df)
    report.numeric_cols_filled = [c for c in NUMERIC_IMPUTE_COLS if c in train_df.columns]
    report.categorical_cols_filled = [c for c in CATEGORICAL_IMPUTE_COLS if c in train_df.columns]

    train_df, test_df = label_encode_ordinal(train_df, test_df)
    report.ordinal_encoded_cols = [c for c in ORDINAL_LEVELS if c in train_df.columns]

    onehot_present = [c for c in ONEHOT_COLS if c in train_df.columns]
    train_df, test_df = one_hot_encode(train_df, test_df)
    report.onehot_encoded_cols = onehot_present

    train_df, test_df = frequency_encode(train_df, test_df)
    report.frequency_encoded_cols = [c for c in FREQUENCY_COLS if c in train_df.columns]

    # Preserve raw amount before log-transform. Layer 6 (statistical
    # detection) needs genuine dollar-magnitude deviation to flag outliers -
    # log1p() below deliberately compresses the tail double-MAD is trying to
    # detect. Confirmed empirically: scoring the log-transformed column
    # flagged 0/2000 real rows, vs. ~14% expected on raw dollar amounts.
    train_df["amount_usd_equivalent_raw"] = train_df["amount_usd_equivalent"]
    test_df["amount_usd_equivalent_raw"] = test_df["amount_usd_equivalent"]

    train_df, test_df = log_transform_skewed(train_df, test_df)
    report.log_transformed_cols = [c for c in LOG_TRANSFORM_COLS if c in train_df.columns]

    train_df = add_timestamp_features(train_df)
    test_df = add_timestamp_features(test_df)
    report.timestamp_features_added = [
        "transaction_hour", "day_of_week", "day_of_month", "month",
        "is_weekend", "is_night", "is_business_hour",
    ]

    return train_df, test_df, report


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 30

    dummy = pd.DataFrame({
        "payment_id": range(n),
        "customer_id": rng.choice(["CUST_1", "CUST_2", "CUST_3"], n),
        "payment_timestamp": pd.date_range("2026-01-01", periods=n, freq="6h"),
        "account_age_days": rng.integers(10, 900, n).astype(float),
        "instrument_age_days": rng.integers(1, 500, n).astype(float),
        "unique_merchants": rng.integers(1, 4, n),
        "unique_products": rng.integers(1, 5, n),
        "item_count": rng.integers(1, 6, n),
        "ip_risk_score": rng.uniform(0, 1, n),
        "amount_risk_score": rng.uniform(0, 1, n),
        "device_risk_score": rng.uniform(0, 1, n),
        "kyc_level": rng.choice(["LOW", "MEDIUM", "HIGH"], n),
        "geo_risk_level": rng.choice(["LOW", "MEDIUM", "HIGH"], n),
        "customer_risk_segment": rng.choice(["LOW", "MEDIUM", "HIGH"], n),
        "payment_method": rng.choice(["CARD", "UPI", "NETBANKING"], n),
        "transaction_channel": rng.choice(["WEB", "MOBILE"], n),
        "transaction_direction": rng.choice(["INBOUND", "OUTBOUND"], n),
        "payment_status": rng.choice(["success", "failed"], n),
        "counterparty_country": rng.choice(["India", "Singapore", "Germany"], n),
        "device_id": rng.choice(["DEV_1", "DEV_2", "DEV_3"], n),
        "amount_usd_equivalent": rng.exponential(500, n),
    })
    dummy.loc[3, "account_age_days"] = np.nan
    dummy.loc[5, "kyc_level"] = None
    dummy.loc[7, "payment_timestamp"] = pd.NaT

    train_df, test_df, report = run_preprocessing(dummy)

    report.summary()
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
