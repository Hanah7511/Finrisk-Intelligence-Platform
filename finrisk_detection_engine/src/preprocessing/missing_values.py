from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_IMPUTE_COLS = [
    "account_age_days",
    "ip_risk_score",
    "instrument_age_days",
    "amount_usd_equivalent",
    "velocity_score",
    "amount_risk_score",
    "device_risk_score",
    "customer_avg_amount_30d",
]

CATEGORICAL_IMPUTE_COLS = [
    "kyc_level",
    "customer_risk_segment",
    "payment_status",
    "payment_method",
    "transaction_channel",
    "geo_risk_level",
    "transaction_direction",
    "counterparty_country",
]


def drop_missing_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "payment_timestamp",
) -> pd.DataFrame:
    before = len(df)
    cleaned = df.dropna(subset=[timestamp_col]).copy()
    dropped = before - len(cleaned)
    if dropped:
        logger.warning("Dropped %d rows with missing %s.", dropped, timestamp_col)
    return cleaned


def fill_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in NUMERIC_IMPUTE_COLS:
        if col not in train_df.columns:
            continue
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)
        logger.info("Filled missing %s with train median=%s", col, median)

    for col in CATEGORICAL_IMPUTE_COLS:
        if col not in train_df.columns:
            continue
        mode = train_df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
        train_df[col] = train_df[col].fillna(fill_value)
        test_df[col] = test_df[col].fillna(fill_value)
        logger.info("Filled missing %s with train mode=%s", col, fill_value)

    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "account_age_days": [100, None, 300],
        "kyc_level": ["HIGH", None, "LOW"],
    })
    train = dummy.iloc[:2]
    test = dummy.iloc[2:]
    train_filled, test_filled = fill_missing_values(train, test)
    print(train_filled)
    print(test_filled)