from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.rules.threshold_rules import get_applicable_threshold, STRUCTURING_NEAR_THRESHOLD_RATIO

logger = logging.getLogger(__name__)

WINDOW_24H = np.timedelta64(24, "h")


def _trailing_24h_sum_and_max(
    group: pd.DataFrame,
    amount_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    group = group.sort_values(timestamp_col)
    times = group[timestamp_col].to_numpy()
    amounts = group[amount_col].to_numpy(dtype=float)

    n = len(group)
    trailing_sum = np.zeros(n)
    trailing_max = np.zeros(n)

    start = 0
    for i in range(n):
        t = times[i]
        while start < i and times[start] < t - WINDOW_24H:
            start += 1
        # Includes the current transaction itself: structuring is about the total
        # moved in a trailing 24h window, evaluated as of this transaction.
        window_amounts = amounts[start:i + 1]
        trailing_sum[i] = window_amounts.sum()
        trailing_max[i] = window_amounts.max()

    group = group.copy()
    group["trailing_24h_amount_sum"] = trailing_sum
    group["trailing_24h_amount_max"] = trailing_max
    return group


def compute_trailing_24h_stats(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    amount_col: str = "amount_usd_equivalent",
    timestamp_col: str = "payment_timestamp",
) -> pd.DataFrame:
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    parts = [
        _trailing_24h_sum_and_max(g, amount_col, timestamp_col)
        for _, g in df.groupby(customer_col, sort=False)
    ]
    return pd.concat(parts).sort_values(timestamp_col).reset_index(drop=True)


def apply_structuring_rule(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    amount_col: str = "amount_usd_equivalent",
    country_col: str = "counterparty_country",
    timestamp_col: str = "payment_timestamp",
) -> pd.DataFrame:
    """Two structuring signals:
    1. near_threshold_flag - a single transaction sized suspiciously close to (90-100%
       of) its jurisdiction's reporting threshold.
    2. structuring_flag - the actual legal definition (31 U.S.C. 5324): a customer's
       trailing-24h total exceeds the jurisdiction threshold while no single transaction
       in that window does - a large sum broken into smaller reportable-evading pieces.
    """
    df = compute_trailing_24h_stats(df, customer_col, amount_col, timestamp_col)

    applicable_threshold = df[country_col].map(get_applicable_threshold)
    df["applicable_threshold_usd"] = applicable_threshold

    near_threshold_ratio = df[amount_col] / applicable_threshold
    df["near_threshold_flag"] = (
        (near_threshold_ratio >= STRUCTURING_NEAR_THRESHOLD_RATIO) & (near_threshold_ratio < 1.0)
    ).astype(int)

    df["structuring_flag"] = (
        (df["trailing_24h_amount_sum"] > applicable_threshold)
        & (df["trailing_24h_amount_max"] <= applicable_threshold)
    ).astype(int)

    logger.info(
        "Structuring rule: %d near-threshold, %d structuring flags out of %d rows.",
        df["near_threshold_flag"].sum(), df["structuring_flag"].sum(), len(df),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2"],
        "payment_timestamp": pd.to_datetime([
            "2026-01-01 08:00", "2026-01-01 14:00", "2026-01-01 20:00", "2026-01-01 09:00",
        ]),
        "amount_usd_equivalent": [4000.0, 4000.0, 4000.0, 9800.0],
        "counterparty_country": ["United States", "United States", "United States", "United States"],
    })
    print(apply_structuring_rule(dummy)[[
        "customer_id", "amount_usd_equivalent", "trailing_24h_amount_sum",
        "trailing_24h_amount_max", "near_threshold_flag", "structuring_flag",
    ]])
