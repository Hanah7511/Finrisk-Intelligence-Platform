from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 5
BUSINESS_START_HOUR = 9
BUSINESS_END_HOUR = 17


def add_timestamp_features(
    df: pd.DataFrame,
    timestamp_col: str = "payment_timestamp",
) -> pd.DataFrame:
    df = df.copy()

    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_col}")

    ts = df[timestamp_col]

    if "transaction_hour" in df.columns:
        hour = df["transaction_hour"]
    else:
        hour = ts.dt.hour
        df["transaction_hour"] = hour

    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"] = ((hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)).astype(int)
    df["is_business_hour"] = (
        (hour >= BUSINESS_START_HOUR) & (hour < BUSINESS_END_HOUR) & (df["is_weekend"] == 0)
    ).astype(int)

    logger.info(
        "Added timestamp features: day_of_week, day_of_month, month, is_weekend, is_night, is_business_hour"
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = pd.DataFrame({
        "payment_timestamp": pd.to_datetime([
            "2026-07-06 02:00:00",
            "2026-07-06 14:00:00",
            "2026-07-11 20:00:00",
        ]),
    })
    result = add_timestamp_features(df)
    print(result)
