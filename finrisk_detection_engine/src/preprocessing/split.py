
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when a preprocessing step fails."""


def time_based_split(
    df: pd.DataFrame,
    timestamp_col: str = "payment_timestamp",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if timestamp_col not in df.columns:
        raise PreprocessingError(f"Missing timestamp column for split: {timestamp_col}")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    cutoff_idx = int(len(ordered) * (1 - test_size))
    cutoff_ts = ordered.loc[cutoff_idx, timestamp_col]

    train_df = ordered.iloc[:cutoff_idx].copy()
    test_df = ordered.iloc[cutoff_idx:].copy()

    logger.info(
        "Time-based split: train=%d rows, test=%d rows, cutoff=%s",
        len(train_df), len(test_df), str(cutoff_ts),
    )

    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "payment_id": range(10),
        "payment_timestamp": pd.date_range("2026-01-01", periods=10, freq="D")
    })
    train, test = time_based_split(dummy, test_size=0.3)
    print(f"train rows: {len(train)}, test rows: {len(test)}")
    print(train.tail(2))
    print(test.head(2))