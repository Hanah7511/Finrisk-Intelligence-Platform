from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from configs.loader import load_thresholds_config
from src.statistical_detection.report import StatisticalDetectionReport
from src.statistical_detection.double_mad import apply_double_mad, compute_double_mad_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_config = load_thresholds_config()["statistical"]
MAD_THRESHOLD: float = _config["mad_threshold"]

# Scope deliberately limited to amount_usd_equivalent - the only field that
# needs no rolling/windowed computation and is the natural "is this dollar
# figure unusual" check. Frequency/behavioral columns (transaction_frequency_*,
# customer_avg_amount_30d) are intentionally excluded: they're already the
# columns Layer 5's velocity rule scores, so double-MAD over them too would
# duplicate signal rather than add new detection coverage.
#
# Note: on this dataset the fitted mad_below is small enough that the low-side
# flag boundary falls below $0 - amounts can never be flagged as "too small."
# This layer is effectively a one-sided "unusually large amount" detector in
# practice, which is the correct behavior for AML (small amounts aren't the
# risk signal), not a bug to fix.
AMOUNT_COL = "amount_usd_equivalent_raw"


def run_statistical_detection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StatisticalDetectionReport]:
    report = StatisticalDetectionReport()
    report.train_rows = len(train_df)
    report.test_rows = len(test_df)
    report.threshold = MAD_THRESHOLD

    median, mad_below, mad_above = compute_double_mad_stats(train_df, AMOUNT_COL)
    report.median = median
    report.mad_below = mad_below
    report.mad_above = mad_above

    train_df, test_df = apply_double_mad(
        train_df, test_df, AMOUNT_COL, median, mad_below, mad_above, threshold=MAD_THRESHOLD,
    )

    flag_col = f"{AMOUNT_COL}_mad_flag"
    # Orchestrator-level combined flag - currently mirrors the single amount
    # flag since it's the only statistical check. Kept as its own column so a
    # second statistical column can be added later without Layer 9's fusion
    # engine needing to change what it reads.
    train_df["statistical_anomaly_flag"] = train_df[flag_col]
    test_df["statistical_anomaly_flag"] = test_df[flag_col]

    report.train_flagged_count = int(train_df[flag_col].sum())
    report.test_flagged_count = int(test_df[flag_col].sum())

    return train_df, test_df, report


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    dummy = pd.DataFrame({
        "amount_usd_equivalent": np.concatenate([
            rng.exponential(2, 90),
            rng.uniform(500, 3000, 10),
        ]),
    })
    train = dummy.iloc[:70].reset_index(drop=True)
    test = dummy.iloc[70:].reset_index(drop=True)

    train_out, test_out, report = run_statistical_detection(train, test)
    report.summary()
    print(train_out[["amount_usd_equivalent", "statistical_anomaly_flag"]]
          .sort_values("amount_usd_equivalent", ascending=False).head(10))
