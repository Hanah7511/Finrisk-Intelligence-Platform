from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Scales MAD to be comparable to a standard deviation under a normal
# distribution, so the same "3.5 = extreme" intuition used for Z-scores
# applies here. Standard value (Iglewicz & Hoya, 1993).
MAD_CONSISTENCY_CONSTANT: float = 1.4826


def compute_double_mad_stats(
    train_df: pd.DataFrame,
    col: str,
) -> Tuple[float, float, float]:
    """Fit median + separate above/below-median MAD on train data only.

    Plain MAD assumes symmetric spread around the median, which breaks down
    on right-skewed columns like transaction amount (dense cluster of small
    values, sparse tail of large ones). Splitting into two MADs lets each
    side of the distribution set its own "normal spread" instead of forcing
    one number onto both. Fit on train only - same leakage rule as
    fill_missing_values/log_transform_skewed in src/preprocessing/.
    """
    values = train_df[col].dropna()
    median = values.median()

    below = values[values <= median]
    above = values[values >= median]

    mad_below = (median - below).median()
    mad_above = (above - median).median()

    logger.info(
        "Fit double-MAD stats for %s: median=%.4f, mad_below=%.4f, mad_above=%.4f",
        col, median, mad_below, mad_above,
    )
    return median, mad_below, mad_above


def double_mad_score(
    values: pd.Series,
    median: float,
    mad_below: float,
    mad_above: float,
    k: float = MAD_CONSISTENCY_CONSTANT,
) -> pd.Series:
    """Vectorized modified Z-score, using the side-appropriate MAD.

    A MAD of 0 (degenerate/constant column on one side) would divide by
    zero - those rows score 0 instead of inf/NaN, since no variation on
    that side means nothing on that side can be an outlier.
    """
    mad_side = np.where(values <= median, mad_below, mad_above)
    denom = k * mad_side
    safe_denom = np.where(denom == 0, 1, denom)
    score = np.where(denom == 0, 0.0, (values - median) / safe_denom)
    return pd.Series(score, index=values.index)


def apply_double_mad(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    median: float,
    mad_below: float,
    mad_above: float,
    threshold: float,
    k: float = MAD_CONSISTENCY_CONSTANT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score train and test against stats already fit on train (see
    compute_double_mad_stats). Adds '{col}_mad_score' (signed modified
    Z-score) and '{col}_mad_flag' (1 if |score| >= threshold).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    score_col = f"{col}_mad_score"
    flag_col = f"{col}_mad_flag"

    train_df[score_col] = double_mad_score(train_df[col], median, mad_below, mad_above, k)
    test_df[score_col] = double_mad_score(test_df[col], median, mad_below, mad_above, k)

    train_df[flag_col] = (train_df[score_col].abs() >= threshold).astype(int)
    test_df[flag_col] = (test_df[score_col].abs() >= threshold).astype(int)

    logger.info(
        "Double-MAD flagged %d/%d train rows and %d/%d test rows on %s (threshold=%.2f)",
        train_df[flag_col].sum(), len(train_df),
        test_df[flag_col].sum(), len(test_df),
        col, threshold,
    )
    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rng = np.random.default_rng(42)
    dummy = pd.DataFrame({
        "amount_usd_equivalent": np.concatenate([
            rng.exponential(2, 90),      # dense low-value cluster
            rng.uniform(500, 3000, 10),  # sparse high-value tail
        ]),
    })
    train = dummy.iloc[:70].reset_index(drop=True)
    test = dummy.iloc[70:].reset_index(drop=True)

    median, mad_below, mad_above = compute_double_mad_stats(train, "amount_usd_equivalent")
    train_scored, test_scored = apply_double_mad(
        train, test, "amount_usd_equivalent", median, mad_below, mad_above, threshold=3.5,
    )
    print(train_scored[["amount_usd_equivalent", "amount_usd_equivalent_mad_score", "amount_usd_equivalent_mad_flag"]]
          .sort_values("amount_usd_equivalent_mad_score", ascending=False).head(10))
    print(test_scored[["amount_usd_equivalent", "amount_usd_equivalent_mad_score", "amount_usd_equivalent_mad_flag"]])
