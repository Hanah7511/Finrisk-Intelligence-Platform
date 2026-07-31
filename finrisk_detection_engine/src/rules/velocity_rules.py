from __future__ import annotations

import logging

import pandas as pd

from configs.loader import load_thresholds_config

logger = logging.getLogger(__name__)

_config = load_thresholds_config()["rules"]
FREQUENCY_24H_THRESHOLD: float = _config["transaction_frequency_24h_threshold"]
VELOCITY_SCORE_THRESHOLD: float = _config["velocity_score_threshold"]


def compute_velocity_score(
    df: pd.DataFrame,
    freq_24h_col: str = "transaction_frequency_24h",
    freq_7d_col: str = "transaction_frequency_7d",
) -> pd.DataFrame:
    """Rebuilds velocity_score as a genuine 0-1 risk score from Layer 4's real frequency
    features, instead of the dataset's original stand-in value. Weighted toward the
    tighter 24h window (a recent burst matters more than a week-long average), normalized
    against the existing configured frequency_24h_threshold.
    """
    df = df.copy()

    normalized_24h = (df[freq_24h_col] / FREQUENCY_24H_THRESHOLD).clip(upper=1.0)
    normalized_7d = (df[freq_7d_col] / (FREQUENCY_24H_THRESHOLD * 7)).clip(upper=1.0)

    df["velocity_score"] = (0.7 * normalized_24h + 0.3 * normalized_7d).round(4)
    df["velocity_flag"] = (df["velocity_score"] >= VELOCITY_SCORE_THRESHOLD).astype(int)

    logger.info(
        "Velocity rule flagged %d/%d transactions above threshold=%.2f.",
        df["velocity_flag"].sum(), len(df), VELOCITY_SCORE_THRESHOLD,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "transaction_frequency_24h": [0, 5, 12, 20],
        "transaction_frequency_7d": [0, 10, 30, 60],
    })
    print(compute_velocity_score(dummy))
