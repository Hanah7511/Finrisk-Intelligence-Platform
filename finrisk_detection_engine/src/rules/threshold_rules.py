from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from configs.loader import load_thresholds_config

logger = logging.getLogger(__name__)

_config = load_thresholds_config()["rules"]

LARGE_TRANSACTION_THRESHOLD_USD: float = _config["large_transaction_threshold"]
JURISDICTION_THRESHOLDS_USD: Dict[str, float] = _config["jurisdiction_thresholds_usd"]
JURISDICTION_COUNTRIES: Dict[str, list] = _config["jurisdiction_countries"]

# A transaction sized at 90-100% of its jurisdiction's reporting threshold is
# "near-threshold" - close enough to look like deliberate avoidance (structuring_rules.py).
STRUCTURING_NEAR_THRESHOLD_RATIO = 0.90

_COUNTRY_TO_JURISDICTION: Dict[str, str] = {}
for _jurisdiction, _countries in JURISDICTION_COUNTRIES.items():
    for _country in _countries:
        _COUNTRY_TO_JURISDICTION[_country] = _jurisdiction


def get_applicable_threshold(country: str) -> float:
    jurisdiction = _COUNTRY_TO_JURISDICTION.get(country, "DEFAULT")
    return JURISDICTION_THRESHOLDS_USD[jurisdiction]


def apply_threshold_rule(
    df: pd.DataFrame,
    amount_col: str = "amount_usd_equivalent",
) -> pd.DataFrame:
    """Flat large-transaction check: amount at or above the configured absolute
    threshold, independent of jurisdiction (the jurisdiction-aware check lives in
    structuring_rules.py)."""
    df = df.copy()
    df["threshold_exceeded_flag"] = (df[amount_col] >= LARGE_TRANSACTION_THRESHOLD_USD).astype(int)

    logger.info(
        "Threshold rule flagged %d/%d transactions >= $%.2f.",
        df["threshold_exceeded_flag"].sum(), len(df), LARGE_TRANSACTION_THRESHOLD_USD,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "amount_usd_equivalent": [50.0, 999.0, 1000.0, 5000.0],
    })
    print(apply_threshold_rule(dummy))
