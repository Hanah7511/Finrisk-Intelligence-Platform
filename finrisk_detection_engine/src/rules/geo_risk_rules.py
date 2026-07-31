from __future__ import annotations

import logging

import pandas as pd

from src.rules.sanctions_rules import is_elevated_or_above

logger = logging.getLogger(__name__)

LARGE_TRANSACTION_PERCENTILE = 0.90


def apply_geo_risk_rules(
    df: pd.DataFrame,
    country_col: str = "counterparty_country",
    new_country_flag_col: str = "new_country_flag",
    direction_col: str = "transaction_direction",
    amount_col: str = "amount_usd_equivalent",
) -> pd.DataFrame:
    """
    geo_mismatch_flag: reuses Layer 4's new_country_flag (first time this customer has
    transacted with this country), but only counts as a genuine mismatch if that new
    country is also grey-list-or-above risk - a first transaction to an ordinary-risk
    country isn't a meaningful geo signal on its own.

    high_risk_corridor_flag: an outbound transaction to a grey-list-or-above destination,
    above this dataset's large-transaction percentile - money actually leaving the
    regulated system toward a risky jurisdiction in size, not just any risky-country contact.
    """
    df = df.copy()
    is_elevated = df[country_col].map(is_elevated_or_above)

    if new_country_flag_col in df.columns:
        df["geo_mismatch_flag"] = (df[new_country_flag_col].astype(bool) & is_elevated).astype(int)
    else:
        logger.warning("%s column missing; geo_mismatch_flag defaulting to 0.", new_country_flag_col)
        df["geo_mismatch_flag"] = 0

    large_amount_cutoff = df[amount_col].quantile(LARGE_TRANSACTION_PERCENTILE)
    is_outbound = df[direction_col].astype(str).str.upper() == "OUTBOUND"
    is_large = df[amount_col] >= large_amount_cutoff

    df["high_risk_corridor_flag"] = (is_outbound & is_elevated & is_large).astype(int)

    logger.info(
        "Geo risk rules: %d geo_mismatch, %d high_risk_corridor flags out of %d rows.",
        df["geo_mismatch_flag"].sum(), df["high_risk_corridor_flag"].sum(), len(df),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "counterparty_country": ["Iran", "Germany", "China", "India"],
        "new_country_flag": [1, 1, 0, 1],
        "transaction_direction": ["OUTBOUND", "OUTBOUND", "OUTBOUND", "INBOUND"],
        "amount_usd_equivalent": [5000.0, 100.0, 20000.0, 3000.0],
    })
    print(apply_geo_risk_rules(dummy))
