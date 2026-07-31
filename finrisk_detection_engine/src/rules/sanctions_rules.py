from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from configs.loader import load_thresholds_config

logger = logging.getLogger(__name__)

_config = load_thresholds_config()["rules"]["country_risk_tiers"]

SANCTIONED_COUNTRIES: List[str] = _config["sanctioned"]["countries"]
GREY_LIST_COUNTRIES: List[str] = _config["grey_list"]["countries"]
MODERATE_RISK_COUNTRIES: List[str] = _config["moderate"]["countries"]
GCC_BASELINE_COUNTRIES: List[str] = _config["gcc_baseline"]["countries"]

_TIER_SCORES: Dict[str, float] = {
    "sanctioned": _config["sanctioned"]["score"],
    "grey_list": _config["grey_list"]["score"],
    "moderate": _config["moderate"]["score"],
    "gcc_baseline": _config["gcc_baseline"]["score"],
    "baseline": _config["baseline_score"],
}

_COUNTRY_TO_TIER: Dict[str, str] = {}
for _country in SANCTIONED_COUNTRIES:
    _COUNTRY_TO_TIER[_country] = "sanctioned"
for _country in GREY_LIST_COUNTRIES:
    _COUNTRY_TO_TIER[_country] = "grey_list"
for _country in MODERATE_RISK_COUNTRIES:
    _COUNTRY_TO_TIER[_country] = "moderate"
for _country in GCC_BASELINE_COUNTRIES:
    _COUNTRY_TO_TIER[_country] = "gcc_baseline"


def get_country_risk_tier(country: str) -> str:
    return _COUNTRY_TO_TIER.get(country, "baseline")


def get_country_risk_score(country: str) -> float:
    return _TIER_SCORES[get_country_risk_tier(country)]


def is_elevated_or_above(country: str) -> bool:
    """True for grey_list, moderate, or sanctioned tiers - anything above ordinary
    baseline/GCC-baseline risk."""
    return get_country_risk_tier(country) in {"grey_list", "moderate", "sanctioned"}


def apply_sanctions_rule(
    df: pd.DataFrame,
    country_col: str = "counterparty_country",
) -> pd.DataFrame:
    df = df.copy()
    df["country_risk_tier"] = df[country_col].map(get_country_risk_tier)
    df["country_risk_score"] = df[country_col].map(get_country_risk_score)
    df["sanctions_country_flag"] = (df["country_risk_tier"] == "sanctioned").astype(int)
    df["country_risk_tier"] = df[country_col].map(get_country_risk_tier)
    df["country_risk_score"] = df[country_col].map(get_country_risk_score)
    df["sanctions_country_flag"] = (df["country_risk_tier"] == "sanctioned").astype(int)

    logger.info(
        "Sanctions rule flagged %d/%d transactions against FATF Black List.",
        df["sanctions_country_flag"].sum(), len(df),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dummy = pd.DataFrame({
        "counterparty_country": ["Iran", "China", "United Arab Emirates", "India", "Germany"],
    })
    print(apply_sanctions_rule(dummy))
