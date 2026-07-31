from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.rules.report import RuleEngineReport
from src.rules.threshold_rules import apply_threshold_rule
from src.rules.structuring_rules import apply_structuring_rule
from src.rules.sanctions_rules import apply_sanctions_rule
from src.rules.geo_risk_rules import apply_geo_risk_rules
from src.rules.velocity_rules import compute_velocity_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_rule_engine(df: pd.DataFrame) -> Tuple[pd.DataFrame, RuleEngineReport]:
    report = RuleEngineReport()
    report.rows_in = len(df)

    df = apply_threshold_rule(df)
    report.threshold_exceeded_count = int(df["threshold_exceeded_flag"].sum())

    df = apply_structuring_rule(df)
    report.near_threshold_count = int(df["near_threshold_flag"].sum())
    report.structuring_count = int(df["structuring_flag"].sum())

    df = apply_sanctions_rule(df)
    report.sanctions_count = int(df["sanctions_country_flag"].sum())
    report.country_risk_tier_counts = df["country_risk_tier"].value_counts().to_dict()

    df = apply_geo_risk_rules(df)
    report.geo_mismatch_count = int(df["geo_mismatch_flag"].sum())
    report.high_risk_corridor_count = int(df["high_risk_corridor_flag"].sum())

    df = compute_velocity_score(df)
    report.velocity_flag_count = int(df["velocity_flag"].sum())

    report.rows_out = len(df)
    return df, report


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 30
    dummy = pd.DataFrame({
        "customer_id": rng.choice(["C1", "C2", "C3"], n),
        "payment_timestamp": pd.date_range("2026-01-01", periods=n, freq="6h"),
        "amount_usd_equivalent": rng.exponential(3000, n),
        "counterparty_country": rng.choice(
            ["United States", "India", "United Arab Emirates", "Iran", "China", "Germany"], n
        ),
        "new_country_flag": rng.integers(0, 2, n),
        "transaction_direction": rng.choice(["INBOUND", "OUTBOUND"], n),
        "transaction_frequency_24h": rng.integers(0, 15, n),
        "transaction_frequency_7d": rng.integers(0, 40, n),
    })

    engineered, report = run_rule_engine(dummy)
    report.summary()
    print(engineered[[
        "customer_id", "counterparty_country", "country_risk_tier",
        "structuring_flag", "sanctions_country_flag", "geo_mismatch_flag",
        "high_risk_corridor_flag", "velocity_score",
    ]].head(15))
