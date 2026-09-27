from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from src.feedback_simulation.report import FeedbackSimulationReport
from src.feedback_simulation.analyst_review import (
    simulate_analyst_confirmation,
    CONFIRMATION_RATES,
    NO_RULE_LABEL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_feedback_simulation(df: pd.DataFrame) -> Tuple[pd.DataFrame, FeedbackSimulationReport]:
    report = FeedbackSimulationReport()
    report.rows_in = len(df)

    df = simulate_analyst_confirmation(df)

    alert_mask = df["fraud_rule_name"] != NO_RULE_LABEL
    report.alert_rows = int(alert_mask.sum())
    report.confirmed_from_alerts = int(df.loc[alert_mask, "analyst_confirmed_label"].sum())
    report.confirmed_false_negatives = int(
        df.loc[~alert_mask, "analyst_confirmed_label"].sum()
    )
    report.total_confirmed = int(df["analyst_confirmed_label"].sum())
    report.confirmation_rate_by_rule = {
        rule: float(df.loc[df["fraud_rule_name"] == rule, "analyst_confirmed_label"].mean())
        for rule in CONFIRMATION_RATES
        if (df["fraud_rule_name"] == rule).any()
    }

    return df, report


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(7)
    n = 2000
    rule_names = rng.choice(
        [NO_RULE_LABEL, "sanctions_country", "structuring_flag",
         "geo_velocity", "corridor_velocity", "amount_frequency", "device_geo_risk"],
        n,
        p=[0.70, 0.19, 0.005, 0.04, 0.04, 0.02, 0.005],
    )
    dummy = pd.DataFrame({
        "fraud_rule_name": rule_names,
        "customer_id": rng.choice([f"CUST_{i}" for i in range(50)], n),
        "amount_usd_equivalent": rng.exponential(500, n),
        "customer_risk_segment": rng.choice(["LOW", "MEDIUM", "HIGH"], n),
        "device_shared_flag": rng.integers(0, 2, n),
        "repeated_beneficiary_flag": rng.integers(0, 2, n),
    })

    out, report = run_feedback_simulation(dummy)
    report.summary()
    print(out["analyst_confirmed_label"].value_counts())
                                                                                        

