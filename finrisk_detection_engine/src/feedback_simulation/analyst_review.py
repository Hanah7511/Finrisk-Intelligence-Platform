from __future__ import annotations

import numpy as np
import pandas as pd

from configs.loader import load_thresholds_config

_config = load_thresholds_config()["feedback_simulation"]
RANDOM_SEED: int = _config["random_seed"]
FALSE_NEGATIVE_RATE: float = _config["false_negative_rate"]
CONFIRMATION_RATES: dict = _config["confirmation_rates"]

NO_RULE_LABEL = "no_rule_triggered"

# How much each risk signal present on a row bumps its confirmation
# probability, and the overall cap on that bump - keeps this a *nudge* on
# top of the base rate, not a rewrite of it, so confirmation is still
# fundamentally a random draw (an analyst weighing context), not a second
# hidden rule engine that would just recreate the original leakage problem.
RISK_BUMP_PER_SIGNAL = 0.5
MAX_RISK_MULTIPLIER = 3.0
AMOUNT_RATIO_THRESHOLD = 2.0


def _compute_risk_multiplier(df: pd.DataFrame) -> pd.Series:
    """Blends four context signals a real analyst would plausibly weigh
    when reviewing an alert, on top of the rule that triggered it - amount
    relative to *this customer's own* typical amount (not raw amount, so a
    company account's routinely large transfers don't look anomalous),
    customer risk segment, and two network-adjacent flags. These are
    independent of the 5 columns that actually drive fraud_rule_name
    (sanctions_country_flag, structuring_flag, geo_mismatch_flag,
    high_risk_corridor_flag, velocity_score), so this doesn't reintroduce
    the original circular-label problem - it only nudges a probability,
    never decides confirmation outright.

    customer_avg here is computed across each customer's full history
    (past and future relative to any given row), which would be a leakage
    problem for a model *feature* - but this only shapes a simulated
    training *label*, never something Layer 8 sees as an input, so there's
    no leakage into what the model actually learns from.
    """
    customer_avg = df.groupby("customer_id")["amount_usd_equivalent"].transform("mean")
    amount_ratio = df["amount_usd_equivalent"] / customer_avg.replace(0, np.nan)
    amount_ratio = amount_ratio.fillna(1.0)

    risk_flags = pd.concat([
        (amount_ratio > AMOUNT_RATIO_THRESHOLD),
        (df["customer_risk_segment"] == "HIGH"),
        (df["device_shared_flag"] == 1),
        (df["repeated_beneficiary_flag"] == 1),
    ], axis=1)

    risk_flag_count = risk_flags.sum(axis=1)
    multiplier = 1 + RISK_BUMP_PER_SIGNAL * risk_flag_count
    return multiplier.clip(upper=MAX_RISK_MULTIPLIER)


def simulate_analyst_confirmation(
    df: pd.DataFrame,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df["analyst_confirmed_label"] = 0

    risk_multiplier = _compute_risk_multiplier(df)

    # Each rule type gets its own base confirmation rate - a sanctions/
    # structuring hit is a specific, meaningful signal (higher rate);
    # generic velocity-type alerts are noisier in practice (lower rate).
    # The base rate is then nudged per-row by risk_multiplier before the
    # draw, so confirmation isn't purely a function of which rule fired.
    for rule_name, rate in CONFIRMATION_RATES.items():
        mask = df["fraud_rule_name"] == rule_name
        n = int(mask.sum())
        if n == 0:
            continue
        adjusted_rate = np.clip(rate * risk_multiplier[mask].values, 0.0, 1.0)
        confirmed = rng.random(n) < adjusted_rate
        df.loc[mask, "analyst_confirmed_label"] = confirmed.astype(int)

    # False negatives: a small number of transactions no rule ever flagged,
    # but a human would have caught anyway - the same risk signals above
    # make some of those misses more findable than others.
    no_rule_mask = df["fraud_rule_name"] == NO_RULE_LABEL
    n_no_rule = int(no_rule_mask.sum())
    if n_no_rule > 0:
        adjusted_fn_rate = np.clip(FALSE_NEGATIVE_RATE * risk_multiplier[no_rule_mask].values, 0.0, 1.0)
        missed = rng.random(n_no_rule) < adjusted_fn_rate
        df.loc[no_rule_mask, "analyst_confirmed_label"] = missed.astype(int)

    return df
