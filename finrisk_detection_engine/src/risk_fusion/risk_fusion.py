from __future__ import annotations

"""
Layer 9: risk fusion. Combines every detection layer's score into one
final_risk_score (0-1) and an alert severity (LOW / MEDIUM / HIGH / CRITICAL).

- Weights and severity cut-offs come from configs/thresholds.yaml (risk_fusion).
- A layer whose score column is missing (e.g. random_forest, not built yet) is
  skipped and its weight is shared out over the layers that are present, so the
  weights used still add up to 1. The report lists what was skipped.
- A sanctioned-country transaction is never below HIGH, whatever its score:
  in AML a sanctions link is escalated even when everything else looks normal.
- top_contributor names the layer that added most to the score, so an analyst
  can see why an alert was raised.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from configs.loader import load_thresholds_config
from src.risk_fusion.report import RiskFusionReport

logger = logging.getLogger(__name__)

_config = load_thresholds_config()["risk_fusion"]

# Which column holds each layer's 0-1 score.
LAYER_SCORE_COLUMNS: Dict[str, str] = {
    "rules": "rules_score",                         # made below from the rule flags
    "xgboost": "fraud_probability",                 # supervised_detection
    "unsupervised": "unsupervised_anomaly_score",   # Isolation Forest + One-Class SVM
    "statistical": "statistical_anomaly_flag",      # double MAD (0 or 1)
    "random_forest": "random_forest_probability",   # not built yet
}

# Any one of these means the transaction breaks a hard AML rule.
HARD_RULE_FLAGS = ["threshold_exceeded_flag", "structuring_flag", "sanctions_country_flag"]
# These only add suspicion: their average is the rules score when no hard rule fired.
SOFT_RULE_FLAGS = ["geo_mismatch_flag", "high_risk_corridor_flag", "velocity_flag"]

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def compute_rules_score(df: pd.DataFrame) -> pd.Series:
    """1.0 if any hard rule fired, otherwise the share of soft rules that fired."""
    hard = [c for c in HARD_RULE_FLAGS if c in df.columns]
    soft = [c for c in SOFT_RULE_FLAGS if c in df.columns]
    if not hard and not soft:
        return pd.Series(np.nan, index=df.index)
    hard_hit = df[hard].max(axis=1) if hard else pd.Series(0, index=df.index)
    soft_share = df[soft].mean(axis=1) if soft else pd.Series(0.0, index=df.index)
    return np.where(hard_hit >= 1, 1.0, soft_share)


def weights_for(df: pd.DataFrame) -> Tuple[Dict[str, float], list]:
    """The config weights of the layers present, scaled to add up to 1."""
    present = {name: w for name, w in _config["weights"].items()
               if LAYER_SCORE_COLUMNS.get(name) in df.columns}
    missing = [name for name in _config["weights"] if name not in present]
    total = sum(present.values())
    if total == 0:
        raise ValueError("No detection layer scores found: run the detection layers first.")
    return {name: w / total for name, w in present.items()}, missing


def severity_for(score: float) -> str:
    """Cut-offs from the config: below low -> LOW, below medium -> MEDIUM, below high -> HIGH, else CRITICAL."""
    cut = _config["severity"]
    if score < cut["low"]:
        return "LOW"
    if score < cut["medium"]:
        return "MEDIUM"
    if score < cut["high"]:
        return "HIGH"
    return "CRITICAL"


def run_risk_fusion(df: pd.DataFrame) -> Tuple[pd.DataFrame, RiskFusionReport]:
    report = RiskFusionReport(rows=len(df))
    df = df.copy()
    df["rules_score"] = compute_rules_score(df)

    weights, missing = weights_for(df)
    report.weights_used, report.layers_missing = weights, missing
    report.layers_used = list(weights)
    if missing:
        logger.warning("Risk fusion: no score for %s; its weight is shared over the other layers.", missing)

    contributions = pd.DataFrame({
        name: df[LAYER_SCORE_COLUMNS[name]].astype(float).clip(0, 1).fillna(0) * w
        for name, w in weights.items()
    })
    df["final_risk_score"] = contributions.sum(axis=1).round(4)
    df["top_contributor"] = contributions.idxmax(axis=1)
    df["alert_severity"] = df["final_risk_score"].map(severity_for)

    # A sanctions link is escalated even when the other layers see nothing unusual.
    if "sanctions_country_flag" in df.columns:
        below_high = df["alert_severity"].isin(["LOW", "MEDIUM"])
        override = (df["sanctions_country_flag"] == 1) & below_high
        df.loc[override, "alert_severity"] = "HIGH"
        df.loc[override, "top_contributor"] = "sanctions_override"
        report.sanctions_overrides = int(override.sum())

    counts = df["alert_severity"].value_counts()
    report.severity_counts = {s: int(counts.get(s, 0)) for s in SEVERITY_ORDER}
    logger.info("Risk fusion scored %d rows: %s", len(df), report.severity_counts)
    return df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Hand-made rows, one per situation, so each result can be checked by eye.
    cases = pd.DataFrame([
        # name,                       threshold structuring sanctions geo corridor velocity  xgb   unsup  stat
        ["normal payment",                    0, 0, 0,  0, 0, 0,   0.02, 0.10, 0],
        ["unusual but no rule broken",        0, 0, 0,  1, 0, 1,   0.35, 0.70, 1],
        ["structuring + model agrees",        0, 1, 0,  0, 0, 1,   0.85, 0.80, 1],
        ["sanctioned country, looks normal",  0, 0, 1,  0, 0, 0,   0.05, 0.10, 0],
        ["everything fires",                  1, 1, 1,  1, 1, 1,   0.95, 0.95, 1],
    ], columns=["case", "threshold_exceeded_flag", "structuring_flag", "sanctions_country_flag",
                "geo_mismatch_flag", "high_risk_corridor_flag", "velocity_flag",
                "fraud_probability", "unsupervised_anomaly_score", "statistical_anomaly_flag"])

    scored, report = run_risk_fusion(cases)
    report.summary()
    pd.set_option("display.width", 140)
    print(scored[["case", "rules_score", "final_risk_score", "alert_severity", "top_contributor"]].to_string(index=False))
