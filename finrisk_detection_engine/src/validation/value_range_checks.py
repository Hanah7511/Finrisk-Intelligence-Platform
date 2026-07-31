from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from src.validation.report import ValidationReport

logger = logging.getLogger(__name__)


def check_invalid_values(df: pd.DataFrame, report: ValidationReport) -> None:
    invalid_checks: Dict[str, int] = {}

    if "amount_usd_equivalent" in df.columns:
        invalid_checks["negative_amount_usd_equivalent"] = int(
            (df["amount_usd_equivalent"] < 0).sum()
        )

    if "transaction_hour" in df.columns:
        invalid_checks["invalid_transaction_hour"] = int(
            ((df["transaction_hour"] < 0) | (df["transaction_hour"] > 23)).sum()
        )

    if "ip_risk_score" in df.columns:
        invalid_checks["invalid_ip_risk_score"] = int(
            ((df["ip_risk_score"] < 0) | (df["ip_risk_score"] > 1)).sum()
        )
        

    if "ip_risk_score" in df.columns:
        invalid_checks["invalid_ip_risk_score"] = int(
            ((df["ip_risk_score"] < 0) | (df["ip_risk_score"] > 1)).sum()
        )

    if "velocity_score" in df.columns:
        invalid_checks["invalid_velocity_score"] = int(
            ((df["velocity_score"] < 0) | (df["velocity_score"] > 1)).sum()
        )

    if "amount_risk_score" in df.columns:
        invalid_checks["invalid_amount_risk_score"] = int(
            ((df["amount_risk_score"] < 0) | (df["amount_risk_score"] > 1)).sum()
        )

    if "device_risk_score" in df.columns:
        invalid_checks["invalid_device_risk_score"] = int(
            ((df["device_risk_score"] < 0) | (df["device_risk_score"] > 1)).sum()
        )

    if "payment_timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["payment_timestamp"]):
        invalid_checks["invalid_payment_timestamp"] = int(df["payment_timestamp"].isna().sum())

    report.invalid_value_checks = invalid_checks

    failed_checks = {k: v for k, v in invalid_checks.items() if v > 0}
    if failed_checks:
        report.passed = False
        report.errors.append(f"Invalid value checks failed: {failed_checks}")
        logger.warning("Invalid value checks failed: %s", failed_checks)
    else:
        logger.info("All invalid value checks passed.")
