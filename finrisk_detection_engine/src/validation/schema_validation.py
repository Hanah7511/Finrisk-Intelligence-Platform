from __future__ import annotations

import logging
from typing import List

import pandas as pd

from src.validation.report import ValidationReport
from src.validation.datatype_validation import check_data_types
from src.validation.missing_values import check_missing_values
from src.validation.duplicate_checks import check_duplicates
from src.validation.value_range_checks import check_invalid_values

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Expected Schema

EXPECTED_COLUMNS: List[str] = [
    "payment_id",
    "order_id",
    "customer_id",
    "counterparty_account_id"
    "account_age_days",
    "kyc_level",
    "customer_risk_segment",
    "repeated_beneficiary_flag",
    "device_id",
    "ip_risk_score",
    "device_shared_flag",
    "payment_status",
    "payment_timestamp",
    "payment_method",
    "transaction_channel",
    "geo_risk_level",
    "transaction_direction",
    "counterparty_country",
    "instrument_age_days",
    "transaction_hour",
    "amount_usd_equivalent",
    "velocity_score",
    "amount_risk_score",
    "device_risk_score",
    "transaction_frequency_24h",
    "transaction_frequency_7d",
    "geo_mismatch_flag",
    "sanctions_country_flag",
    "high_risk_corridor_flag",
    "structuring_flag",
    "fraud_label",
    "fraud_rule_name",
    "label_source",
    "item_count",
    "total_quantity",
    "unique_products",
    "unique_merchants",
    "max_product_risk_score",
    "avg_product_risk_score",
    "customer_txn_count_30d",
    "customer_avg_amount_30d",
    "customer_unique_devices_30d",
    "customer_unique_countries_30d",
    "customer_unique_merchants_30d",
    "customer_failed_payments_7d",
    "new_device_flag",
    "new_country_flag",
]


def check_required_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        report.missing_columns = missing
        report.passed = False
        report.errors.append(f"Missing required columns: {missing}")
        logger.warning("Missing columns: %s", missing)
    else:
        logger.info("All required columns present.")


# Main Validator

def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    logger.info("Starting dataset validation. Shape=%s", df.shape)
    report = ValidationReport()

    check_required_columns(df, report)
    check_data_types(df, report)
    check_missing_values(df, report)
    check_duplicates(df, report)
    check_invalid_values(df, report)

    if report.passed:
        logger.info("Validation PASSED.")
    else:
        logger.warning("Validation FAILED. See report for details.")

    return report


if __name__ == "__main__":
    dummy_data = {
        "payment_id": [1, 2, 2],
        "order_id": [101, 102, 102],
        "customer_id": [1001, 1002, 1002],
        "account_age_days": [200, 300, 300],
        "kyc_level": ["HIGH", "LOW", "LOW"],
        "customer_risk_segment": ["HIGH", "MEDIUM", "MEDIUM"],
        "repeated_beneficiary_flag": [0, 1, 1],
        "device_id": ["DEV_1", "DEV_2", "DEV_2"],
        "ip_risk_score": [0.2, 0.5, 1.3],
        "device_shared_flag": [0, 1, 1],
        "payment_status": ["success", "failed", "failed"],
        "payment_timestamp": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-02"]),
        "payment_method": ["CARD", "UPI", "UPI"],
        "transaction_channel": ["WEB", "MOBILE", "MOBILE"],
        "geo_risk_level": ["LOW", "HIGH", "HIGH"],
        "transaction_direction": ["INBOUND", "OUTBOUND", "OUTBOUND"],
        "counterparty_country": ["India", "Singapore", "Singapore"],
        "instrument_age_days": [100, 50, 50],
        "transaction_hour": [10, 26, 26],
        "amount_usd_equivalent": [100.0, -50.0, -50.0],
        "velocity_score": [0.2, 0.5, 1.5],
        "amount_risk_score": [0.2, 0.3, 0.3],
        "device_risk_score": [0.2, 0.3, 0.3],
        "transaction_frequency_24h": [2, 5, 5],
        "transaction_frequency_7d": [5, 10, 10],
        "geo_mismatch_flag": [0, 1, 1],
        "sanctions_country_flag": [0, 0, 0],
        "high_risk_corridor_flag": [0, 1, 1],
        "structuring_flag": [0, 1, 1],
        "fraud_label": [0, 1, 1],
        "fraud_rule_name": [None, "STRUCTURING_RULE_7", "STRUCTURING_RULE_7"],
        "label_source": ["SYSTEM", "RULE_ENGINE", "RULE_ENGINE"],
        "item_count": [1, 2, 2],
        "total_quantity": [1, 3, 3],
        "unique_products": [1, 2, 2],
        "unique_merchants": [1, 2, 2],
        "max_product_risk_score": [0.1, 0.5, 0.5],
        "avg_product_risk_score": [0.1, 0.3, 0.3],
        "customer_txn_count_30d": [5, 10, 10],
        "customer_avg_amount_30d": [80.0, 120.0, 120.0],
        "customer_unique_devices_30d": [1, 2, 2],
        "customer_unique_countries_30d": [1, 2, 2],
        "customer_unique_merchants_30d": [2, 3, 3],
        "customer_failed_payments_7d": [0, 1, 1],
        "new_device_flag": [0, 1, 1],
        "new_country_flag": [0, 1, 1],
    }

    df_test = pd.DataFrame(dummy_data)
    report = validate_dataset(df_test)
    report.summary()