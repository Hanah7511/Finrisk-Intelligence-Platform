from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from src.validation.report import ValidationReport

logger = logging.getLogger(__name__)

EXPECTED_TYPES: Dict[str, str] = {
    "payment_id": "numeric",
    "order_id": "numeric",
    "customer_id": "numeric",
    "counterparty_account_id": "numeric",
    "account_age_days": "numeric",
    "kyc_level": "string",
    "customer_risk_segment": "string",
    "repeated_beneficiary_flag": "numeric",
    "device_id": "string",
    "ip_risk_score": "numeric",
    "device_shared_flag": "numeric",
    "payment_status": "string",
    "payment_timestamp": "datetime",
    "payment_method": "string",
    "transaction_channel": "string",
    "geo_risk_level": "string",
    "transaction_direction": "string",
    "counterparty_country": "string",
    "instrument_age_days": "numeric",
    "transaction_hour": "numeric",
    "amount_usd_equivalent": "numeric",
    "velocity_score": "numeric",
    "amount_risk_score": "numeric",
    "device_risk_score": "numeric",
    "transaction_frequency_24h": "numeric",
    "transaction_frequency_7d": "numeric",
    "geo_mismatch_flag": "numeric",
    "sanctions_country_flag": "numeric",
    "high_risk_corridor_flag": "numeric",
    "structuring_flag": "numeric",
    "fraud_label": "numeric",
    "fraud_rule_name": "string",
    "label_source": "string",
    "item_count": "numeric",
    "total_quantity": "numeric",
    "unique_products": "numeric",
    "unique_merchants": "numeric",
    "max_product_risk_score": "numeric",
    "avg_product_risk_score": "numeric",
    "customer_txn_count_30d": "numeric",
    "customer_avg_amount_30d": "numeric",
    "customer_unique_devices_30d": "numeric",
    "customer_unique_countries_30d": "numeric",
    "customer_unique_merchants_30d": "numeric",
    "customer_failed_payments_7d": "numeric",
    "new_device_flag": "numeric",
    "new_country_flag": "numeric",
}

def check_data_types(df: pd.DataFrame, report: ValidationReport) -> None:
    for col, expected_type in EXPECTED_TYPES.items():
        if col not in df.columns:
            continue

        series = df[col]

        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                report.type_mismatches[col] = f"expected=numeric, actual={series.dtype}"

        elif expected_type == "string":
            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                report.type_mismatches[col] = f"expected=string, actual={series.dtype}"

        elif expected_type == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(series):
                report.type_mismatches[col] = f"expected=datetime, actual={series.dtype}"

    if report.type_mismatches:
        report.passed = False
        report.errors.append("Data type mismatches detected.")
        logger.warning("Data type mismatches found.")
    else:
        logger.info("All data types validated.")