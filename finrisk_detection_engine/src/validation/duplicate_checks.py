from __future__ import annotations

import logging

import pandas as pd

from src.validation.report import ValidationReport

logger = logging.getLogger(__name__)


def check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    duplicate_count = int(df.duplicated().sum())
    report.duplicate_row_count = duplicate_count

    if duplicate_count > 0:
        logger.warning("Duplicate rows detected: %d", duplicate_count)
    else:
        logger.info("No duplicate rows detected.")

    if "payment_id" in df.columns:
        duplicate_payment_ids = int(df.duplicated(subset=["payment_id"]).sum())
        report.duplicate_payment_id_count = duplicate_payment_ids

        if duplicate_payment_ids > 0:
            report.passed = False
            report.errors.append(
                f"Duplicate payment_id values found: {duplicate_payment_ids}"
            )
            logger.warning("Duplicate payment_id detected: %d", duplicate_payment_ids)
        else:
            logger.info("No duplicate payment_id values detected.")