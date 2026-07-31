from __future__ import annotations

import logging

import pandas as pd

from src.validation.report import ValidationReport

logger = logging.getLogger(__name__)


def check_missing_values(
    df: pd.DataFrame,
    report: ValidationReport,
    threshold: float = 0.3,
) -> None:
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        report.missing_value_counts = missing_counts.to_dict()
        for col, count in missing_counts.items():
            ratio = count / len(df)
            if ratio > threshold:
                report.passed = False
                report.errors.append(
                    f"Column '{col}' has {ratio:.1%} missing values "
                    f"(threshold={threshold:.0%})"
                )
                logger.warning(
                    "High missing values in column=%s ratio=%.2f", col, ratio
                )
    else:
        logger.info("No missing values detected.")