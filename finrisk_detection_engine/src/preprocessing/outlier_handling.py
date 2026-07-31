from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOG_TRANSFORM_COLS: List[str] = [
    "amount_usd_equivalent",
    "customer_avg_amount_30d",
    "customer_txn_count_30d",
    "transaction_frequency_24h",
    "transaction_frequency_7d",
    "item_count",
    "total_quantity",
    "unique_products",
    "unique_merchants",
    "customer_unique_devices_30d",
    "customer_unique_countries_30d",
    "customer_unique_merchants_30d",
    "customer_failed_payments_7d",
    "amount_vs_customer_avg_ratio",
    "txn_frequency_ratio_24h_7d",
    "failed_payment_ratio_7d",
]


def log_transform_skewed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: List[str] = LOG_TRANSFORM_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cols:
        if col not in train_df.columns:
            continue

        train_min = train_df[col].min()
        test_min = test_df[col].min() if col in test_df.columns else 0
        if train_min < 0 or test_min < 0:
            logger.warning(
                "%s has negative values (train_min=%s, test_min=%s) before log-transform; clipping to 0",
                col, train_min, test_min,
            )

        train_df[col] = np.log1p(train_df[col].clip(lower=0))
        test_df[col] = np.log1p(test_df[col].clip(lower=0))
        logger.info("Log-transformed %s", col)

    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train = pd.DataFrame({"amount_usd_equivalent": [10.0, 500.0, 50000.0]})
    test = pd.DataFrame({"amount_usd_equivalent": [200.0, -30.0]})

    train, test = log_transform_skewed(train, test)
    print(train)
    print(test)