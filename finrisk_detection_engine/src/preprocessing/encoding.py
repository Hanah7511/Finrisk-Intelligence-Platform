from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ORDINAL_LEVELS: Dict[str, List[str]] = {
    "kyc_level": ["LOW", "MEDIUM", "HIGH"],
    "geo_risk_level": ["LOW", "MEDIUM", "HIGH"],
    "customer_risk_segment": ["LOW", "MEDIUM", "HIGH"],
}

ONEHOT_COLS = [
    "payment_method",
    "transaction_channel",
    "transaction_direction",
    "payment_status",
]

FREQUENCY_COLS = [
    "counterparty_country",
    "device_id",
]


def label_encode_ordinal(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col, levels in ORDINAL_LEVELS.items():
        if col not in train_df.columns:
            continue
        mapping = {level: idx for idx, level in enumerate(levels)}
        train_df[col] = train_df[col].map(mapping)
        test_df[col] = test_df[col].map(mapping)
        unseen = test_df[col].isna().sum()
        if unseen:
            logger.warning("%d unseen/unmapped values in test for %s", unseen, col)
        logger.info("Label-encoded %s using order=%s", col, levels)

    return train_df, test_df


def one_hot_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in ONEHOT_COLS if c in train_df.columns]
    if not cols:
        return train_df, test_df

    train_encoded = pd.get_dummies(train_df, columns=cols)
    test_encoded = pd.get_dummies(test_df, columns=cols)

    # test must have exactly the same dummy columns as train, in the same order
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    dummy_cols = [c for c in train_encoded.columns if c not in train_df.columns]
    train_encoded[dummy_cols] = train_encoded[dummy_cols].astype(int)
    test_encoded[dummy_cols] = test_encoded[dummy_cols].astype(int)

    logger.info("One-hot encoded columns: %s", cols)
    return train_encoded, test_encoded


def frequency_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in FREQUENCY_COLS:
        if col not in train_df.columns:
            continue
        freq_map = train_df[col].value_counts(normalize=True)
        train_df[col] = train_df[col].map(freq_map)
        test_df[col] = test_df[col].map(freq_map).fillna(0.0)
        logger.info("Frequency-encoded %s (%d unique in train)", col, len(freq_map))

    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train = pd.DataFrame({
        "kyc_level": ["LOW", "HIGH", "MEDIUM"],
        "payment_method": ["CARD", "UPI", "CARD"],
        "counterparty_country": ["India", "India", "Singapore"],
    })
    test = pd.DataFrame({
        "kyc_level": ["HIGH", "LOW"],
        "payment_method": ["UPI", "NETBANKING"],
        "counterparty_country": ["India", "Germany"],
    })

    train, test = label_encode_ordinal(train, test)
    train, test = one_hot_encode(train, test)
    train, test = frequency_encode(train, test)
    print(train)
    print(test)