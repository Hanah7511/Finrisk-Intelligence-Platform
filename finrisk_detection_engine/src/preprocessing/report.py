from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessingReport:
    rows_before_split: int = 0
    rows_dropped_missing_timestamp: int = 0
    train_rows: int = 0
    test_rows: int = 0
    history_features_recomputed: List[str] = field(default_factory=list)
    derived_features_added: List[str] = field(default_factory=list)
    numeric_cols_filled: List[str] = field(default_factory=list)
    categorical_cols_filled: List[str] = field(default_factory=list)
    ordinal_encoded_cols: List[str] = field(default_factory=list)
    onehot_encoded_cols: List[str] = field(default_factory=list)
    frequency_encoded_cols: List[str] = field(default_factory=list)
    log_transformed_cols: List[str] = field(default_factory=list)
    timestamp_features_added: List[str] = field(default_factory=list)

    def summary(self) -> None:
        print("\n--- Preprocessing Report --------------------")
        print(f"Rows before split          : {self.rows_before_split}")
        print(f"Rows dropped (missing ts)  : {self.rows_dropped_missing_timestamp}")
        print(f"Train / Test rows          : {self.train_rows} / {self.test_rows}")
        print(f"History features recomputed: {self.history_features_recomputed or 'None'}")
        print(f"Derived features added     : {self.derived_features_added or 'None'}")
        print(f"Numeric cols filled        : {self.numeric_cols_filled or 'None'}")
        print(f"Categorical cols filled    : {self.categorical_cols_filled or 'None'}")
        print(f"Ordinal encoded cols       : {self.ordinal_encoded_cols or 'None'}")
        print(f"One-hot encoded cols       : {self.onehot_encoded_cols or 'None'}")
        print(f"Frequency encoded cols     : {self.frequency_encoded_cols or 'None'}")
        print(f"Log-transformed cols       : {self.log_transformed_cols or 'None'}")
        print(f"Timestamp features added   : {self.timestamp_features_added or 'None'}")
        print("----------------------------------------------\n")
