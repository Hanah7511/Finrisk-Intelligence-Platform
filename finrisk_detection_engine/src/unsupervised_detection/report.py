from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnsupervisedDetectionReport:
    train_rows: int = 0
    test_rows: int = 0
    feature_count: int = 0
    contamination: float = 0.0
    train_flagged_count: int = 0
    test_flagged_count: int = 0

    def summary(self) -> None:
        print("\n--- Unsupervised Detection Report -------------")
        print(f"Train / test rows             : {self.train_rows} / {self.test_rows}")
        print(f"Feature count                  : {self.feature_count}")
        print(f"Contamination                  : {self.contamination}")
        train_rate = self.train_flagged_count / self.train_rows if self.train_rows else 0
        test_rate = self.test_flagged_count / self.test_rows if self.test_rows else 0
        print(f"Train flagged                  : {self.train_flagged_count} ({train_rate:.1%})")
        print(f"Test flagged                   : {self.test_flagged_count} ({test_rate:.1%})")
        print("------------------------------------------------\n")
