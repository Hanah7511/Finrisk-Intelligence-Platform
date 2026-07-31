from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StatisticalDetectionReport:
    train_rows: int = 0
    test_rows: int = 0
    median: float = 0.0
    mad_below: float = 0.0
    mad_above: float = 0.0
    threshold: float = 0.0
    train_flagged_count: int = 0
    test_flagged_count: int = 0

    def summary(self) -> None:
        print("\n--- Statistical Detection Report --------------")
        print(f"Train / test rows            : {self.train_rows} / {self.test_rows}")
        print(f"Median (amount_usd_equivalent): {self.median:.4f}")
        print(f"MAD below / above median     : {self.mad_below:.4f} / {self.mad_above:.4f}")
        print(f"MAD threshold                 : {self.threshold}")
        train_rate = self.train_flagged_count / self.train_rows if self.train_rows else 0
        test_rate = self.test_flagged_count / self.test_rows if self.test_rows else 0
        print(f"Train flagged                 : {self.train_flagged_count} ({train_rate:.1%})")
        print(f"Test flagged                  : {self.test_flagged_count} ({test_rate:.1%})")
        print("------------------------------------------------\n")
