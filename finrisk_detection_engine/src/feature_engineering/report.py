from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureEngineeringReport:
    rows_in: int = 0
    rows_out: int = 0
    history_features_recomputed: List[str] = field(default_factory=list)
    derived_features_added: List[str] = field(default_factory=list)

    def summary(self) -> None:
        print("\n--- Feature Engineering Report --------------")
        print(f"Rows in / out               : {self.rows_in} / {self.rows_out}")
        print(f"History features recomputed : {self.history_features_recomputed or 'None'}")
        print(f"Derived features added      : {self.derived_features_added or 'None'}")
        print("----------------------------------------------\n")