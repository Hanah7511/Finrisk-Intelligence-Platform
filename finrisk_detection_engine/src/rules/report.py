from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RuleEngineReport:
    rows_in: int = 0
    rows_out: int = 0
    threshold_exceeded_count: int = 0
    near_threshold_count: int = 0
    structuring_count: int = 0
    sanctions_count: int = 0
    geo_mismatch_count: int = 0
    high_risk_corridor_count: int = 0
    velocity_flag_count: int = 0
    country_risk_tier_counts: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> None:
        print("\n--- Rule Engine Report -----------------------")
        print(f"Rows in / out               : {self.rows_in} / {self.rows_out}")
        print(f"Threshold exceeded          : {self.threshold_exceeded_count}")
        print(f"Near-threshold (structuring smell): {self.near_threshold_count}")
        print(f"Structuring (trailing-24h)  : {self.structuring_count}")
        print(f"Sanctions country           : {self.sanctions_count}")
        print(f"Geo mismatch                : {self.geo_mismatch_count}")
        print(f"High-risk corridor          : {self.high_risk_corridor_count}")
    velocity_flag_count: int = 0
    country_risk_tier_counts: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> None:
        print("\n--- Rule Engine Report -----------------------")
        print(f"Rows in / out               : {self.rows_in} / {self.rows_out}")
        print(f"Threshold exceeded          : {self.threshold_exceeded_count}")
        print(f"Near-threshold (structuring smell): {self.near_threshold_count}")
        print(f"Structuring (trailing-24h)  : {self.structuring_count}")
        print(f"Sanctions country           : {self.sanctions_count}")
        print(f"Geo mismatch                : {self.geo_mismatch_count}")
        print(f"High-risk corridor          : {self.high_risk_corridor_count}")
        print(f"Velocity flagged            : {self.velocity_flag_count}")
        print(f"Country risk tier counts    : {self.country_risk_tier_counts or 'None'}")
        print("----------------------------------------------\n")