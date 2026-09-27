from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FeedbackSimulationReport:
    rows_in: int = 0
    alert_rows: int = 0
    confirmed_from_alerts: int = 0
    confirmed_false_negatives: int = 0
    total_confirmed: int = 0
    confirmation_rate_by_rule: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> None:
        print("\n--- Feedback Simulation Report ----------------")
        print(f"Rows in                      : {self.rows_in}")
        print(f"Alert rows (rule triggered)  : {self.alert_rows}")
        print(f"Confirmed from alerts        : {self.confirmed_from_alerts}")
        print(f"Confirmed false negatives    : {self.confirmed_false_negatives}")
        overall_rate = self.total_confirmed / self.rows_in if self.rows_in else 0
        print(f"Total confirmed fraud        : {self.total_confirmed} ({overall_rate:.2%})")
        print("Confirmation rate by rule:")
        for rule, rate in self.confirmation_rate_by_rule.items():
            print(f"  {rule:<20}: {rate:.2%}")
        print("------------------------------------------------\n")
