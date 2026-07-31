from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ValidationReport:
    passed: bool = True
    missing_columns: List[str] = field(default_factory=list)
from typing import Dict, List


@dataclass
class ValidationReport:
    passed: bool = True
    missing_columns: List[str] = field(default_factory=list)
    type_mismatches: Dict[str, str] = field(default_factory=dict)
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_row_count: int = 0
    duplicate_payment_id_count: int = 0
    invalid_value_checks: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> None:
        print("\n--- Validation Report ----------------------")
        print(f"Status                : {'PASSED' if self.passed else 'FAILED'}")
        print(f"Missing cols          : {self.missing_columns or 'None'}")
        print(f"Type issues           : {self.type_mismatches or 'None'}")
        print(f"Missing values        : {self.missing_value_counts or 'None'}")
        print(f"Duplicate rows        : {self.duplicate_row_count}")
        print(f"Duplicate payment_id  : {self.duplicate_payment_id_count}")
        print(f"Invalid value checks  : {self.invalid_value_checks or 'None'}")
        if self.errors:
            print("Errors:")
            for e in self.errors:
                print(f"  - {e}")
        print("-------------------------------------------\n")