from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RiskFusionReport:
    rows: int = 0
    layers_used: List[str] = field(default_factory=list)
    layers_missing: List[str] = field(default_factory=list)
    weights_used: Dict[str, float] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    sanctions_overrides: int = 0

    def summary(self) -> None:
        print("\n--- Risk Fusion Report ------------------------")
        print(f"Rows scored                   : {self.rows}")
        print(f"Layers used (weight)          : "
              + ", ".join(f"{name} {w:.3f}" for name, w in self.weights_used.items()))
        if self.layers_missing:
            print(f"Layers missing (weight shared): {', '.join(self.layers_missing)}")
        print(f"Severity counts               : {self.severity_counts}")
        print(f"Raised to HIGH by sanctions   : {self.sanctions_overrides}")
        print("------------------------------------------------\n")
