from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SupervisedDetectionReport:
    train_rows: int = 0
    test_rows: int = 0
    feature_count: int = 0
    train_positive_count: int = 0
    test_positive_count: int = 0
    scale_pos_weight: float = 0.0
    optimal_threshold: float = 0.5
    train_flagged_count: int = 0
    test_flagged_count: int = 0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    test_auc_roc: float = 0.0
    test_auc_pr: float = 0.0
    top_global_features: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> None:
        print("\n--- Supervised Detection Report (Layer 8) -----")
        print(f"Train / test rows            : {self.train_rows} / {self.test_rows}")
        print(f"Feature count                 : {self.feature_count}")
        print(f"Train positives (confirmed)   : {self.train_positive_count}")
        print(f"Test positives (confirmed)    : {self.test_positive_count}")
        print(f"scale_pos_weight (train-fit)  : {self.scale_pos_weight:.2f}")
        print(f"Optimal threshold (train F1)  : {self.optimal_threshold:.4f}")
        train_rate = self.train_flagged_count / self.train_rows if self.train_rows else 0
        test_rate = self.test_flagged_count / self.test_rows if self.test_rows else 0
        print(f"Train flagged (tuned thresh)  : {self.train_flagged_count} ({train_rate:.2%})")
        print(f"Test flagged (tuned thresh)   : {self.test_flagged_count} ({test_rate:.2%})")
        print(f"Test precision                : {self.test_precision:.4f}")
        print(f"Test recall                   : {self.test_recall:.4f}")
        print(f"Test F1                       : {self.test_f1:.4f}")
        print(f"Test AUC-ROC                  : {self.test_auc_roc:.4f}")
        print(f"Test AUC-PR                   : {self.test_auc_pr:.4f}")
        print("Top global features (mean |SHAP|):")
        for feat, val in self.top_global_features.items():
            print(f"  {feat:<35}: {val:.4f}")
        print("------------------------------------------------\n")
