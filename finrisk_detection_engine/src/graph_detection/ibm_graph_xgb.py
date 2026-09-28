from __future__ import annotations

"""
Graph features + XGBoost on the IBM AML HI-Small benchmark.

IBM's labels are real ground truth (unlike this project's simulated
production labels), so a supervised model is legitimate here.

Leakage rules, same as every other layer:
- Time-based split: train Sept 1-5, validation Sept 6, test Sept 7 onward.
- Graph features use a TRAILING window only (the transaction's day and the
  3 days before it) - never future transactions.
- The decision threshold is tuned on validation, never on test.

Ablation: the same model is trained twice - transaction features only, then
transaction + graph features - so the gain from the graph is measured, not
assumed.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score,
)
from xgboost import XGBClassifier

from src.graph_detection.ibm_aml_loader import (
    load_ibm_patterns, load_ibm_transactions,
)

logger = logging.getLogger(__name__)

WINDOW_DAYS = 4          # every fan-in/fan-out attempt lasts <= 96 hours
TRAIN_END_DAY = 5        # days 0-4  = Sept 1-5
VAL_END_DAY = 6          # day 5     = Sept 6
START = pd.Timestamp("2022-09-01")

TXN_FEATURES = ["log_amount", "payment_format_code", "hour", "cross_bank"]
GRAPH_FEATURES = [
    "src_out_degree", "src_out_count", "src_in_degree",
    "dst_in_degree", "dst_in_count", "dst_out_degree",
    "src_passthrough", "dst_passthrough", "reciprocal_edge",
]


@dataclass
class ModelResult:
    name: str
    threshold: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_pr: float = 0.0
    flagged: int = 0


@dataclass
class IbmGraphReport:
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    test_positives: int = 0
    test_base_rate: float = 0.0
    results: List[ModelResult] = field(default_factory=list)
    attempt_recall_by_typology: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> None:
        print("\n--- IBM AML benchmark: graph features + XGBoost -----")
        print(f"Train / val / test rows : {self.train_rows:,} / {self.val_rows:,} / {self.test_rows:,}")
        print(f"Test laundering txns    : {self.test_positives:,} (base rate {self.test_base_rate:.4%})")
        print(f"\n{'model':<22}{'precision':>10}{'recall':>9}{'F1':>8}{'AUC-PR':>9}{'flagged':>10}")
        for r in self.results:
            print(f"{r.name:<22}{r.precision:>10.3f}{r.recall:>9.3f}{r.f1:>8.3f}"
                  f"{r.auc_pr:>9.3f}{r.flagged:>10,}")
        print("\nLaundering attempts caught on test (>=1 txn flagged), graph model:")
        for typ, val in self.attempt_recall_by_typology.items():
            print(f"  {typ:<16}: {val}")
        print("------------------------------------------------\n")


def add_transaction_features(t: pd.DataFrame) -> pd.DataFrame:
    t["day"] = (t["timestamp"].dt.normalize() - START).dt.days
    t["hour"] = t["timestamp"].dt.hour
    t["log_amount"] = np.log1p(t["amount_paid"])
    t["payment_format_code"] = t["payment_format"].astype("category").cat.codes
    t["cross_bank"] = (t["from_node"].str.split("_").str[0]
                       != t["to_node"].str.split("_").str[0]).astype(int)
    return t


def add_graph_features(t: pd.DataFrame) -> pd.DataFrame:
    """Per account per day, over the trailing WINDOW_DAYS days (inclusive).
    Nodes are factorized to ints first - string groupbys on 4.5M rows are slow."""
    codes, _ = pd.factorize(pd.concat([t["from_node"], t["to_node"]]))
    t["src"] = codes[: len(t)]
    t["dst"] = codes[len(t):]

    per_day = []
    for d in range(int(t["day"].max()) + 1):
        w = t[(t["day"] > d - WINDOW_DAYS) & (t["day"] <= d)]
        today = t.loc[t["day"] == d, ["src", "dst"]]
        if today.empty:
            continue
        out_deg = w.groupby("src")["dst"].nunique()
        out_cnt = w.groupby("src").size()
        in_deg = w.groupby("dst")["src"].nunique()
        in_cnt = w.groupby("dst").size()

        feats = pd.DataFrame(index=today.index)
        feats["src_out_degree"] = today["src"].map(out_deg).fillna(0).values
        feats["src_out_count"] = today["src"].map(out_cnt).fillna(0).values
        feats["src_in_degree"] = today["src"].map(in_deg).fillna(0).values
        feats["dst_in_degree"] = today["dst"].map(in_deg).fillna(0).values
        feats["dst_in_count"] = today["dst"].map(in_cnt).fillna(0).values
        feats["dst_out_degree"] = today["dst"].map(out_deg).fillna(0).values

        # Did money also flow the other way (dst -> src) inside the window?
        # A 2-hop cycle, the smallest round-trip.
        reverse = pd.MultiIndex.from_frame(
            w[["dst", "src"]].drop_duplicates().rename(columns={"dst": "a", "src": "b"}))
        feats["reciprocal_edge"] = pd.MultiIndex.from_frame(today).isin(reverse).astype(int)
        per_day.append(feats)

    g = pd.concat(per_day)
    t = t.join(g)
    # Pass-through: an account both receiving from and sending to many -
    # the mule / intermediary shape in stack and scatter-gather patterns.
    t["src_passthrough"] = np.minimum(t["src_in_degree"], t["src_out_degree"])
    t["dst_passthrough"] = np.minimum(t["dst_in_degree"], t["dst_out_degree"])
    return t


def _tune_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y, probs)
    denom = precisions[:-1] + recalls[:-1]
    f1 = np.divide(2 * precisions[:-1] * recalls[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def fit_and_score(name: str, features: List[str], train: pd.DataFrame,
                  val: pd.DataFrame, test: pd.DataFrame):
    y_train = train["is_laundering"].values
    spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="aucpr",
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(train[features], y_train)

    threshold = _tune_threshold(val["is_laundering"].values,
                                model.predict_proba(val[features])[:, 1])
    probs = model.predict_proba(test[features])[:, 1]
    preds = (probs >= threshold).astype(int)
    y = test["is_laundering"].values
    result = ModelResult(
        name=name, threshold=threshold,
        precision=precision_score(y, preds, zero_division=0),
        recall=recall_score(y, preds, zero_division=0),
        f1=f1_score(y, preds, zero_division=0),
        auc_pr=average_precision_score(y, probs),
        flagged=int(preds.sum()),
    )
    return result, preds


def attempt_recall(test: pd.DataFrame, preds: np.ndarray,
                   patterns: pd.DataFrame) -> Dict[str, str]:
    """An attempt counts as caught if at least one of its test-period
    transactions was flagged - an analyst who sees one leg of a ring can
    pull the rest of it."""
    flagged = test.loc[preds == 1, ["timestamp", "from_node", "to_node", "amount_paid"]]
    key = ["timestamp", "from_node", "to_node", "amount_paid"]
    p = patterns[patterns["timestamp"] >= START + pd.Timedelta(days=VAL_END_DAY)]
    p = p.merge(flagged.assign(hit=1).drop_duplicates(key), on=key, how="left")
    caught = p.groupby(["typology", "attempt_id"])["hit"].max().fillna(0)
    out = {}
    for typ, s in caught.groupby(level="typology"):
        out[typ] = f"{int(s.sum())}/{len(s)} ({s.mean():.0%})"
    return out


def run_ibm_graph_benchmark() -> IbmGraphReport:
    t = load_ibm_transactions()
    patterns = load_ibm_patterns()
    t = add_transaction_features(t)
    t = add_graph_features(t)

    train = t[t["day"] < TRAIN_END_DAY]
    val = t[(t["day"] >= TRAIN_END_DAY) & (t["day"] < VAL_END_DAY)]
    test = t[t["day"] >= VAL_END_DAY]

    report = IbmGraphReport(
        train_rows=len(train), val_rows=len(val), test_rows=len(test),
        test_positives=int(test["is_laundering"].sum()),
        test_base_rate=float(test["is_laundering"].mean()),
    )
    base_result, _ = fit_and_score("transaction only", TXN_FEATURES, train, val, test)
    graph_result, graph_preds = fit_and_score(
        "transaction + graph", TXN_FEATURES + GRAPH_FEATURES, train, val, test)
    report.results = [base_result, graph_result]
    report.attempt_recall_by_typology = attempt_recall(test, graph_preds, patterns)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ibm_graph_benchmark().summary()