from __future__ import annotations
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

import json
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.unsupervised_detection.ensemble import FEATURE_EXCLUDE_COLS as _LAYER7_EXCLUDE_COLS

logger = logging.getLogger(__name__)

# Layer 7's exclude list, plus Layer 7's own output columns (this layer
# doesn't re-learn Layer 7's signal - Layer 9/Risk Fusion is where layer
# outputs get combined, not here) and the new target itself.
FEATURE_EXCLUDE_COLS = _LAYER7_EXCLUDE_COLS + [
    "isolation_forest_score", "ocsvm_score",
    "unsupervised_anomaly_score", "unsupervised_anomaly_flag",
    "analyst_confirmed_label",
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    candidates = [c for c in df.columns if c not in FEATURE_EXCLUDE_COLS]
    numeric_df = df[candidates].select_dtypes(include=[np.number])

    dropped_non_numeric = [c for c in candidates if c not in numeric_df.columns]
    if dropped_non_numeric:
        logger.warning(
            "Dropped non-numeric columns not in FEATURE_EXCLUDE_COLS: %s",
            dropped_non_numeric,
        )

    return list(numeric_df.columns)


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """neg/pos ratio, computed fresh each run rather than hardcoded in
    config - stays correct if feedback-simulation confirmation rates change.
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    random_state: int = 42,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def predict_fraud_probability(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]

def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Sweeps the precision-recall curve and returns the threshold that
    maximizes F1 - replaces the naive 0.5 cutoff, which combined with a
    heavily upweighted scale_pos_weight pushes far too many true negatives
    above 0.5 (confirmed on real data: 18% flagged vs. ~2% true rate).
    Fit on train only, same leakage rule as every other layer in this
    pipeline; applied as a fixed cutoff to test.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def compute_classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Threshold-dependent (precision/recall/F1) and threshold-independent
    (AUC-ROC, AUC-PR) metrics. AUC-PR matters more than AUC-ROC here given
    the ~2% base rate - PR curves are the standard choice for heavily
    imbalanced classification, since ROC-AUC can look deceptively good.
    """
    preds = (probs >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probs)),
        "auc_pr": float(average_precision_score(y_true, probs)),
    }

def _fix_base_score_for_shap(model: XGBClassifier) -> None:
    """XGBoost >=2.0 serializes base_score as a bracketed string (e.g.
    '[5E-1]', to support multi-output models), but this shap version's
    TreeExplainer calls float() on it directly and errors. Strip the
    brackets so SHAP can parse it - the numeric value is unchanged, only
    its string representation in the booster's config.
    """
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    param = config["learner"]["learner_model_param"]
    param["base_score"] = param["base_score"].strip("[]")
    booster.load_config(json.dumps(config))
    
def compute_shap_values(model: XGBClassifier, X_test: np.ndarray) -> np.ndarray:
    """SHAP on the test set only - per-alert explainability is about
    explaining predictions on unseen data, not training data.
    """

    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_test)


def top_shap_features_per_row(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 3,
) -> List[str]:
    """Per row, the top_n features with the largest |SHAP value| as a
    compact string - the per-alert explainability the architecture review
    flagged as missing (AML is a regulated, explainability-sensitive domain;
    a raw score alone isn't enough)."""
    results = []
    for row in shap_values:
        top_idx = np.argsort(np.abs(row))[::-1][:top_n]
        parts = [f"{feature_names[i]}({row[i]:+.3f})" for i in top_idx]
        results.append(", ".join(parts))
    return results


def global_top_shap_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> dict:
    """Mean |SHAP| per feature across the test set - the report-level
    summary that sits alongside the per-row column above."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    return {feature_names[i]: float(mean_abs[i]) for i in order}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rng = np.random.default_rng(42)
    n = 500
    dummy = pd.DataFrame({
        "payment_id": range(n),
        "customer_id": rng.choice(["C1", "C2", "C3"], n),
        "payment_timestamp": pd.date_range("2026-01-01", periods=n, freq="2h"),
        "amount_usd_equivalent": rng.exponential(2, n),
        "kyc_level": rng.integers(0, 3, n),
        "transaction_hour": rng.integers(0, 24, n),
        "analyst_confirmed_label": (rng.random(n) < 0.02).astype(int),
    })
    train = dummy.iloc[:400].reset_index(drop=True)
    test = dummy.iloc[400:].reset_index(drop=True)

    feature_cols = get_feature_cols(train)
    print("Feature cols:", feature_cols)

    X_train = train[feature_cols].values
    y_train = train["analyst_confirmed_label"].values
    X_test = test[feature_cols].values

    spw = compute_scale_pos_weight(train["analyst_confirmed_label"])
    print("scale_pos_weight:", spw)

    model = fit_xgboost(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=spw)
    probs = predict_fraud_probability(model, X_test)
    print("Test probabilities (top 5):", np.sort(probs)[-5:])

    shap_values = compute_shap_values(model, X_test)
    print("Top shap per row (first 3):", top_shap_features_per_row(shap_values, feature_cols)[:3])
    print("Global top features:", global_top_shap_features(shap_values, feature_cols))

