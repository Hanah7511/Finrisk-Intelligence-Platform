from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from configs.loader import load_thresholds_config
from src.supervised_detection.report import SupervisedDetectionReport
from src.supervised_detection.model import (
    get_feature_cols,
    compute_scale_pos_weight,
    fit_xgboost,
    predict_fraud_probability,
    find_best_threshold,
    compute_classification_metrics,
    compute_shap_values,
    top_shap_features_per_row,
    global_top_shap_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_config = load_thresholds_config()["supervised"]["xgboost"]
N_ESTIMATORS: int = _config["n_estimators"]
MAX_DEPTH: int = _config["max_depth"]
LEARNING_RATE: float = _config["learning_rate"]
RANDOM_STATE: int = _config["random_state"]

TARGET_COL = "analyst_confirmed_label"


def run_supervised_detection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, SupervisedDetectionReport]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    report = SupervisedDetectionReport()
    report.train_rows = len(train_df)
    report.test_rows = len(test_df)

    feature_cols = get_feature_cols(train_df)
    report.feature_count = len(feature_cols)

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values

    report.train_positive_count = int(y_train.sum())
    report.test_positive_count = int(y_test.sum())

    scale_pos_weight = compute_scale_pos_weight(train_df[TARGET_COL])
    report.scale_pos_weight = scale_pos_weight

    model = fit_xgboost(
        X_train, y_train,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    train_df["fraud_probability"] = predict_fraud_probability(model, X_train)
    test_df["fraud_probability"] = predict_fraud_probability(model, X_test)

    optimal_threshold = find_best_threshold(y_train, train_df["fraud_probability"].values)
    report.optimal_threshold = optimal_threshold

    train_df["fraud_prediction_flag"] = (train_df["fraud_probability"] >= optimal_threshold).astype(int)
    test_df["fraud_prediction_flag"] = (test_df["fraud_probability"] >= optimal_threshold).astype(int)

    report.train_flagged_count = int(train_df["fraud_prediction_flag"].sum())
    report.test_flagged_count = int(test_df["fraud_prediction_flag"].sum())

    metrics = compute_classification_metrics(y_test, test_df["fraud_probability"].values, optimal_threshold)
    report.test_precision = metrics["precision"]
    report.test_recall = metrics["recall"]
    report.test_f1 = metrics["f1"]
    report.test_auc_roc = metrics["auc_roc"]
    report.test_auc_pr = metrics["auc_pr"]


    # SHAP on the test set only - per-alert explainability is about unseen
    # predictions, not training data.
    shap_values = compute_shap_values(model, X_test)
    test_df["top_shap_features"] = top_shap_features_per_row(shap_values, feature_cols)
    report.top_global_features = global_top_shap_features(shap_values, feature_cols)

    logger.info(
        "Supervised detection flagged %d/%d train rows and %d/%d test rows (%d features, scale_pos_weight=%.2f)",
        report.train_flagged_count, report.train_rows,
        report.test_flagged_count, report.test_rows,
        report.feature_count, scale_pos_weight,
    )

    return train_df, test_df, report


if __name__ == "__main__":
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

    train_out, test_out, report = run_supervised_detection(train, test)
    report.summary()
    print(test_out[["fraud_probability", "fraud_prediction_flag", "top_shap_features"]]
          .sort_values("fraud_probability", ascending=False).head(10))
