from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from configs.loader import load_thresholds_config
from src.unsupervised_detection.report import UnsupervisedDetectionReport
from src.unsupervised_detection.ensemble import (
    get_feature_cols,
    fit_scaler,
    scale_features,
    fit_isolation_forest,
    fit_ocsvm,
    get_scores,
    fit_minmax_stats,
    normalize_and_invert,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_config = load_thresholds_config()["unsupervised"]
CONTAMINATION: float = _config["contamination"]


def run_unsupervised_detection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, UnsupervisedDetectionReport]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    report = UnsupervisedDetectionReport()
    report.train_rows = len(train_df)
    report.test_rows = len(test_df)
    report.contamination = CONTAMINATION

    feature_cols = get_feature_cols(train_df)
    report.feature_count = len(feature_cols)

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Isolation Forest - unscaled, tree splits are scale-invariant
    iforest = fit_isolation_forest(X_train, contamination=CONTAMINATION)
    train_if_raw = get_scores(iforest, X_train)
    test_if_raw = get_scores(iforest, X_test)

    # One-Class SVM - needs scaled input (kernel/distance-based)
    scaler = fit_scaler(train_df, feature_cols)
    X_train_scaled = scale_features(scaler, train_df, feature_cols)
    X_test_scaled = scale_features(scaler, test_df, feature_cols)

    ocsvm = fit_ocsvm(X_train_scaled, nu=CONTAMINATION)
    train_ocsvm_raw = get_scores(ocsvm, X_train_scaled)
    test_ocsvm_raw = get_scores(ocsvm, X_test_scaled)

    # Score-level fusion: normalize each model's raw scores to [0,1] using
    # train-only fit bounds, invert so higher = more anomalous, then average
    # the two into one combined score. Not decision-level (OR-of-flags) -
    # that pattern is correct for Layer 5's inherently binary rule outputs,
    # but throws away information here since both models output continuous
    # scores. Matches Layer 9 (Risk Fusion)'s own weighted-score philosophy.
    if_min, if_max = fit_minmax_stats(train_if_raw)
    train_if_norm = normalize_and_invert(train_if_raw, if_min, if_max)
    test_if_norm = normalize_and_invert(test_if_raw, if_min, if_max)

    ocsvm_min, ocsvm_max = fit_minmax_stats(train_ocsvm_raw)
    train_ocsvm_norm = normalize_and_invert(train_ocsvm_raw, ocsvm_min, ocsvm_max)
    test_ocsvm_norm = normalize_and_invert(test_ocsvm_raw, ocsvm_min, ocsvm_max)

    train_df["isolation_forest_score"] = train_if_norm
    test_df["isolation_forest_score"] = test_if_norm
    train_df["ocsvm_score"] = train_ocsvm_norm
    test_df["ocsvm_score"] = test_ocsvm_norm

    train_df["unsupervised_anomaly_score"] = (train_if_norm + train_ocsvm_norm) / 2
    test_df["unsupervised_anomaly_score"] = (test_if_norm + test_ocsvm_norm) / 2

    # Flag threshold = train's own top-contamination quantile of the combined
    # score, reusing the same contamination number a third time (see
    # thresholds.yaml comment) instead of introducing a fourth, independent cutoff.
    cutoff = np.quantile(train_df["unsupervised_anomaly_score"], 1 - CONTAMINATION)
    train_df["unsupervised_anomaly_flag"] = (train_df["unsupervised_anomaly_score"] >= cutoff).astype(int)
    test_df["unsupervised_anomaly_flag"] = (test_df["unsupervised_anomaly_score"] >= cutoff).astype(int)

    report.train_flagged_count = int(train_df["unsupervised_anomaly_flag"].sum())
    report.test_flagged_count = int(test_df["unsupervised_anomaly_flag"].sum())

    logger.info(
        "Unsupervised detection flagged %d/%d train rows and %d/%d test rows (%d features)",
        report.train_flagged_count, report.train_rows,
        report.test_flagged_count, report.test_rows,
        report.feature_count,
    )

    return train_df, test_df, report


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 200
    dummy = pd.DataFrame({
        "payment_id": range(n),
        "customer_id": rng.choice(["C1", "C2", "C3", "C4"], n),
        "payment_timestamp": pd.date_range("2026-01-01", periods=n, freq="4h"),
        "amount_usd_equivalent": np.concatenate([rng.exponential(2, 180), rng.uniform(50, 100, 20)]),
        "kyc_level": rng.integers(0, 3, n),
        "transaction_hour": rng.integers(0, 24, n),
        "customer_txn_count_30d": rng.integers(0, 20, n),
    })
    train = dummy.iloc[:150].reset_index(drop=True)
    test = dummy.iloc[150:].reset_index(drop=True)

    train_out, test_out, report = run_unsupervised_detection(train, test)
    report.summary()
    print(train_out[["amount_usd_equivalent", "unsupervised_anomaly_score", "unsupervised_anomaly_flag"]]
          .sort_values("unsupervised_anomaly_score", ascending=False).head(10))
