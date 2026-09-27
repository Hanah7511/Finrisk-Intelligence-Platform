from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

# Exclude-list approach (opposite direction from Layer 6, which needed the
# raw pre-log-transform amount column specifically). Everything NOT listed
# here becomes a feature - encoded categoricals, timestamp features, the
# log-transformed amount, engineered ratio/history features.
FEATURE_EXCLUDE_COLS = [
    # identifiers - carry no generalizable signal, would just let the model
    # memorize individual customers/payments instead of learning behavior
    "payment_id", "order_id", "customer_id", "counterpart_account_id" ,"payment_timestamp",
    # target + leakage-risk metadata
    "fraud_label", "fraud_rule_name", "label_source",
    # Layer 5 (rule engine) outputs - excluded so this layer doesn't just
    # re-learn Layer 5's rules; Layer 9 (Risk Fusion) is where layer outputs
    # get combined, not here
    "velocity_score", "geo_mismatch_flag", "sanctions_country_flag",
    "high_risk_corridor_flag", "structuring_flag",
    # Layer 6 (statistical detection) outputs - same reasoning as above, plus
    # the raw (pre-log-transform) amount column Layer 6 needed: it duplicates
    # the log-transformed amount already in the feature set and is unscaled,
    # which would reintroduce OC-SVM's scale-sensitivity problem
    "amount_usd_equivalent_raw", "amount_usd_equivalent_raw_mad_score",
    "amount_usd_equivalent_raw_mad_flag", "statistical_anomaly_flag",
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Exclude known non-feature columns, then keep only numeric dtypes.

    The dtype filter is a safety net, not the primary scoping mechanism -
    it catches anything that slipped through the exclude list (e.g. a
    forgotten string column) rather than silently feeding it to sklearn,
    which would error anyway but with a less obvious message.
    """
    candidates = [c for c in df.columns if c not in FEATURE_EXCLUDE_COLS]
    numeric_df = df[candidates].select_dtypes(include=[np.number])

    dropped_non_numeric = [c for c in candidates if c not in numeric_df.columns]
    if dropped_non_numeric:
        logger.warning(
            "Dropped non-numeric columns not in FEATURE_EXCLUDE_COLS: %s",
            dropped_non_numeric,
        )

    return list(numeric_df.columns)


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """Fit on train only - same leakage rule as every other fit step in
    this pipeline. Only needed for OC-SVM; Isolation Forest's tree splits
    are scale-invariant.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def scale_features(scaler: StandardScaler, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return scaler.transform(df[feature_cols])


def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float,
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X_train)
    return model


def fit_ocsvm(X_train_scaled: np.ndarray, nu: float) -> OneClassSVM:
    model = OneClassSVM(kernel="rbf", nu=nu)
    model.fit(X_train_scaled)
    return model


def get_scores(model, X: np.ndarray) -> np.ndarray:
    """Raw decision_function scores. Sklearn convention: higher = more
    normal, lower/negative = more anomalous - not yet normalized/inverted.
    """
    return model.decision_function(X)


def fit_minmax_stats(train_scores: np.ndarray) -> Tuple[float, float]:
    return float(train_scores.min()), float(train_scores.max())


def normalize_and_invert(
    scores: np.ndarray,
    fit_min: float,
    fit_max: float,
) -> np.ndarray:
    """Min-max normalize to [0, 1] using train-fit bounds, then invert so
    higher = more anomalous (opposite of sklearn's raw decision_function
    convention). Scores outside the train-fit range (possible on test data)
    are clipped rather than left unbounded, keeping the combined score
    comparable across train and test.
    """
    denom = fit_max - fit_min
    if denom == 0:
        normalized = np.zeros_like(scores, dtype=float)
    else:
        normalized = (scores - fit_min) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    return 1.0 - normalized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rng = np.random.default_rng(42)
    n = 100
    dummy = pd.DataFrame({
        "payment_id": range(n),
        "customer_id": rng.choice(["C1", "C2", "C3"], n),
        "payment_timestamp": pd.date_range("2026-01-01", periods=n, freq="6h"),
        "amount_usd_equivalent": np.concatenate([rng.exponential(2, 90), rng.uniform(50, 100, 10)]),
        "kyc_level": rng.integers(0, 3, n),
        "transaction_hour": rng.integers(0, 24, n),
    })
    train = dummy.iloc[:70].reset_index(drop=True)
    test = dummy.iloc[70:].reset_index(drop=True)

    feature_cols = get_feature_cols(train)
    print("Feature cols:", feature_cols)

    scaler = fit_scaler(train, feature_cols)
    X_train_scaled = scale_features(scaler, train, feature_cols)
    X_test_scaled = scale_features(scaler, test, feature_cols)

    iforest = fit_isolation_forest(train[feature_cols].values, contamination=0.10)
    train_if_scores = get_scores(iforest, train[feature_cols].values)
    test_if_scores = get_scores(iforest, test[feature_cols].values)

    fit_min, fit_max = fit_minmax_stats(train_if_scores)
    train_if_norm = normalize_and_invert(train_if_scores, fit_min, fit_max)
    test_if_norm = normalize_and_invert(test_if_scores, fit_min, fit_max)

    print("Train IF normalized (top 5):", np.sort(train_if_norm)[-5:])
    print("Test IF normalized:", test_if_norm)
