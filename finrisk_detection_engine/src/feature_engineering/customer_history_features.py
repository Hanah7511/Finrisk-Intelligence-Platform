from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_FEATURE_NAMES: List[str] = [
    "customer_txn_count_30d",
    "customer_avg_amount_30d",
    "customer_unique_devices_30d",
    "customer_unique_countries_30d",
    "customer_unique_merchants_30d",
    "customer_failed_payments_7d",
    "transaction_frequency_24h",
    "transaction_frequency_7d",
    "new_device_flag",
    "new_country_flag",
]

WINDOW_30D = np.timedelta64(30, "D")
WINDOW_7D = np.timedelta64(7, "D")
WINDOW_24H = np.timedelta64(24, "h")


class _TrailingDistinctCounter:
    """Tracks distinct-value count within a sliding window via incremental add/remove."""

    def __init__(self) -> None:
        self._counts: dict = {}

    def add(self, value) -> None:
        self._counts[value] = self._counts.get(value, 0) + 1

    def remove(self, value) -> None:
        self._counts[value] -= 1
        if self._counts[value] <= 0:
            del self._counts[value]

    def distinct(self) -> int:
        return len(self._counts)


def _compute_group(
    group: pd.DataFrame,
    amount_col: str,
    device_col: str,
    country_col: str,
    merchant_diversity_col: Optional[str],
    status_col: Optional[str],
    timestamp_col: str,
) -> pd.DataFrame:
    group = group.sort_values(timestamp_col)
    times = group[timestamp_col].to_numpy()
    amounts = group[amount_col].to_numpy(dtype=float)
    devices = group[device_col].to_numpy()
    countries = group[country_col].to_numpy()

    has_merchants = merchant_diversity_col is not None and merchant_diversity_col in group.columns
    merchants = group[merchant_diversity_col].to_numpy(dtype=float) if has_merchants else None

    has_status = status_col is not None and status_col in group.columns
    failed = (group[status_col].to_numpy() == "failed") if has_status else np.zeros(len(group), dtype=bool)

    n = len(group)
    txn_count_30d = np.zeros(n)
    avg_amount_30d = np.zeros(n)
    unique_devices_30d = np.zeros(n)
    unique_countries_30d = np.zeros(n)
    # Proxy, not a true cross-order distinct-merchant count: no per-transaction merchant
    # identity column exists, only an order-level "distinct merchants in this order" count.
    # We sum that order-level stat across the trailing 30d window as a merchant-exposure proxy.
    merchant_exposure_30d = np.zeros(n)
    failed_count_7d = np.zeros(n)
    freq_24h = np.zeros(n)
    freq_7d = np.zeros(n)
    new_device_flag = np.zeros(n, dtype=int)
    new_country_flag = np.zeros(n, dtype=int)

    start_30 = start_7 = start_24h = 0
    device_counter = _TrailingDistinctCounter()
    country_counter = _TrailingDistinctCounter()
    seen_devices: set = set()
    seen_countries: set = set()
    running_amount_sum_30d = 0.0
    running_merchant_sum_30d = 0.0
    running_failed_7d = 0

    for i in range(n):
        t = times[i]

        while start_30 < i and times[start_30] < t - WINDOW_30D:
            running_amount_sum_30d -= amounts[start_30]
            device_counter.remove(devices[start_30])
            country_counter.remove(countries[start_30])
            if has_merchants:
                running_merchant_sum_30d -= merchants[start_30]
            start_30 += 1

        while start_7 < i and times[start_7] < t - WINDOW_7D:
            running_failed_7d -= int(failed[start_7])
            start_7 += 1

        while start_24h < i and times[start_24h] < t - WINDOW_24H:
            start_24h += 1

        txn_count_30d[i] = i - start_30
        avg_amount_30d[i] = running_amount_sum_30d / txn_count_30d[i] if txn_count_30d[i] > 0 else 0.0
        unique_devices_30d[i] = device_counter.distinct()
        unique_countries_30d[i] = country_counter.distinct()
        merchant_exposure_30d[i] = running_merchant_sum_30d
        failed_count_7d[i] = running_failed_7d
        freq_24h[i] = i - start_24h
        freq_7d[i] = i - start_7
        new_device_flag[i] = 0 if devices[i] in seen_devices else 1
        new_country_flag[i] = 0 if countries[i] in seen_countries else 1

        # Fold current row into history for subsequent rows (strictly-prior-only guarantee).
        running_amount_sum_30d += amounts[i]
        device_counter.add(devices[i])
        country_counter.add(countries[i])
        if has_merchants:
            running_merchant_sum_30d += merchants[i]
        running_failed_7d += int(failed[i])
        seen_devices.add(devices[i])
        seen_countries.add(countries[i])

    group = group.copy()
    group["customer_txn_count_30d"] = txn_count_30d
    group["customer_avg_amount_30d"] = avg_amount_30d
    group["customer_unique_devices_30d"] = unique_devices_30d
    group["customer_unique_countries_30d"] = unique_countries_30d
    group["customer_unique_merchants_30d"] = merchant_exposure_30d
    group["customer_failed_payments_7d"] = failed_count_7d
    group["transaction_frequency_24h"] = freq_24h
    group["transaction_frequency_7d"] = freq_7d
    group["new_device_flag"] = new_device_flag
    group["new_country_flag"] = new_country_flag
    return group


def compute_customer_history_features(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    device_col: str = "device_id",
    country_col: str = "counterparty_country",
    amount_col: str = "amount_usd_equivalent",
    merchant_diversity_col: str = "unique_merchants",
    status_col: str = "payment_status",
    timestamp_col: str = "payment_timestamp",
) -> pd.DataFrame:
    """Recomputes customer-history aggregates from raw row-level history.

    Must run on the full time-sorted dataset before any train/test split, and before
    any encoding/log-transform of device_id, counterparty_country, or amount_usd_equivalent
    (those raw values are required to track identity and true scale). Each row only uses
    strictly-prior transactions for that customer, so this is safe to compute across the
    whole dataset without leaking a row's own future into its own features.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    parts = [
        _compute_group(g, amount_col, device_col, country_col, merchant_diversity_col, status_col, timestamp_col)
        for _, g in df.groupby(customer_col, sort=False)
    ]
    result = pd.concat(parts).sort_values(timestamp_col).reset_index(drop=True)

    logger.info(
        "Computed customer history features for %d customers, %d rows",
        df[customer_col].nunique(), len(result),
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rng = np.random.default_rng(42)
    n = 20
    dummy = pd.DataFrame({
        "customer_id": rng.choice(["C1", "C2"], n),
        "payment_timestamp": pd.date_range("2026-01-01", periods=n, freq="18h"),
        "device_id": rng.choice(["DEV_1", "DEV_2"], n),
        "counterparty_country": rng.choice(["India", "Germany"], n),
        "amount_usd_equivalent": rng.exponential(200, n),
        "payment_status": rng.choice(["success", "failed"], n, p=[0.8, 0.2]),
        "unique_merchants": rng.integers(1, 4, n),
    })
    result = compute_customer_history_features(dummy)
    print(result[[
        "customer_id", "payment_timestamp", "customer_txn_count_30d",
        "customer_avg_amount_30d", "new_device_flag", "new_country_flag",
    ]])
