from __future__ import annotations

"""
Disclosed synthetic ring-injection benchmark for Layer 10.

NOT production data - never touches dbo.finrisk_modeling_dataset. This
generates a small self-contained transaction graph: mostly random
background noise (no ring structure - matches what real chance produces,
per the permutation test that found no signal in production data), plus a
handful of deliberately embedded, disclosed laundering typologies (fan-in,
cycle, stack) with a ground-truth ring_id. Used only to evaluate whether
the connected-components method actually finds rings when they exist.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RingBenchmarkReport:
    background_rows: int = 0
    planted_rings: int = 0
    planted_ring_members: int = 0
    detected_components: int = 0
    rings_fully_recovered: int = 0
    rings_partially_recovered: int = 0
    rings_missed: int = 0
    background_customers_falsely_grouped: int = 0

    def summary(self) -> None:
        print("\n--- Ring Detection Benchmark (synthetic, disclosed) -----")
        print(f"Background (noise) rows          : {self.background_rows}")
        print(f"Planted rings                     : {self.planted_rings}")
        print(f"Planted ring members (total)      : {self.planted_ring_members}")
        print(f"Detected graph components         : {self.detected_components}")
        print(f"Rings fully recovered             : {self.rings_fully_recovered}")
        print(f"Rings partially recovered         : {self.rings_partially_recovered}")
        print(f"Rings missed entirely             : {self.rings_missed}")
        print(f"Background customers falsely grouped into a ring-sized component: "
              f"{self.background_customers_falsely_grouped}")
        print("------------------------------------------------\n")


def build_background_noise(n_customers: int, seed: int = 42) -> pd.DataFrame:
    """Random customer -> random large-range counterparty, one row each.
    No overlap with customer IDs by design - mirrors genuinely unrelated
    external payments, the honest 'no ring' case."""
    r = np.random.default_rng(seed)
    customers = np.arange(1, n_customers + 1)
    counterparties = r.integers(100_000, 999_999, size=n_customers)
    return pd.DataFrame({
        "customer_id": customers,
        "counterparty_account_id": counterparties,
        "true_ring_id": None,
    })


def plant_fan_in_ring(ring_id: int, hub_id: int, sender_ids: List[int]) -> pd.DataFrame:
    """Several distinct customers all pay the same hub account - the
    classic 'funnel/mule collection account' pattern."""
    return pd.DataFrame({
        "customer_id": sender_ids,
        "counterparty_account_id": [hub_id] * len(sender_ids),
        "true_ring_id": ring_id,
    })


def plant_cycle_ring(ring_id: int, node_ids: List[int]) -> pd.DataFrame:
    """A -> B -> C -> A. Requires node_ids to act as both customer and
    counterparty across rows - the real 'round-trip layering' pattern."""
    senders = node_ids
    receivers = node_ids[1:] + node_ids[:1]
    return pd.DataFrame({
        "customer_id": senders,
        "counterparty_account_id": receivers,
        "true_ring_id": ring_id,
    })


def plant_stack_ring(ring_id: int, source: int, mules: List[int], sink: int) -> pd.DataFrame:
    """source -> each mule -> sink. Fan-out then fan-in through
    intermediary 'mule' accounts, a common layering chain."""
    rows = [{"customer_id": source, "counterparty_account_id": m, "true_ring_id": ring_id} for m in mules]
    rows += [{"customer_id": m, "counterparty_account_id": sink, "true_ring_id": ring_id} for m in mules]
    return pd.DataFrame(rows)


def build_benchmark_dataset(seed: int = 42) -> pd.DataFrame:
    background = build_background_noise(n_customers=300, seed=seed)
    rings = [
        plant_fan_in_ring(ring_id=1, hub_id=9001, sender_ids=[1001, 1002, 1003, 1004]),
        plant_cycle_ring(ring_id=2, node_ids=[2001, 2002, 2003]),
        plant_stack_ring(ring_id=3, source=3001, mules=[3002, 3003], sink=3004),
    ]
    return pd.concat([background, *rings], ignore_index=True)


def build_unified_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Single shared node space for customer_id and counterparty_account_id -
    required because the real schema confirms they overlap (a counterparty
    can also be a customer elsewhere), so cycles must merge correctly."""
    graph = nx.DiGraph()
    for cust, cp in df[["customer_id", "counterparty_account_id"]].itertuples(index=False):
        graph.add_edge(cust, cp)
    return graph


    return graph


def run_ring_benchmark(seed: int = 42) -> Tuple[pd.DataFrame, RingBenchmarkReport]:
    df = build_benchmark_dataset(seed=seed)
    report = RingBenchmarkReport()
    report.background_rows = int(df["true_ring_id"].isna().sum())
    report.planted_rings = int(df["true_ring_id"].dropna().nunique())
    report.planted_ring_members = int(df.loc[df["true_ring_id"].notna(), "customer_id"].nunique())

    graph = build_unified_graph(df).to_undirected()
    components = list(nx.connected_components(graph))
    report.detected_components = len(components)

    node_to_comp: Dict = {n: i for i, comp in enumerate(components) for n in comp}
    ring_customer_ids = set(df.loc[df["true_ring_id"].notna(), "customer_id"])

    for ring_id, ring_df in df.dropna(subset=["true_ring_id"]).groupby("true_ring_id"):
        members = set(ring_df["customer_id"]) | set(ring_df["counterparty_account_id"])
        comp_ids = {node_to_comp.get(m) for m in members}
        if len(comp_ids) == 1:
            report.rings_fully_recovered += 1
        elif len(comp_ids) == len(members):
            report.rings_missed += 1
        else:
            report.rings_partially_recovered += 1

    background_nodes = set(df.loc[df["true_ring_id"].isna(), "customer_id"])
    for comp in components:
        if len(comp) >= 3 and (comp & background_nodes) and not (comp & ring_customer_ids):
            report.background_customers_falsely_grouped += len(comp & background_nodes)

    return df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, report = run_ring_benchmark()
    report.summary()


if __name__ == "__main__":
     logging.basicConfig(level=logging.INFO)
     df, report = run_ring_benchmark()
     report.summary()