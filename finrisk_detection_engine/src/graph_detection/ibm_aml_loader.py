
from __future__ import annotations

"""
Loader for the IBM "Transactions for Anti Money Laundering" HI-Small files
(Altman et al., AMLworld). External benchmark only - used to validate the
Layer 10 ring-detection method on independently-published laundering
patterns. Never mixed into dbo.finrisk_modeling_dataset.

Two quirks this handles:
- Account IDs are only unique within a bank, so a node is bank + account.
- Bank codes have leading zeros ("010"), so everything is read as str.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

IBM_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "ibm_aml"
TRANS_PATH = IBM_DIR / "HI-Small_Trans.csv"
PATTERNS_PATH = IBM_DIR / "HI-Small_Patterns.txt"

# The CSV header repeats "Account" twice, so names are set by position.
_COLUMNS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid",
    "payment_currency", "payment_format", "is_laundering",
]
_BEGIN_RE = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)")


def _add_node_ids(df: pd.DataFrame) -> pd.DataFrame:
    df["from_node"] = df["from_bank"] + "_" + df["from_account"]
    df["to_node"] = df["to_bank"] + "_" + df["to_account"]
    return df


def load_ibm_transactions(
    path: Path = TRANS_PATH,
    nrows: Optional[int] = None,
    drop_self_transfers: bool = True,
) -> pd.DataFrame:
    """Self-transfers (same bank+account, mostly 'Reinvestment') are dropped
    by default - a self-loop carries no relationship between accounts."""
    df = pd.read_csv(
        path,
        header=0,
        names=_COLUMNS,
        usecols=["timestamp", "from_bank", "from_account", "to_bank",
                 "to_account", "amount_paid", "payment_format", "is_laundering"],
        dtype={"from_bank": str, "from_account": str,
               "to_bank": str, "to_account": str},
        nrows=nrows,
    )
    df = _add_node_ids(df)
    if drop_self_transfers:
        before = len(df)
        df = df[df["from_node"] != df["to_node"]].reset_index(drop=True)
        logger.info("Dropped %d self-transfers", before - len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    return df[["timestamp", "from_node", "to_node", "amount_paid",
               "payment_format", "is_laundering"]]


def load_ibm_patterns(path: Path = PATTERNS_PATH) -> pd.DataFrame:
    """One row per transaction inside a laundering attempt, tagged with
    attempt_id and typology (FAN-IN, CYCLE, STACK, ...). This is the answer
    key for ring-level evaluation. It covers only the pattern-based
    laundering - Is Laundering in the CSV flags more transactions than these.
    """
    rows = []
    attempt_id, typology = -1, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = _BEGIN_RE.match(line)
            if match:
                attempt_id += 1
                typology = match.group(1)
                continue
                attempt_id += 1
                typology = match.group(1)
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                typology = None
                continue
            parts = line.split(",")
            rows.append({
                "attempt_id": attempt_id,
                "typology": typology,
                "timestamp": parts[0],
                "from_bank": parts[1], "from_account": parts[2],
                "from_bank": parts[1], "from_account": parts[2],
                "to_bank": parts[3], "to_account": parts[4],
                "amount_paid": float(parts[7]),
            })
    df = _add_node_ids(pd.DataFrame(rows))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    return df[["attempt_id", "typology", "timestamp", "from_node",
               "to_node", "amount_paid"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    patterns = load_ibm_patterns()
    print(f"Attempts: {patterns['attempt_id'].nunique()}  "
          f"pattern transactions: {len(patterns)}")
    print(patterns.groupby("typology")["attempt_id"].nunique())

    trans = load_ibm_transactions()
    print(f"\nTransactions (self-transfers dropped): {len(trans):,}")
    print(f"Unique accounts: {pd.concat([trans['from_node'], trans['to_node']]).nunique():,}")
    print(f"Laundering rate: {trans['is_laundering'].mean():.4%}")