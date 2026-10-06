from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from configs.sql_config import (
    DEFAULT_SQL_CONFIG, 
    SqlServerConfig,
)



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Money columns converted from the table's stored unit to real USD at load.
AMOUNT_COLUMNS = ["amount_usd_equivalent", "customer_avg_amount_30d"]


def _to_real_usd(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    for col in AMOUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col] * multiplier
    return df


class SqlIngestionError(Exception):
    """Raised when SQL ingestion fails."""


def build_connection_url(config: SqlServerConfig) -> str:
    if not config.trusted_connection:
        raise SqlIngestionError("This version supports trusted connection only.")

    return (
        f"mssql+pyodbc://@{config.server}/{config.database}"
        f"?driver={config.driver.replace(' ', '+')}"
        f"&trusted_connection=yes"
    )


def get_engine(config: SqlServerConfig) -> Engine:
    try:
        engine = create_engine(
            build_connection_url(config),
            pool_pre_ping=True,
            pool_recycle=config.pool_recycle,
            future=True,
        )
        logger.info(
            "SQLAlchemy engine created for server=%s database=%s",
            config.server,
            config.database,
        )
        return engine
    except Exception as exc:
        logger.exception("Failed to create SQLAlchemy engine.")
        raise SqlIngestionError("Failed to create SQL engine.") from exc


def load_finrisk_modeling_dataset(
    config: Optional[SqlServerConfig] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    config = config or DEFAULT_SQL_CONFIG
    engine = get_engine(config)

    query = f"SELECT * FROM {config.table}"
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        query = f"SELECT TOP ({limit}) * FROM {config.table}"

    try:
        logger.info("Loading dataset from table=%s", config.table)

        with engine.connect() as connection:
            df = pd.read_sql_query(text(query), connection)

        if df.empty:
            logger.warning("Query returned zero rows from %s", config.table)
        else:
            logger.info("Loaded dataset successfully with shape=%s", df.shape)
            logger.info("Columns loaded: %s", df.columns.tolist())

        return _to_real_usd(df, config.amount_unit_multiplier)

    except Exception as exc:
        logger.exception("Failed to load dataset from %s", config.table)
        raise SqlIngestionError(
            f"Failed to load dataset from table '{config.table}'."
        ) from exc


if __name__ == "__main__":
    df = load_finrisk_modeling_dataset(limit=5)
    print(df.head())