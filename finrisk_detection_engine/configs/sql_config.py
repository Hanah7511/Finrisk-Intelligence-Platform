from dataclasses import dataclass


@dataclass(frozen=True)
class SqlServerConfig:
    server: str = "localhost\\SQLEXPRESS"
    database: str = "FinRiskIntelDB"
    table: str = "dbo.finrisk_modeling_dataset"
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True
    pool_recycle: int = 1800
    amount_unit_multiplier: float = 1000.0


DEFAULT_SQL_CONFIG = SqlServerConfig()

    # Amount columns in this table are stored in THOUSANDS of USD. Inferred, not
    # documented: a median of 0.8 and 25th percentile of 0.1 would otherwise mean
    # $0.80 / $0.10 payments. Multiplied once at load so every rule compares real
    # dollars against real-dollar thresholds.