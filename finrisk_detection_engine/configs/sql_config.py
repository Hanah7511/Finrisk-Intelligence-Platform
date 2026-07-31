from dataclasses import dataclass


@dataclass(frozen=True)
class SqlServerConfig:
    server: str = "localhost\\SQLEXPRESS"
    database: str = "FinRiskIntelDB"
    table: str = "dbo.finrisk_modeling_dataset"
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True
    pool_recycle: int = 1800
from dataclasses import dataclass


@dataclass(frozen=True)
class SqlServerConfig:
    server: str = "localhost\\SQLEXPRESS"
    database: str = "FinRiskIntelDB"
    table: str = "dbo.finrisk_modeling_dataset"
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True
    pool_recycle: int = 1800


DEFAULT_SQL_CONFIG = SqlServerConfig()