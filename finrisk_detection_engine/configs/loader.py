from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from configs.api_config import ApiConfig
from configs.sql_config import SqlServerConfig

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent


class ConfigLoadError(Exception):
    """Raised when a YAML config file fails to load or fails validation."""


def _load_yaml(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename

    if not path.exists():
        raise ConfigLoadError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        raise ConfigLoadError(f"Config file is empty: {path}")

    logger.info("Loaded config file: %s", path.name)
    return data


def load_app_config() -> Dict[str, Any]:
    return _load_yaml("app.yaml")


def load_data_sources_config() -> Dict[str, Any]:
    return _load_yaml("data_sources.yaml")


def load_models_config() -> Dict[str, Any]:
    return _load_yaml("models.yaml")


def load_thresholds_config() -> Dict[str, Any]:
    thresholds = _load_yaml("thresholds.yaml")

    weights = thresholds.get("risk_fusion", {}).get("weights", {})
    total = sum(weights.values())
    if weights and abs(total - 1.0) > 1e-6:
        raise ConfigLoadError(
            f"risk_fusion.weights must sum to 1.0, got {total:.4f}: {weights}"
        )

    return thresholds


def build_sql_config() -> SqlServerConfig:
    sql = load_data_sources_config()["sql"]
    return SqlServerConfig(
        server=sql["server"],
        database=sql["database"],
        table=sql["table"],
        driver=sql["driver"],
        trusted_connection=sql["trusted_connection"],
        pool_recycle=sql["pool_recycle"],
    )


def build_api_config(url: str, data_key: str | None = None) -> ApiConfig:
    api = load_data_sources_config()["api"]
    return ApiConfig(
        url=url,
        timeout=api["timeout"],
        retries=api["retries"],
        backoff_factor=api["backoff_factor"],
        data_key=data_key,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print(build_sql_config())
    print(load_thresholds_config()["risk_fusion"]["weights"])