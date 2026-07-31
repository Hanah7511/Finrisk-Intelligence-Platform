from __future__ import annotations

import logging

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from configs.api_config import ApiConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ApiIngestionError(Exception):
    """Raised when API ingestion fails."""


def build_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """
    Build a requests Session with retry logic.
    Retries on 500, 502, 503, 504 (server errors only).
    Does NOT retry on 400, 401, 404 (client errors — your fault, not server).
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _to_dataframe(data: object, config: ApiConfig) -> pd.DataFrame:
    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        if config.data_key is not None:
            if config.data_key not in data:
                raise ApiIngestionError(
                    f"Configured data_key '{config.data_key}' not found in "
                    f"response from {config.url}"
                )
            return pd.DataFrame(data[config.data_key])

        for key in ["data", "records", "results", "transactions", "items"]:
            if key in data:
                logger.info("Auto-detected data key: '%s'", key)
                return pd.DataFrame(data[key])

        return pd.DataFrame([data])

    raise ApiIngestionError(
        f"Unexpected response format from {config.url}: {type(data)}"
    )


def load_data_from_api(config: ApiConfig) -> pd.DataFrame:
    """
    Load data from a REST API endpoint into a Pandas DataFrame.

    Args:
        config: ApiConfig with URL, headers, params, timeout, retry settings.

    Returns:
        Pandas DataFrame with API response data.

    Raises:
        ApiIngestionError: If the API call fails for any reason.
    """
    with build_session(retries=config.retries, backoff_factor=config.backoff_factor) as session:
        logger.info("Calling API endpoint: %s | params: %s", config.url, config.params)

        try:
            response = session.get(
                config.url,
                headers=config.headers,
                params=config.params,
                timeout=config.timeout,
            )
            response.raise_for_status()

        except requests.exceptions.Timeout as exc:
            logger.error("API request timed out: %s", config.url)
            raise ApiIngestionError(
                f"Request timed out after {config.timeout}s: {config.url}"
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            logger.error("Cannot connect to API: %s", config.url)
            raise ApiIngestionError(f"Connection failed: {config.url}") from exc

        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error %s from API: %s", response.status_code, config.url)
            raise ApiIngestionError(
                f"HTTP {response.status_code} from {config.url}"
            ) from exc

        except requests.exceptions.RequestException as exc:
            logger.error("Request to API failed: %s", config.url)
            raise ApiIngestionError(f"Request failed: {config.url}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("Failed to parse JSON response from: %s", config.url)
            raise ApiIngestionError(f"Invalid JSON response from {config.url}") from exc

        df = _to_dataframe(data, config)

    if df.empty:
        logger.warning("API returned empty dataset from: %s", config.url)
    else:
        logger.info("API ingestion successful. Shape=%s | URL=%s", df.shape, config.url)

    return df


if __name__ == "__main__":
    # Test with a free public API (JSONPlaceholder)
    config = ApiConfig(
        url="https://jsonplaceholder.typicode.com/users",
        timeout=10,
        retries=3,
        backoff_factor=0.3,
    )

    try:
        df = load_data_from_api(config)
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(3))
    except ApiIngestionError as e:
        print(f"Ingestion failed: {e}")