from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from configs.csv_config import CsvConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class CsvIngestionError(Exception):
    """Raised when CSV ingestion fails."""


def load_data_from_csv(config: CsvConfig) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Args:
        config: CsvConfig with file path, delimiter, and encoding settings.

    Returns:
        Pandas DataFrame with the CSV contents.

    Raises:
        CsvIngestionError: If the file is missing, unreadable, or malformed.
    """
    path = Path(config.file_path)

    if not path.exists():
        logger.error("CSV file not found: %s", path)
        raise CsvIngestionError(f"File not found: {path}")

    if not path.is_file():
        logger.error("Path is not a file: %s", path)
        raise CsvIngestionError(f"Not a file: {path}")

    logger.info("Loading CSV file: %s", path)

    try:
        result = pd.read_csv(
            path,
            delimiter=config.delimiter,
            encoding=config.encoding,
            chunksize=config.chunksize,
        )
        df = pd.concat(result, ignore_index=True) if config.chunksize else result

    except pd.errors.EmptyDataError as exc:
        logger.error("CSV file is empty: %s", path)
        raise CsvIngestionError(f"File is empty: {path}") from exc

    except pd.errors.ParserError as exc:
        logger.error("Failed to parse CSV file: %s", path)
        raise CsvIngestionError(f"Malformed CSV: {path}") from exc

    except UnicodeDecodeError as exc:
        logger.error("Encoding error reading CSV file: %s", path)
        raise CsvIngestionError(
            f"Failed to decode '{path}' using encoding '{config.encoding}'"
        ) from exc

    if df.empty:
        logger.warning("CSV loaded but contains zero rows: %s", path)
    else:
        logger.info("CSV ingestion successful. Shape=%s | File=%s", df.shape, path)

    return df


if __name__ == "__main__":
    config = CsvConfig(file_path="data/sample_transactions.csv")

    try:
        df = load_data_from_csv(config)
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(3))
    except CsvIngestionError as e:
        print(f"Ingestion failed: {e}")