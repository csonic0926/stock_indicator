"""Interfaces with the SEC for company and submission data."""

# TODO: review

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

from .config import (
    SEC_COMPANY_TICKERS_URL,
    SEC_SUBMISSIONS_URL_TEMPLATE,
    SEC_USER_AGENT,
    SUBMISSIONS_DIRECTORY,
)
from .utils import (
    ensure_directory_exists,
    save_json_file,
    load_json_file,
    sleep_politely,
    normalize_ticker_symbol,
)

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": SEC_USER_AGENT}


def _format_central_index_key(central_index_key: int) -> str:
    """Return the central index key padded to ten digits."""
    return f"{int(central_index_key):010d}"


def _request_json(url: str) -> Dict[str, Any]:
    """Fetch JSON data from ``url`` with basic retries."""
    for attempt in range(3):
        try:
            logger.info("Requesting %s", url)
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            logger.error(
                "Request to %s failed on attempt %s: %s",
                url,
                attempt + 1,
                error,
            )
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return {}


def fetch_company_ticker_table() -> pd.DataFrame:
    """Retrieve a table mapping ticker symbols to central index keys."""
    data = _request_json(SEC_COMPANY_TICKERS_URL)
    rows = []
    for company_info in data.values():
        ticker_normalized = normalize_ticker_symbol(company_info["ticker"])
        rows.append(
            {
                "ticker": ticker_normalized,
                "cik": int(company_info["cik_str"]),
                "title": company_info.get("title"),
            }
        )
    return pd.DataFrame(rows)


def _submissions_cache_path(central_index_key: int) -> Path:
    """Return the cache path for a submission JSON file."""
    file_name = f"CIK{_format_central_index_key(central_index_key)}.json"
    return SUBMISSIONS_DIRECTORY / file_name


def fetch_submissions_json(central_index_key: int, use_cache: bool = True) -> Dict[str, Any] | None:
    """Fetch SEC submission data for ``central_index_key`` using a local cache."""
    ensure_directory_exists(SUBMISSIONS_DIRECTORY)
    cache_path = _submissions_cache_path(central_index_key)
    if use_cache and cache_path.exists():
        try:
            return load_json_file(cache_path)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Failed to load cache %s: %s", cache_path, error)
    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(
        cik_padded=_format_central_index_key(central_index_key)
    )
    data = _request_json(url)
    save_json_file(data, cache_path)
    sleep_politely()
    return data


def extract_standard_industrial_classification(submission_data: Dict[str, Any]) -> int | None:
    """Extract the standard industrial classification code from a submission payload."""
    try:
        sic_value = submission_data.get("sic")
        return int(sic_value) if sic_value is not None else None
    except (TypeError, ValueError):
        return None


def map_tickers_to_central_index_and_classification(
    universe_data_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Attach central index keys and SIC codes to ticker symbols."""
    sec_mapping_data_frame = fetch_company_ticker_table()
    universe_copy = universe_data_frame.copy()
    universe_copy["ticker_normalized"] = universe_copy["ticker"].map(
        normalize_ticker_symbol
    )
    sec_mapping_data_frame["ticker_normalized"] = sec_mapping_data_frame["ticker"]
    merged_data_frame = universe_copy.merge(
        sec_mapping_data_frame[["ticker_normalized", "cik"]],
        on="ticker_normalized",
        how="left",
    ).drop(columns=["ticker_normalized"])
    central_index_key_values = sorted(
        value for value in merged_data_frame["cik"].dropna().unique()
    )
    classification_rows = []
    for central_index_key_value in central_index_key_values:
        submissions_json = fetch_submissions_json(int(central_index_key_value), use_cache=True)
        classification_code = None
        if submissions_json:
            classification_code = extract_standard_industrial_classification(submissions_json)
        classification_rows.append(
            {
                "cik": int(central_index_key_value),
                "sic": classification_code,
            }
        )
    cik_sic_data_frame = pd.DataFrame(classification_rows)
    return merged_data_frame.merge(cik_sic_data_frame, on="cik", how="left")
