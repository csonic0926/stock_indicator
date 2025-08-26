"""Functions for downloading historical stock market data.

The :func:`download_history` utility normalizes all column names in the
returned data frame to ``snake_case`` and adjusts prices and volume to account
for dividends and stock splits. The data retains both ``close`` and
``adj_close`` columns.
"""
# TODO: review

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas
import yfinance

LOGGER = logging.getLogger(__name__)


def download_history(
    symbol: str,
    start: str,
    end: str,
    cache_path: Path | None = None,
    **download_options: Any,
) -> pandas.DataFrame:
    """Download historical price data for a stock symbol.

    Parameters
    ----------
    symbol: str
        Stock ticker symbol to download.
    start: str
        Start date in ISO format (``YYYY-MM-DD``).
    end: str
        End date in ISO format (``YYYY-MM-DD``).
    cache_path: Path | None, optional
        Optional path to a CSV file used as a local cache. When the file exists,
        only missing rows are requested from the remote source and the merged
        result is written back to this file.
    **download_options
        Additional keyword arguments forwarded to :func:`yfinance.download`, such
        as ``actions``, ``auto_adjust``, or ``interval``.

    Returns
    -------
    pandas.DataFrame
        Data frame containing the historical data. Column names are normalized
        to ``snake_case`` and include both ``close`` and ``adj_close`` values.

    Raises
    ------
    ValueError
        If the provided symbol is not known.
    Exception
        Propagates the last error if downloading repeatedly fails.
    """
    from .symbols import load_symbols

    available_symbol_list = load_symbols()
    if available_symbol_list and symbol not in available_symbol_list:
        raise ValueError(f"Unknown symbol: {symbol}")

    cached_frame = pandas.DataFrame()
    if cache_path is not None and cache_path.exists():
        cached_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
        if not cached_frame.empty:
            next_download_date = cached_frame.index.max() + pandas.Timedelta(days=1)
            if next_download_date > pandas.Timestamp(end):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cached_frame.to_csv(cache_path)
                return cached_frame
            start = next_download_date.strftime("%Y-%m-%d")

    download_options = dict(download_options)
    download_options["auto_adjust"] = False

    maximum_attempts = 3
    for attempt_number in range(1, maximum_attempts + 1):
        try:
            downloaded_frame = yfinance.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                **download_options,
            )
            if isinstance(downloaded_frame.columns, pandas.MultiIndex):
                downloaded_frame.columns = downloaded_frame.columns.get_level_values(0)

            adjustment_ratio = (
                downloaded_frame["Adj Close"] / downloaded_frame["Close"]
            )
            for price_column_name in ["Open", "High", "Low", "Close"]:
                downloaded_frame[price_column_name] = (
                    downloaded_frame[price_column_name] * adjustment_ratio
                )
            downloaded_frame["Volume"] = (
                downloaded_frame["Volume"] / adjustment_ratio
            )
            downloaded_frame.columns = [
                str(column_name).lower().replace(" ", "_")
                for column_name in downloaded_frame.columns
            ]

            if not cached_frame.empty:
                downloaded_frame = pandas.concat([cached_frame, downloaded_frame])
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_frame.to_csv(cache_path)
            return downloaded_frame
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning(
                "Attempt %d to download data for %s failed: %s",
                attempt_number,
                symbol,
                download_error,
            )
            if attempt_number == maximum_attempts:
                LOGGER.error(
                    "Failed to download data for %s after %d attempts",
                    symbol,
                    maximum_attempts,
                )
                raise
            time.sleep(1)
