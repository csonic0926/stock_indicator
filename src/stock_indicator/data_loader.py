"""Functions for downloading historical stock market data.

The :func:`download_history` utility normalizes all column names in the
returned data frame to ``snake_case``. Starting with ``yfinance`` version
``0.2.51``, the ``download`` function returns a ``close`` column that already
reflects any dividends or stock splits, so no separate adjusted closing price
is provided.
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
    'auto_adjust':
        auto_adjust is set to true to avoid warning.
    **download_options
        Additional keyword arguments forwarded to :func:`yfinance.download`, such
        as ``actions``, or ``interval``.

    Returns
    -------
    pandas.DataFrame
        Data frame containing the historical data.

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

    maximum_attempts = 3
    for attempt_number in range(1, maximum_attempts + 1):
        try:
            downloaded_frame = yfinance.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True
                **download_options,
            )
            if isinstance(downloaded_frame.columns, pandas.MultiIndex):
                downloaded_frame.columns = downloaded_frame.columns.get_level_values(0)
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
