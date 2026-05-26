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


def _normalize_columns(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Return ``frame`` with flattened, snake_case column names."""
    # TODO: review
    if isinstance(frame.columns, pandas.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [
        str(column_name).lower().replace(" ", "_")
        for column_name in frame.columns
    ]
    return frame


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
        as ``actions`` or ``interval``. By default, ``auto_adjust`` is set to
        ``True`` to avoid warnings when retrieving data.

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
    from .symbols import load_symbols, SP500_SYMBOL

    available_symbol_list = load_symbols()
    if available_symbol_list and symbol not in available_symbol_list and symbol != SP500_SYMBOL:
        LOGGER.warning(
            "Symbol %s is not in the local cache; attempting download from Yahoo Finance anyway",
            symbol,
        )

    cached_frame = pandas.DataFrame()
    if cache_path is not None and cache_path.exists():
        cached_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)

    if "auto_adjust" not in download_options:
        download_options["auto_adjust"] = True

    if not cached_frame.empty:
        earliest_cached_date = cached_frame.index.min()
        requested_start_timestamp = pandas.Timestamp(start)
        # TODO: review
        if requested_start_timestamp < earliest_cached_date:
            try:
                earlier_frame = yfinance.download(
                    symbol,
                    start=start,
                    end=earliest_cached_date.strftime("%Y-%m-%d"),
                    progress=False,
                    **download_options,
                )
                earlier_frame = _normalize_columns(earlier_frame)
                cached_frame = pandas.concat([earlier_frame, cached_frame]).sort_index()
            except Exception as download_error:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to download missing history for %s: %s",
                    symbol,
                    download_error,
                )
        next_download_date = cached_frame.index.max() + pandas.Timedelta(days=1)
        # Yahoo Finance treats the end date as exclusive.  When the next
        # missing date is exactly equal to the requested end date, the cache is
        # already complete for the requested half-open date range.
        if next_download_date >= pandas.Timestamp(end):
            if cache_path is not None:
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
                **download_options,
            )
            downloaded_frame = _normalize_columns(downloaded_frame)
            if not cached_frame.empty:
                downloaded_frame = (
                    pandas.concat([cached_frame, downloaded_frame]).sort_index()
                )
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


def load_local_history(
    symbol: str,
    start: str,
    end: str,
    cache_path: Path | None = None,
    **_: Any,
) -> pandas.DataFrame:
    """Load historical price data strictly from a local CSV.

    This helper mirrors the return shape of :func:`download_history` but never
    performs any network requests. When the CSV is missing, corrupt, or empty,
    an empty data frame is returned. Column names are normalized to
    ``snake_case`` to match the downloader.

    Parameters
    ----------
    symbol: str
        Stock ticker symbol (used only for logging).
    start: str
        Inclusive start date (``YYYY-MM-DD``) for the slice returned.
    end: str
        Exclusive end date (``YYYY-MM-DD``) for the slice returned.
    cache_path: Path | None
        Path to the local CSV file. If ``None``, an empty frame is returned.

    Returns
    -------
    pandas.DataFrame
        Price history contained in the local CSV, sliced to ``[start, end)``
        and with normalized column names. Empty if not available.
    """
    if cache_path is None or not cache_path.exists():
        LOGGER.warning("Local CSV not found for %s: %s", symbol, cache_path)
        return pandas.DataFrame()
    try:
        frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
    except Exception as read_error:  # noqa: BLE001
        LOGGER.warning("Failed to read local CSV for %s: %s", symbol, read_error)
        return pandas.DataFrame()
    if frame.empty:
        return frame

    # Normalize columns to snake_case to match downloader
    if isinstance(frame.columns, pandas.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(name).lower().replace(" ", "_") for name in frame.columns]

    try:
        # Slice to [start, end) to mirror yfinance behavior
        start_ts = pandas.Timestamp(start)
        end_ts = pandas.Timestamp(end)
        sliced = frame.loc[(frame.index >= start_ts) & (frame.index < end_ts)]
        # If slicing drops everything due to timezone mismatch or index dtype,
        # fall back to returning the full frame rather than raising.
        return sliced if not sliced.empty else frame
    except Exception:  # noqa: BLE001
        return frame
