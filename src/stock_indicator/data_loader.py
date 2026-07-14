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

PRICE_SCALE_RATIO_LOWER_BOUND = 0.5
PRICE_SCALE_RATIO_UPPER_BOUND = 2.0
VOLUME_INVERSE_RATIO_LOWER_BOUND = 0.5
VOLUME_INVERSE_RATIO_UPPER_BOUND = 2.0


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


def _safe_ratio(numerator_value: Any, denominator_value: Any) -> float | None:
    """Return a positive finite ratio, or ``None`` when inputs are unusable."""
    try:
        numerator_float = float(numerator_value)
        denominator_float = float(denominator_value)
    except (TypeError, ValueError):
        return None
    if (
        not pandas.notna(numerator_float)
        or not pandas.notna(denominator_float)
        or numerator_float <= 0
        or denominator_float <= 0
    ):
        return None
    return numerator_float / denominator_float


def _is_large_price_scale_ratio(price_ratio: float | None) -> bool:
    """Return whether a price ratio is too large for a safe partial merge."""
    if price_ratio is None:
        return False
    return (
        price_ratio <= PRICE_SCALE_RATIO_LOWER_BOUND
        or price_ratio >= PRICE_SCALE_RATIO_UPPER_BOUND
    )


def _has_incompatible_overlap_scale(
    cached_frame: pandas.DataFrame,
    downloaded_frame: pandas.DataFrame,
) -> bool:
    """Return whether overlapping rows use incompatible price scales.

    Partial Yahoo refreshes are unsafe when a split has changed the adjusted
    price basis: newly downloaded rows can be split-adjusted while older cached
    rows remain on the pre-split basis. Same-date close ratios expose that case
    before mixed-scale rows are written back to disk.
    """
    required_columns = {"close", "volume"}
    if (
        not required_columns.issubset(cached_frame.columns)
        or not required_columns.issubset(downloaded_frame.columns)
    ):
        return False
    common_index = cached_frame.index.intersection(downloaded_frame.index)
    if common_index.empty:
        return False
    price_ratios: list[float] = []
    volume_ratios: list[float] = []
    for common_timestamp in common_index:
        price_ratio = _safe_ratio(
            downloaded_frame.at[common_timestamp, "close"],
            cached_frame.at[common_timestamp, "close"],
        )
        volume_ratio = _safe_ratio(
            downloaded_frame.at[common_timestamp, "volume"],
            cached_frame.at[common_timestamp, "volume"],
        )
        if price_ratio is not None and volume_ratio is not None:
            price_ratios.append(price_ratio)
            volume_ratios.append(volume_ratio)
    if not price_ratios:
        return False
    median_price_ratio = float(pandas.Series(price_ratios).median())
    median_volume_ratio = float(pandas.Series(volume_ratios).median())
    price_volume_product = median_price_ratio * median_volume_ratio
    return _is_large_price_scale_ratio(median_price_ratio) and (
        VOLUME_INVERSE_RATIO_LOWER_BOUND
        <= price_volume_product
        <= VOLUME_INVERSE_RATIO_UPPER_BOUND
    )


def _has_incompatible_adjacent_scale(
    cached_frame: pandas.DataFrame,
    downloaded_frame: pandas.DataFrame,
) -> bool:
    """Return whether the cache/download boundary looks like a split splice."""
    required_columns = {"close", "volume"}
    if (
        not required_columns.issubset(cached_frame.columns)
        or not required_columns.issubset(downloaded_frame.columns)
        or cached_frame.empty
        or downloaded_frame.empty
    ):
        return False
    last_cached_row = cached_frame.sort_index().iloc[-1]
    first_downloaded_row = downloaded_frame.sort_index().iloc[0]
    price_ratio = _safe_ratio(
        first_downloaded_row["close"],
        last_cached_row["close"],
    )
    if not _is_large_price_scale_ratio(price_ratio):
        return False
    volume_ratio = _safe_ratio(
        first_downloaded_row["volume"],
        last_cached_row["volume"],
    )
    if volume_ratio is None or price_ratio is None:
        return False
    price_volume_product = price_ratio * volume_ratio
    return (
        VOLUME_INVERSE_RATIO_LOWER_BOUND
        <= price_volume_product
        <= VOLUME_INVERSE_RATIO_UPPER_BOUND
    )


def _download_and_normalize_history(
    symbol: str,
    start: str,
    end: str,
    download_options: dict[str, Any],
) -> pandas.DataFrame:
    """Download one Yahoo history range and normalize the column names."""
    downloaded_frame = yfinance.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        **download_options,
    )
    return _normalize_columns(downloaded_frame)


def _save_cache_frame(frame: pandas.DataFrame, cache_path: Path | None) -> None:
    """Write ``frame`` to ``cache_path`` when caching is enabled."""
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path)


def download_history(
    symbol: str,
    start: str,
    end: str,
    cache_path: Path | None = None,
    refresh_lookback_days: int | None = None,
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
    refresh_lookback_days: int | None, optional
        When a cache exists, force a re-download from ``end`` minus this many
        calendar days. Cached rows before that refresh window are preserved,
        while overlapping recent rows are replaced by Yahoo Finance data.
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

    requested_start = start
    available_symbol_list = load_symbols()
    if (
        available_symbol_list
        and symbol not in available_symbol_list
        and symbol != SP500_SYMBOL
    ):
        LOGGER.warning(
            "Symbol %s is not in the local cache; attempting download from "
            "Yahoo Finance anyway",
            symbol,
        )

    cached_frame = pandas.DataFrame()
    if cache_path is not None and cache_path.exists():
        cached_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
    original_cached_frame = cached_frame.copy()
    cached_frame_before_refresh: pandas.DataFrame | None = None
    cached_frame_before_download = pandas.DataFrame()

    if "auto_adjust" not in download_options:
        download_options["auto_adjust"] = True
    if refresh_lookback_days is not None and refresh_lookback_days < 0:
        raise ValueError("refresh_lookback_days must be >= 0")

    if not cached_frame.empty:
        earliest_cached_date = cached_frame.index.min()
        requested_start_timestamp = pandas.Timestamp(start)
        requested_end_timestamp = pandas.Timestamp(end)
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
                original_cached_frame = cached_frame.copy()
            except Exception as download_error:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to download missing history for %s: %s",
                    symbol,
                    download_error,
                )
        if refresh_lookback_days is not None:
            refresh_start_timestamp = max(
                requested_start_timestamp,
                requested_end_timestamp - pandas.Timedelta(
                    days=refresh_lookback_days
                ),
            )
            cached_frame_before_refresh = cached_frame.copy()
            cached_frame = cached_frame.loc[
                cached_frame.index < refresh_start_timestamp
            ]
            cached_frame_before_download = cached_frame.copy()
            start = refresh_start_timestamp.strftime("%Y-%m-%d")
        else:
            next_download_date = cached_frame.index.max() + pandas.Timedelta(days=1)
            # Yahoo Finance treats the end date as exclusive.  When the next
            # missing date is exactly equal to the requested end date, the cache is
            # already complete for the requested half-open date range.
            if next_download_date >= requested_end_timestamp:
                _save_cache_frame(cached_frame, cache_path)
                return cached_frame
            cached_frame_before_download = cached_frame.copy()
            start = next_download_date.strftime("%Y-%m-%d")

    maximum_attempts = 3
    for attempt_number in range(1, maximum_attempts + 1):
        try:
            downloaded_frame = _download_and_normalize_history(
                symbol,
                start,
                end,
                download_options,
            )
            if (
                downloaded_frame.empty
                and refresh_lookback_days is not None
                and cached_frame_before_refresh is not None
                and not cached_frame_before_refresh.empty
            ):
                LOGGER.warning(
                    "Yahoo returned no refreshed data for %s; preserving "
                    "existing cache",
                    symbol,
                )
                _save_cache_frame(cached_frame_before_refresh, cache_path)
                return cached_frame_before_refresh
            if (
                cached_frame_before_refresh is not None
                and not downloaded_frame.empty
                and _has_incompatible_overlap_scale(
                    cached_frame_before_refresh,
                    downloaded_frame,
                )
            ):
                LOGGER.warning(
                    "Detected incompatible adjusted price scale for %s; "
                    "redownloading full requested history from %s",
                    symbol,
                    requested_start,
                )
                downloaded_frame = _download_and_normalize_history(
                    symbol,
                    requested_start,
                    end,
                    download_options,
                )
                if downloaded_frame.empty and not original_cached_frame.empty:
                    LOGGER.warning(
                        "Yahoo returned no full refresh data for %s; "
                        "preserving existing cache",
                        symbol,
                    )
                    _save_cache_frame(original_cached_frame, cache_path)
                    return original_cached_frame
                cached_frame = pandas.DataFrame()
            elif _has_incompatible_adjacent_scale(
                cached_frame_before_download,
                downloaded_frame,
            ):
                LOGGER.warning(
                    "Detected split-like cache/download boundary for %s; "
                    "redownloading full requested history from %s",
                    symbol,
                    requested_start,
                )
                downloaded_frame = _download_and_normalize_history(
                    symbol,
                    requested_start,
                    end,
                    download_options,
                )
                if downloaded_frame.empty and not original_cached_frame.empty:
                    LOGGER.warning(
                        "Yahoo returned no full refresh data for %s; "
                        "preserving existing cache",
                        symbol,
                    )
                    _save_cache_frame(original_cached_frame, cache_path)
                    return original_cached_frame
                cached_frame = pandas.DataFrame()
            if not cached_frame.empty:
                downloaded_frame = (
                    pandas.concat([cached_frame, downloaded_frame]).sort_index()
                )
            downloaded_frame = downloaded_frame.loc[
                ~downloaded_frame.index.duplicated(keep="last")
            ]
            _save_cache_frame(downloaded_frame, cache_path)
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
