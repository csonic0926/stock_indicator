"""Helper functions for managing historical data and signals."""
# TODO: review

from __future__ import annotations

import csv
import datetime
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy
import pandas
import yfinance
from pandas.tseries.offsets import BDay

from .cron import parse_daily_task_arguments, run_daily_tasks
from .data_loader import download_history, load_local_history
from .production_symbol_status import load_symbols_allowed_for_price_refresh
from .symbols import SP500_SYMBOL, load_symbols
from . import strategy

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2019-01-01"
# Earliest date used when refreshing historical data; this guards against
# missing rows in local caches. Modify only when extending the supported
# history range.
MINIMUM_HISTORY_DATE = "2014-01-01"
# Maximum trailing window (in calendar days) of history needed to evaluate
# indicator windows safely when recomputing signals for a single date.
SIGNAL_HISTORY_LOOKBACK_DAYS = 756
YAHOO_CACHE_REFRESH_LOOKBACK_DAYS = 365
YAHOO_MISSING_DATE_RETRY_TIMEOUT_SECONDS = 20
DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"
PRODUCTION_SYMBOLS_PATH = DATA_DIRECTORY / "production_symbols.txt"
PRODUCTION_SYMBOL_STATUS_PATH = DATA_DIRECTORY / "production_symbol_status.csv"
BACKTEST_UNIVERSE_DIRECTORY = DATA_DIRECTORY / "backtest_universe_alpha_vantage"
ETF_SYMBOLS_PATH = (
    BACKTEST_UNIVERSE_DIRECTORY / "backtest_etf_symbols_2010_2026_plus_runtime.csv"
)
LISTING_STATUS_RAW_PATH = (
    BACKTEST_UNIVERSE_DIRECTORY / "listing_status_2010_2026_plus_runtime_raw.csv"
)
NON_COMMON_STOCK_NAME_PATTERN = re.compile(
    r"\b(warrants?|units?|rights?|preferred|preference)\b",
    flags=re.IGNORECASE,
)
NON_COMMON_STOCK_SYMBOL_PATTERN = re.compile(
    r"(?:[.-](?:WT|WS|W|U|R)|(?:WS|WT))$",
    flags=re.IGNORECASE,
)

CRON_RUNTIME_FIELD_NAMES = [
    "signal_date",
    "start_iso",
    "end_iso",
    "total_seconds",
    "update_seconds",
    "signal_seconds",
    "first_download_seconds",
    "retry_download_seconds",
    "total_download_seconds",
    "process_seconds",
]


@dataclass(frozen=True)
class YahooMissingDateRetryResult:
    """Summary for a post-refresh Yahoo missing-date retry pass."""

    target_date: str
    attempted_symbols: int
    recovered_symbols: tuple[str, ...]
    remaining_missing_symbols: tuple[str, ...]


def record_cron_runtime(
    csv_path: Path,
    *,
    signal_date: str,
    start_epoch: float,
    update_end_epoch: float,
    end_epoch: float,
    first_download_end_epoch: float | None = None,
) -> None:
    """Append phase-level wall-clock timing for the daily cron wrapper.

    The helper intentionally uses append mode and only writes the header when
    the target CSV does not exist. Older ledgers are upgraded in place when
    new timing columns are introduced so historical rows remain readable.
    """

    # TODO: review

    def _round_duration_seconds(duration_seconds: float) -> int:
        """Round a non-negative duration to the nearest whole second."""

        return int(math.floor(duration_seconds + 0.5))

    resolved_first_download_end_epoch = (
        first_download_end_epoch
        if first_download_end_epoch is not None
        else update_end_epoch
    )
    if not (
        start_epoch
        <= resolved_first_download_end_epoch
        <= update_end_epoch
        <= end_epoch
    ):
        raise ValueError("cron runtime timestamps must be chronological")

    csv_field_names = _prepare_cron_runtime_csv_schema(csv_path)
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    start_datetime = datetime.datetime.fromtimestamp(start_epoch).astimezone()
    end_datetime = datetime.datetime.fromtimestamp(end_epoch).astimezone()
    update_seconds = _round_duration_seconds(update_end_epoch - start_epoch)
    signal_seconds = _round_duration_seconds(end_epoch - update_end_epoch)
    total_seconds = _round_duration_seconds(end_epoch - start_epoch)
    first_download_seconds = _round_duration_seconds(
        resolved_first_download_end_epoch - start_epoch
    )
    retry_download_seconds = _round_duration_seconds(
        update_end_epoch - resolved_first_download_end_epoch
    )

    with csv_path.open("a", newline="", encoding="utf-8") as cron_runtime_file:
        writer = csv.DictWriter(
            cron_runtime_file,
            fieldnames=csv_field_names,
        )
        if not csv_exists:
            writer.writeheader()
        writer.writerow(
            {
                "signal_date": signal_date,
                "start_iso": start_datetime.isoformat(),
                "end_iso": end_datetime.isoformat(),
                "total_seconds": total_seconds,
                "update_seconds": update_seconds,
                "signal_seconds": signal_seconds,
                "first_download_seconds": first_download_seconds,
                "retry_download_seconds": retry_download_seconds,
                "total_download_seconds": update_seconds,
                "process_seconds": signal_seconds,
            }
        )


def _prepare_cron_runtime_csv_schema(csv_path: Path) -> list[str]:
    """Add newly required timing columns while preserving existing rows."""

    # TODO: review
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return list(CRON_RUNTIME_FIELD_NAMES)

    with csv_path.open("r", newline="", encoding="utf-8") as runtime_file:
        reader = csv.DictReader(runtime_file)
        existing_field_names = list(reader.fieldnames or [])
        existing_rows = list(reader)

    resolved_field_names = list(existing_field_names)
    for field_name in CRON_RUNTIME_FIELD_NAMES:
        if field_name not in resolved_field_names:
            resolved_field_names.append(field_name)
    if resolved_field_names == existing_field_names:
        return resolved_field_names

    temporary_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as runtime_file:
        writer = csv.DictWriter(runtime_file, fieldnames=resolved_field_names)
        writer.writeheader()
        writer.writerows(existing_rows)
    temporary_path.replace(csv_path)
    return resolved_field_names


def determine_latest_trading_date(
    now: datetime.datetime | None = None,
) -> datetime.date:
    """Return the most recent trading date based on Eastern Time.

    Parameters
    ----------
    now:
        Current timestamp. When ``None`` the system time in the ``US/Eastern``
        timezone is used.

    Returns
    -------
    datetime.date
        The prior business day if the time is earlier than 16:00 Eastern,
        otherwise the current date.
    """

    eastern_zone = ZoneInfo("US/Eastern")
    if now is None:
        current_time = datetime.datetime.now(tz=eastern_zone)
    else:
        current_time = (
            now.astimezone(eastern_zone)
            if now.tzinfo is not None
            else now.replace(tzinfo=eastern_zone)
        )

    if current_time.time() < datetime.time(16, 0):
        previous_business_day = (
            pandas.Timestamp(current_time.date()) - BDay(1)
        ).date()
        return previous_business_day
    return current_time.date()


def determine_start_date(data_directory: Path) -> str:
    """Return the earliest date across all CSV files in ``data_directory``.

    When no CSV files are available, ``DEFAULT_START_DATE`` is returned.
    """

    earliest_date: datetime.date | None = None
    if not data_directory.exists():
        return DEFAULT_START_DATE
    for csv_file_path in data_directory.glob("*.csv"):
        try:
            date_frame = pandas.read_csv(
                csv_file_path, usecols=[0], parse_dates=[0]
            )
        except Exception as read_error:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)
            continue
        if date_frame.empty:
            continue
        try:
            column_minimum = date_frame.iloc[:, 0].min()
        except TypeError:
            LOGGER.warning(
                "Skipping %s due to non-date values in the first column",
                csv_file_path,
            )
            continue
        if not hasattr(column_minimum, "date"):
            continue
        earliest_candidate = column_minimum.date()
        if earliest_date is None or earliest_candidate < earliest_date:
            earliest_date = earliest_candidate
    if earliest_date is None:
        return DEFAULT_START_DATE
    return earliest_date.isoformat()


def determine_last_cached_date(data_directory: Path) -> datetime.date:
    """Return the most recent date found in any CSV under ``data_directory``."""

    latest_date: datetime.date | None = None
    if data_directory.exists():
        for csv_file_path in data_directory.glob("*.csv"):
            try:
                date_frame = pandas.read_csv(
                    csv_file_path, usecols=[0], parse_dates=[0]
                )
            except Exception as read_error:  # noqa: BLE001
                LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)
                continue
            if date_frame.empty:
                continue
            value = date_frame.iloc[-1, 0]
            if hasattr(value, "date"):
                current_date = value.date()
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date
    if latest_date is None:
        return datetime.date.fromisoformat(DEFAULT_START_DATE)
    return latest_date


def determine_latest_cached_market_date(data_directory: Path) -> datetime.date:
    """Return the latest cached broad-market trading date.

    The cron wrapper may run on exchange holidays that are still ordinary
    business days. In that case, a few individual Yahoo symbols can contain a
    row for the holiday while the broad market did not trade. The S&P 500 cache
    is used as the market-date anchor so daily signal generation evaluates the
    last real market session.
    """

    market_cache_path = data_directory / f"{SP500_SYMBOL}.csv"
    if market_cache_path.exists():
        try:
            market_date_frame = pandas.read_csv(
                market_cache_path,
                usecols=[0],
                parse_dates=[0],
            )
        except Exception as read_error:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %s", market_cache_path, read_error)
        else:
            if not market_date_frame.empty:
                latest_market_timestamp = market_date_frame.iloc[:, 0].max()
                if hasattr(latest_market_timestamp, "date"):
                    return latest_market_timestamp.date()

    LOGGER.warning(
        "Market cache %s is unavailable; falling back to latest cached symbol date",
        market_cache_path,
    )
    return determine_last_cached_date(data_directory)


def _symbol_separator_aliases(symbol_name: str) -> set[str]:
    """Return dot/dash aliases used by local CSV and vendor symbols."""

    normalized_symbol = symbol_name.strip().upper()
    aliases = {normalized_symbol}
    if "." in normalized_symbol:
        aliases.add(normalized_symbol.replace(".", "-"))
    if "-" in normalized_symbol:
        aliases.add(normalized_symbol.replace("-", "."))
    return aliases


def load_symbols_rejected_by_asset_metadata() -> set[str]:
    """Return symbols identified as ETFs or funds by listing-status metadata."""

    if not ETF_SYMBOLS_PATH.exists():
        LOGGER.warning("ETF metadata not found: %s", ETF_SYMBOLS_PATH)
        return set()

    try:
        etf_frame = pandas.read_csv(ETF_SYMBOLS_PATH)
    except (OSError, pandas.errors.ParserError) as read_error:
        LOGGER.warning(
            "Could not read ETF metadata %s: %s",
            ETF_SYMBOLS_PATH,
            read_error,
        )
        return set()

    symbol_columns = [
        column_name
        for column_name in (
            "local_symbol_candidate",
            "yahoo_symbol_candidate",
            "alpha_vantage_symbol",
            "symbol",
        )
        if column_name in etf_frame.columns
    ]
    rejected_symbols: set[str] = set()
    for column_name in symbol_columns:
        for symbol_name in etf_frame[column_name].dropna().astype(str):
            rejected_symbols.update(_symbol_separator_aliases(symbol_name))
    return rejected_symbols

def load_symbols_rejected_by_listing_name() -> set[str]:
    """Return symbols whose listing names identify non-common-stock instruments."""

    if not LISTING_STATUS_RAW_PATH.exists():
        LOGGER.warning(
            "Listing-status raw metadata not found: %s", LISTING_STATUS_RAW_PATH
        )
        return set()

    try:
        listing_frame = pandas.read_csv(LISTING_STATUS_RAW_PATH, low_memory=False)
    except (OSError, pandas.errors.ParserError) as read_error:
        LOGGER.warning(
            "Could not read listing-status raw metadata %s: %s",
            LISTING_STATUS_RAW_PATH,
            read_error,
        )
        return set()

    if "symbol" not in listing_frame.columns or "name" not in listing_frame.columns:
        return set()

    named_listing_frame = listing_frame.dropna(subset=["symbol", "name"])
    rejected_symbols: set[str] = set()
    for listing_row in named_listing_frame.itertuples(index=False):
        listing_name = str(getattr(listing_row, "name", ""))
        if not NON_COMMON_STOCK_NAME_PATTERN.search(listing_name):
            continue
        symbol_name = str(getattr(listing_row, "symbol", ""))
        rejected_symbols.update(_symbol_separator_aliases(symbol_name))
    return rejected_symbols


def load_runtime_download_symbols() -> list[str]:
    """Return the symbol-cache universe for the daily Yahoo refresh.

    Runtime trading must not depend on every cached CSV under ``stock_data``.
    ``production_symbols.txt`` is the append-preserving source contract.
    Confirmed inactive instruments are skipped through
    ``production_symbol_status.csv``; ``price_unavailable`` symbols remain in
    the refresh set so a recovered Yahoo feed can reactivate them.  FF12
    coverage remains mandatory because production selection is group-aware.
    """

    current_symbols = [
        symbol_name.strip().upper()
        for symbol_name in load_symbols_allowed_for_price_refresh(
            PRODUCTION_SYMBOLS_PATH,
            PRODUCTION_SYMBOL_STATUS_PATH,
        )
        if symbol_name and symbol_name.strip() and symbol_name != SP500_SYMBOL
    ]
    current_symbols = sorted(dict.fromkeys(current_symbols))
    symbol_to_group_identifier = strategy.load_ff12_groups_by_symbol()

    if not symbol_to_group_identifier:
        LOGGER.warning(
            "FF12 sector map is unavailable; runtime refresh will trust "
            "production_symbols.txt without sector coverage checks"
        )

    runtime_symbols: list[str] = []
    skipped_missing_sector_count = 0
    for symbol_name in current_symbols:
        if (
            symbol_to_group_identifier
            and symbol_name not in symbol_to_group_identifier
        ):
            skipped_missing_sector_count += 1
            continue
        runtime_symbols.append(symbol_name)

    runtime_symbols.append(SP500_SYMBOL)
    LOGGER.info(
        "Runtime Yahoo refresh universe: %d symbols (%d production-status, "
        "%d missing-sector skipped)",
        len(runtime_symbols),
        len(current_symbols),
        skipped_missing_sector_count,
    )
    return runtime_symbols


def update_all_data_from_yf(
    start_date: str, end_date: str, data_directory: Path
) -> None:
    """Download historical data for the sector-safe runtime universe.

    The ``end_date`` argument is treated as inclusive. To accommodate the
    exclusive end-date semantics of the Yahoo Finance API, this function adds
    one day to ``end_date`` before requesting data. Per-symbol failures are
    logged and skipped so one bad Yahoo response cannot stop the cron job.
    """

    exclusive_end_date = (
        datetime.date.fromisoformat(end_date) + datetime.timedelta(days=1)
    ).isoformat()
    for symbol_name in load_runtime_download_symbols():
        csv_path = data_directory / f"{symbol_name}.csv"
        try:
            download_history(
                symbol_name,
                start=start_date,
                end=exclusive_end_date,
                cache_path=csv_path,
                refresh_lookback_days=YAHOO_CACHE_REFRESH_LOOKBACK_DAYS,
            )
            try:
                cached_frame = pandas.read_csv(
                    csv_path, index_col=0, parse_dates=True
                )
                deduplicated_frame = cached_frame.loc[
                    ~cached_frame.index.duplicated(keep="last")
                ]
                deduplicated_frame.to_csv(csv_path)
            except Exception as cache_error:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to deduplicate %s: %s", csv_path, cache_error
                )
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning(
                "Failed to refresh data for %s: %s", symbol_name, download_error
            )


def _normalize_cache_datetime_index(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Return ``frame`` with a timezone-naive ``DatetimeIndex``."""

    normalized_frame = frame.copy()
    normalized_index = pandas.to_datetime(normalized_frame.index)
    if getattr(normalized_index, "tz", None) is not None:
        normalized_index = normalized_index.tz_localize(None)
    normalized_frame.index = normalized_index
    return normalized_frame


def _cache_has_target_date(
    cache_path: Path,
    target_timestamp: pandas.Timestamp,
) -> bool:
    """Return whether a cache file contains the requested trading date."""

    if not cache_path.exists():
        return False
    try:
        cached_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
    except (OSError, ValueError, pandas.errors.ParserError) as read_error:
        LOGGER.warning("Failed to inspect cache %s: %s", cache_path, read_error)
        return False
    if cached_frame.empty:
        return False
    cached_frame = _normalize_cache_datetime_index(cached_frame)
    return bool((cached_frame.index.normalize() == target_timestamp).any())


def _find_symbols_missing_cache_date(
    symbol_names: list[str],
    data_directory: Path,
    target_timestamp: pandas.Timestamp,
) -> list[str]:
    """Return symbols whose cache lacks ``target_timestamp``."""

    missing_symbol_names: list[str] = []
    for symbol_name in symbol_names:
        cache_path = data_directory / f"{symbol_name}.csv"
        if not _cache_has_target_date(cache_path, target_timestamp):
            missing_symbol_names.append(symbol_name)
    return missing_symbol_names


def _normalize_retry_download_columns(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Return ``frame`` with cache-compatible lower snake case columns."""

    normalized_frame = frame.copy()
    normalized_frame.columns = [
        str(column_name).lower().replace(" ", "_")
        for column_name in normalized_frame.columns
    ]
    return normalized_frame


def _extract_target_date_rows_from_symbol_download(
    downloaded_frame: pandas.DataFrame,
    target_timestamp: pandas.Timestamp,
) -> pandas.DataFrame:
    """Extract target-date rows from one symbol's isolated Yahoo response."""

    # TODO: review
    if downloaded_frame.empty:
        return pandas.DataFrame()

    symbol_frame = downloaded_frame.dropna(how="all")
    if symbol_frame.empty:
        return pandas.DataFrame()

    symbol_frame = _normalize_retry_download_columns(symbol_frame)
    symbol_frame = _normalize_cache_datetime_index(symbol_frame)
    return symbol_frame.loc[symbol_frame.index.normalize() == target_timestamp]


def _merge_retry_rows_into_cache(
    cache_path: Path,
    target_rows: pandas.DataFrame,
) -> bool:
    """Merge target-date retry rows into ``cache_path``."""

    if target_rows.empty:
        return False

    if cache_path.exists():
        try:
            cached_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
        except (OSError, ValueError, pandas.errors.ParserError) as read_error:
            LOGGER.warning(
                "Failed to read cache %s before retry merge: %s",
                cache_path,
                read_error,
            )
            cached_frame = pandas.DataFrame()
    else:
        cached_frame = pandas.DataFrame()

    if not cached_frame.empty:
        cached_frame = _normalize_cache_datetime_index(cached_frame)

    merged_frame = pandas.concat([cached_frame, target_rows]).sort_index()
    merged_frame = merged_frame.loc[~merged_frame.index.duplicated(keep="last")]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged_frame.to_csv(cache_path)
    return True


def retry_missing_date_from_yf(
    target_date: str,
    data_directory: Path,
    *,
    symbol_names: list[str] | None = None,
) -> YahooMissingDateRetryResult:
    """Run a second per-symbol pass for caches missing ``target_date``.

    The main cron refresh completes its first pass over the full universe
    before this function runs. This pass audits the resulting caches rather
    than relying on exceptions because ``yfinance.download`` can report an
    individual failure by returning an empty frame. Each missing symbol then
    receives one isolated one-day request so another symbol cannot affect its
    retry result.
    """

    # TODO: review

    target_timestamp = pandas.Timestamp(datetime.date.fromisoformat(target_date))
    exclusive_end_date = (
        datetime.date.fromisoformat(target_date) + datetime.timedelta(days=1)
    ).isoformat()
    runtime_symbol_names = (
        list(symbol_names)
        if symbol_names is not None
        else load_runtime_download_symbols()
    )
    missing_symbol_names = _find_symbols_missing_cache_date(
        runtime_symbol_names,
        data_directory,
        target_timestamp,
    )

    if not missing_symbol_names:
        LOGGER.info(
            "Yahoo missing-date retry skipped for %s: no missing symbols",
            target_date,
        )
        return YahooMissingDateRetryResult(
            target_date=target_date,
            attempted_symbols=0,
            recovered_symbols=(),
            remaining_missing_symbols=(),
        )

    LOGGER.info(
        "Retrying Yahoo missing-date rows for %s: %d symbols",
        target_date,
        len(missing_symbol_names),
    )

    recovered_symbol_names: list[str] = []
    for symbol_name in missing_symbol_names:
        try:
            downloaded_frame = yfinance.Ticker(symbol_name).history(
                start=target_date,
                end=exclusive_end_date,
                interval="1d",
                auto_adjust=True,
                actions=False,
                timeout=YAHOO_MISSING_DATE_RETRY_TIMEOUT_SECONDS,
                raise_errors=True,
            )
        except Exception as download_error:  # noqa: BLE001
            LOGGER.warning(
                "Yahoo missing-date retry failed for %s on %s: %s",
                symbol_name,
                target_date,
                download_error,
            )
            continue

        target_rows = _extract_target_date_rows_from_symbol_download(
            downloaded_frame,
            target_timestamp,
        )
        cache_path = data_directory / f"{symbol_name}.csv"
        if _merge_retry_rows_into_cache(cache_path, target_rows):
            recovered_symbol_names.append(symbol_name)

    remaining_missing_symbol_names = _find_symbols_missing_cache_date(
        runtime_symbol_names,
        data_directory,
        target_timestamp,
    )
    if recovered_symbol_names:
        LOGGER.info(
            "Yahoo missing-date retry recovered %d/%d symbols for %s",
            len(recovered_symbol_names),
            len(missing_symbol_names),
            target_date,
        )
    if remaining_missing_symbol_names:
        LOGGER.warning(
            "Yahoo missing-date retry left %d symbols without %s; "
            "first symbols: %s",
            len(remaining_missing_symbol_names),
            target_date,
            remaining_missing_symbol_names[:20],
        )

    return YahooMissingDateRetryResult(
        target_date=target_date,
        attempted_symbols=len(missing_symbol_names),
        recovered_symbols=tuple(recovered_symbol_names),
        remaining_missing_symbols=tuple(remaining_missing_symbol_names),
    )


def _load_current_symbols_with_cached_price(
    evaluation_timestamp: pandas.Timestamp,
) -> List[str]:
    """Return current symbols that have a cached row for the evaluation date.

    The live cron path must not treat old CSV files as the tradable universe.
    The authoritative runtime universe is ``symbols.txt``; local CSVs only
    prove that the current symbol has price data for the requested date.
    """

    current_symbol_list = [
        symbol_name
        for symbol_name in load_runtime_download_symbols()
        if symbol_name and symbol_name != SP500_SYMBOL
    ]
    cached_symbol_set = {
        csv_file_path.stem
        for csv_file_path in STOCK_DATA_DIRECTORY.glob("*.csv")
        if csv_file_path.stem and csv_file_path.stem != SP500_SYMBOL
    }
    stale_cached_symbols = sorted(cached_symbol_set - set(current_symbol_list))
    if stale_cached_symbols:
        LOGGER.info(
            "Ignoring %d cached symbols that are not in the current symbol list",
            len(stale_cached_symbols),
        )

    symbols_with_price: List[str] = []
    missing_symbols: List[str] = []
    for symbol_name in current_symbol_list:
        csv_file_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
        if not csv_file_path.exists():
            missing_symbols.append(symbol_name)
            continue
        try:
            history_frame = pandas.read_csv(
                csv_file_path, index_col=0, parse_dates=True
            )
        except Exception:  # noqa: BLE001
            missing_symbols.append(symbol_name)
            continue
        if evaluation_timestamp in history_frame.index:
            symbols_with_price.append(symbol_name)
        else:
            missing_symbols.append(symbol_name)

    if missing_symbols:
        LOGGER.debug(
            "Skipping %d current symbols missing %s",
            len(missing_symbols),
            evaluation_timestamp.date().isoformat(),
        )
    return symbols_with_price


def find_history_signal(
    date_string: str | None,
    dollar_volume_filter: str,
    buy_strategy: str,
    sell_strategy: str,
    stop_loss: float,
    allowed_fama_french_groups: set[int] | None = None,
    near_delta_range: tuple[float, float] | None = None,
    price_tightness_range: tuple[float, float] | None = None,
) -> Dict[str, List[str]]:
    """Find entry and exit signals for a single historical date.

    When ``date_string`` is ``None`` the most recent trading date is determined
    via :func:`determine_latest_trading_date` and used for evaluation. Entries
    based on generated signals occur on the next trading day's open.

    Parameters
    ----------
    date_string:
        ISO formatted date string representing the signal date. ``None``
        triggers evaluation for the latest trading day.
    dollar_volume_filter:
        Filter applied to select symbols based on dollar volume.
    buy_strategy:
        Name of the strategy used to generate entry signals.
    sell_strategy:
        Name of the strategy used to generate exit signals.
    stop_loss:
        Fractional loss used for downstream simulations; not used in signal
        detection here but preserved for parity with other entry points.
    allowed_fama_french_groups:
        Optional set of FF12 group identifiers (1–12) used to restrict the
        tradable universe. Group 12 (Other) is selectable for group-aware
        filtering.

    Historical data starting from the later of
    ``SIGNAL_HISTORY_LOOKBACK_DAYS`` before the evaluation date or
    ``MINIMUM_HISTORY_DATE`` is used. When the cached history begins after that
    point, the available start date is used instead. This bounds the amount of
    data loaded for each symbol while maintaining enough history for indicator
    calculations.

    Returns
    -------
    Dict[str, List[str] | List[tuple[str, int | None]]]
        Dictionary containing ``filtered_symbols`` (pairs of symbol and
        Fama–French group identifiers), ``entry_signals`` and ``exit_signals``.
    """

    # TODO: review
    if date_string is None:
        date_string = determine_latest_trading_date().isoformat()
    group_token = (
        ""
        if not allowed_fama_french_groups
        else "group=" + ",".join(str(i) for i in sorted(allowed_fama_french_groups)) + " "
    )
    argument_line = f"{group_token}{dollar_volume_filter} {buy_strategy} {sell_strategy} {stop_loss}"
    try:
        evaluation_timestamp = pandas.Timestamp(date_string)
        evaluation_end_date_string = evaluation_timestamp.date().isoformat()
    except Exception:  # noqa: BLE001
        evaluation_timestamp = pandas.Timestamp.today()
        evaluation_end_date_string = evaluation_timestamp.date().isoformat()
    cached_start_timestamp = pandas.Timestamp(
        determine_start_date(STOCK_DATA_DIRECTORY)
    )
    minimum_timestamp = pandas.Timestamp(MINIMUM_HISTORY_DATE)
    requested_start_timestamp = max(
        minimum_timestamp,
        evaluation_timestamp - pandas.Timedelta(days=SIGNAL_HISTORY_LOOKBACK_DAYS),
    )
    start_timestamp = max(cached_start_timestamp, requested_start_timestamp)
    start_date_string = start_timestamp.date().isoformat()
    local_symbols = _load_current_symbols_with_cached_price(evaluation_timestamp)
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        parsed_buy_strategy,
        parsed_sell_strategy,
        _,
        allowed_groups,
    ) = parse_daily_task_arguments(argument_line)
    signal_result: Dict[str, List[str] | List[tuple[str, int | None]]] = run_daily_tasks(
        buy_strategy_name=parsed_buy_strategy,
        sell_strategy_name=parsed_sell_strategy,
        start_date=start_date_string,
        end_date=evaluation_end_date_string,
        symbol_list=local_symbols,
        data_download_function=load_local_history,
        data_directory=STOCK_DATA_DIRECTORY,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        top_dollar_volume_rank=top_dollar_volume_rank,
        allowed_fama_french_groups=allowed_groups,
        maximum_symbols_per_group=maximum_symbols_per_group,
        use_unshifted_signals=True,
        near_delta_range=near_delta_range,
        price_tightness_range=price_tightness_range,
    )
    entry_signals = signal_result.get("entry_signals", [])
    exit_signals = signal_result.get("exit_signals", [])
    filtered_symbols = signal_result.get("filtered_symbols", [])
    return {
        "filtered_symbols": filtered_symbols,
        "entry_signals": entry_signals,
        "exit_signals": exit_signals,
    }


def filter_debug_values(
    symbol_name: str,
    evaluation_date_string: str,
    buy_strategy_name: str,
    sell_strategy_name: str,
    exit_alpha_factor: float | None = None,
) -> Dict[str, float | bool | None]:
    """Return indicator debug values for ``symbol_name`` on ``evaluation_date_string``.

    Loads local price history, attaches indicators for the provided buy and sell
    strategies on separate data copies, and extracts a handful of useful columns
    for debugging threshold-based filters. Computing indicators on individual
    copies prevents sell-side indicators from overwriting buy-side results when
    both strategies share the same base name.
    """

    # TODO: review
    csv_file_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
    if not csv_file_path.exists():
        LOGGER.warning("Local CSV not found for %s: %s", symbol_name, csv_file_path)
        return {
            "sma_angle": None,
            "sma_angle_previous": None,
            "near_price_volume_ratio": None,
            "near_price_volume_ratio_previous": None,
            "above_price_volume_ratio": None,
            "above_price_volume_ratio_previous": None,
            "near_delta": None,
            "slope_60": None,
            "entry": False,
            "exit": False,
        }

    price_history_frame = strategy.load_price_data(csv_file_path)
    if price_history_frame.empty:
        return {
            "sma_angle": None,
            "sma_angle_previous": None,
            "near_price_volume_ratio": None,
            "near_price_volume_ratio_previous": None,
            "above_price_volume_ratio": None,
            "above_price_volume_ratio_previous": None,
            "near_delta": None,
            "slope_60": None,
            "entry": False,
            "exit": False,
        }

    evaluation_timestamp = pandas.Timestamp(evaluation_date_string)
    if evaluation_timestamp in price_history_frame.index:
        selected_timestamp = evaluation_timestamp
    else:
        candidate_index = price_history_frame.index[
            price_history_frame.index <= evaluation_timestamp
        ]
        if len(candidate_index) == 0:
            return {
                "sma_angle": None,
                "sma_angle_previous": None,
                "near_price_volume_ratio": None,
                "near_price_volume_ratio_previous": None,
                "above_price_volume_ratio": None,
                "above_price_volume_ratio_previous": None,
                "near_delta": None,
                "slope_60": None,
                "entry": False,
                "exit": False,
            }
        selected_timestamp = candidate_index[-1]

    selected_position_candidates = numpy.where(
        price_history_frame.index == selected_timestamp
    )[0]
    if selected_position_candidates.size == 0:
        return {
            "sma_angle": None,
            "sma_angle_previous": None,
            "near_price_volume_ratio": None,
            "near_price_volume_ratio_previous": None,
            "above_price_volume_ratio": None,
            "above_price_volume_ratio_previous": None,
            "near_delta": None,
            "slope_60": None,
            "entry": False,
            "exit": False,
        }
    selected_position = int(selected_position_candidates[-1])
    last_included_position = min(
        selected_position + 1, len(price_history_frame.index) - 1
    )

    price_history_frame = price_history_frame.iloc[
        : last_included_position + 1
    ].copy()
    buy_price_history_frame = price_history_frame.copy()
    sell_price_history_frame = price_history_frame.copy()

    (
        buy_base_name,
        buy_window_size,
        buy_angle_range,
        buy_near_range,
        buy_above_range,
    ) = strategy.parse_strategy_name(buy_strategy_name)
    buy_function = strategy.BUY_STRATEGIES.get(buy_base_name)
    if buy_function is not None:
        buy_arguments: Dict[str, Any] = {"include_raw_signals": True}
        if buy_window_size is not None:
            buy_arguments["window_size"] = buy_window_size
        if buy_angle_range is not None:
            buy_arguments["angle_range"] = buy_angle_range
        if buy_near_range is not None:
            buy_arguments["near_range"] = buy_near_range
        if buy_above_range is not None:
            buy_arguments["above_range"] = buy_above_range
        buy_function(buy_price_history_frame, **buy_arguments)
        strategy.rename_signal_columns(
            buy_price_history_frame, buy_base_name, buy_strategy_name
        )
        if (
            "sma_angle" in buy_price_history_frame.columns
            and "sma_angle_previous" not in buy_price_history_frame.columns
        ):
            buy_price_history_frame["sma_angle_previous"] = buy_price_history_frame[
                "sma_angle"
            ].shift(1)

    (
        sell_base_name,
        sell_window_size,
        sell_angle_range,
        sell_near_range,
        sell_above_range,
    ) = strategy.parse_strategy_name(sell_strategy_name)
    sell_function = strategy.SELL_STRATEGIES.get(sell_base_name)
    if sell_function is not None:
        sell_arguments: Dict[str, Any] = {"include_raw_signals": True}
        if sell_window_size is not None:
            sell_arguments["window_size"] = sell_window_size
        if sell_angle_range is not None:
            sell_arguments["angle_range"] = sell_angle_range
        if sell_near_range is not None:
            sell_arguments["near_range"] = sell_near_range
        if sell_above_range is not None:
            sell_arguments["above_range"] = sell_above_range
        if (
            sell_base_name == "ema_sma_cross_testing"
            and exit_alpha_factor is not None
        ):
            sell_arguments["exit_alpha_factor"] = exit_alpha_factor
        sell_function(sell_price_history_frame, **sell_arguments)
        strategy.rename_signal_columns(
            sell_price_history_frame, sell_base_name, sell_strategy_name
        )
        if (
            "sma_angle" in sell_price_history_frame.columns
            and "sma_angle_previous" not in sell_price_history_frame.columns
        ):
            sell_price_history_frame["sma_angle_previous"] = (
                sell_price_history_frame["sma_angle"].shift(1)
            )

    # slope_60 = (close[T] - close[T-59]) / close[T-59]; mirrors the
    # simulator's per-trade enrichment in strategy.py:4414-4418 so the
    # production today-slice can apply slope_max / slope_min /
    # free_fall / slope_dead_zone filters and tp_slope_amplify.
    if "close" in buy_price_history_frame.columns:
        close_60_bars_ago = buy_price_history_frame["close"].shift(59)
        buy_price_history_frame["slope_60"] = (
            buy_price_history_frame["close"] - close_60_bars_ago
        ) / close_60_bars_ago

    # TODO: review
    debug_column_names = [
        "sma_angle",
        "sma_angle_previous",
        "near_price_volume_ratio",
        "near_price_volume_ratio_previous",
        "above_price_volume_ratio",
        "above_price_volume_ratio_previous",
        "near_delta",
        "slope_60",
    ]
    buy_debug_column_names = [
        column_name
        for column_name in debug_column_names
        if column_name in buy_price_history_frame.columns
    ]
    buy_entry_signal_column = f"{buy_strategy_name}_entry_signal"
    buy_raw_entry_signal_column = f"{buy_strategy_name}_raw_entry_signal"
    combined_entry_series = pandas.Series(
        False, index=buy_price_history_frame.index
    )
    if buy_entry_signal_column in buy_price_history_frame.columns or (
        buy_raw_entry_signal_column in buy_price_history_frame.columns
    ):
        raw_entry_series = (
            buy_price_history_frame.get(
                buy_raw_entry_signal_column,
                pandas.Series(False, index=buy_price_history_frame.index),
            )
            .fillna(False)
            .astype(bool)
        )
        shifted_entry_series = (
            buy_price_history_frame.get(
                buy_entry_signal_column,
                pandas.Series(False, index=buy_price_history_frame.index),
            )
            .fillna(False)
            .astype(bool)
        )
        aligned_shifted_entry_series = shifted_entry_series.shift(
            -1, fill_value=False
        )
        combined_entry_series = (
            raw_entry_series | aligned_shifted_entry_series.astype(bool)
        )
    if buy_entry_signal_column in buy_price_history_frame.columns:
        buy_debug_column_names.append(buy_entry_signal_column)
    debug_frame = buy_price_history_frame[buy_debug_column_names]

    sell_exit_signal_column = f"{sell_strategy_name}_exit_signal"
    sell_raw_exit_signal_column = f"{sell_strategy_name}_raw_exit_signal"
    combined_exit_series = pandas.Series(
        False, index=sell_price_history_frame.index
    )
    if sell_exit_signal_column in sell_price_history_frame.columns or (
        sell_raw_exit_signal_column in sell_price_history_frame.columns
    ):
        raw_exit_series = (
            sell_price_history_frame.get(
                sell_raw_exit_signal_column,
                pandas.Series(False, index=sell_price_history_frame.index),
            )
            .fillna(False)
            .astype(bool)
        )
        shifted_exit_series = (
            sell_price_history_frame.get(
                sell_exit_signal_column,
                pandas.Series(False, index=sell_price_history_frame.index),
            )
            .fillna(False)
            .astype(bool)
        )
        combined_exit_series = raw_exit_series | shifted_exit_series
    if sell_exit_signal_column in sell_price_history_frame.columns:
        debug_frame = debug_frame.join(
            sell_price_history_frame[[sell_exit_signal_column]], how="outer"
        )

    if evaluation_timestamp not in debug_frame.index:
        candidate_index = debug_frame.index[debug_frame.index <= evaluation_timestamp]
        if len(candidate_index) == 0:
            return {
                "sma_angle": None,
                "sma_angle_previous": None,
                "near_price_volume_ratio": None,
                "near_price_volume_ratio_previous": None,
                "above_price_volume_ratio": None,
                "above_price_volume_ratio_previous": None,
                "near_delta": None,
                "slope_60": None,
                "entry": False,
                "exit": False,
            }
        selected_timestamp = candidate_index[-1]
        row = debug_frame.loc[selected_timestamp]
    else:
        selected_timestamp = evaluation_timestamp
        row = debug_frame.loc[evaluation_timestamp]
    entry_value = False
    if selected_timestamp in combined_entry_series.index:
        entry_value = bool(combined_entry_series.loc[selected_timestamp])
    exit_value = False
    if selected_timestamp in combined_exit_series.index:
        exit_value = bool(combined_exit_series.loc[selected_timestamp])
    def normalize_debug_value(value: Any) -> Any:
        if value is None:
            return None
        if pandas.isna(value):
            return None
        return value

    return {
        "sma_angle": normalize_debug_value(row.get("sma_angle")),
        "sma_angle_previous": normalize_debug_value(
            row.get("sma_angle_previous")
        ),
        "near_price_volume_ratio": normalize_debug_value(
            row.get("near_price_volume_ratio")
        ),
        "near_price_volume_ratio_previous": normalize_debug_value(
            row.get("near_price_volume_ratio_previous")
        ),
        "above_price_volume_ratio": normalize_debug_value(
            row.get("above_price_volume_ratio")
        ),
        "above_price_volume_ratio_previous": normalize_debug_value(
            row.get("above_price_volume_ratio_previous")
        ),
        "near_delta": normalize_debug_value(row.get("near_delta")),
        "slope_60": normalize_debug_value(row.get("slope_60")),
        "entry": entry_value,
        "exit": exit_value,
    }
