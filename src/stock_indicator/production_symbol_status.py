"""Maintain the non-destructive production-symbol trading status contract.

``production_symbols.txt`` is an append-preserving observation universe.  A
symbol that stops matching the trading system is retained there and receives a
status row instead of being deleted.  Runtime price refresh and entry selection
consume different views of this status contract so temporarily unavailable
prices can recover while inactive instruments stay out of new positions.
"""

# TODO: review

from __future__ import annotations

import datetime
from dataclasses import dataclass
from io import StringIO
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas
import requests

from stock_indicator.production_ff12_promotion import (
    validate_production_ff12_sector_contract,
)
from stock_indicator.symbols import (
    identify_non_common_stock_reason,
    normalize_symbol_for_cache,
)

LOGGER = logging.getLogger(__name__)

NASDAQ_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)
OTHER_LISTED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)

PRODUCTION_SYMBOLS_FILE_NAME = "production_symbols.txt"
PRODUCTION_SYMBOL_STATUS_FILE_NAME = "production_symbol_status.csv"
PRODUCTION_SECTOR_PARQUET_FILE_NAME = "production_symbols_with_sector.parquet"
PRODUCTION_SECTOR_CSV_FILE_NAME = "production_symbols_with_sector.csv"
PRODUCTION_ELIGIBILITY_FILE_NAME = "production_symbol_eligibility.csv"
RUNTIME_SYMBOLS_FILE_NAME = "symbols.txt"

ACTIVE_STATUS = "active"
INACTIVE_STATUS = "inactive"
PRICE_UNAVAILABLE_STATUS = "price_unavailable"
VALID_STATUSES = {
    ACTIVE_STATUS,
    INACTIVE_STATUS,
    PRICE_UNAVAILABLE_STATUS,
}

STATUS_COLUMNS = [
    "symbol",
    "status",
    "status_reason",
    "exchange",
    "security_name",
    "price_last_date",
    "status_changed_on",
]

EXCHANGE_NAMES_BY_CODE = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "V": "IEX",
    "Z": "Cboe BZX",
}


@dataclass(frozen=True)
class ProductionSymbolStatusPaths:
    """Paths participating in the production-symbol status contract."""

    data_directory: Path

    @property
    def production_symbols_path(self) -> Path:
        """Return the append-preserving production symbol path."""

        return self.data_directory / PRODUCTION_SYMBOLS_FILE_NAME

    @property
    def status_path(self) -> Path:
        """Return the current production symbol status path."""

        return self.data_directory / PRODUCTION_SYMBOL_STATUS_FILE_NAME

    @property
    def production_sector_parquet_path(self) -> Path:
        """Return the production FF12 parquet path."""

        return self.data_directory / PRODUCTION_SECTOR_PARQUET_FILE_NAME

    @property
    def production_sector_csv_path(self) -> Path:
        """Return the production FF12 CSV path."""

        return self.data_directory / PRODUCTION_SECTOR_CSV_FILE_NAME

    @property
    def eligibility_path(self) -> Path:
        """Return the production seasoning eligibility path."""

        return self.data_directory / PRODUCTION_ELIGIBILITY_FILE_NAME

    @property
    def runtime_symbols_path(self) -> Path:
        """Return the compatibility runtime symbol mirror path."""

        return self.data_directory / RUNTIME_SYMBOLS_FILE_NAME


@dataclass(frozen=True)
class ProductionSymbolStatusReport:
    """Summary of one production status refresh."""

    published: bool
    production_symbol_count: int
    group_row_count: int
    eligibility_row_count: int
    status_counts: dict[str, int]
    changed_symbols: list[str]
    target_price_date: str
    status_path: Path

    def to_lines(self) -> list[str]:
        """Return a compact human-readable refresh report."""

        action_label = "published" if self.published else "dry run completed"
        count_text = ", ".join(
            f"{status_name}={self.status_counts[status_name]}"
            for status_name in sorted(self.status_counts)
        )
        changed_sample = ", ".join(self.changed_symbols[:20]) or "none"
        return [
            f"Production symbol status refresh {action_label}",
            f"production symbols: {self.production_symbol_count}",
            f"FF12 rows: {self.group_row_count}",
            f"seasoning eligibility rows: {self.eligibility_row_count}",
            f"target price date: {self.target_price_date}",
            f"statuses: {count_text}",
            f"changed statuses: {len(self.changed_symbols)} ({changed_sample})",
            f"status path: {self.status_path}",
        ]


def load_production_symbols_preserving_vendor_format(
    production_symbols_path: Path,
) -> list[str]:
    """Load production symbols without changing Yahoo separator syntax."""

    if not production_symbols_path.exists():
        raise FileNotFoundError(
            f"production symbols file not found: {production_symbols_path}"
        )

    production_symbols: list[str] = []
    normalized_symbol_to_vendor_symbol: dict[str, str] = {}
    for line_number, line_text in enumerate(
        production_symbols_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        vendor_symbol = line_text.strip().upper()
        if not vendor_symbol:
            continue
        normalized_symbol = normalize_symbol_for_cache(vendor_symbol)
        prior_vendor_symbol = normalized_symbol_to_vendor_symbol.get(
            normalized_symbol
        )
        if prior_vendor_symbol is not None:
            raise ValueError(
                "production symbols file has duplicate normalized symbol "
                f"on line {line_number}: {prior_vendor_symbol}, {vendor_symbol}"
            )
        normalized_symbol_to_vendor_symbol[normalized_symbol] = vendor_symbol
        production_symbols.append(vendor_symbol)

    if not production_symbols:
        raise ValueError(f"production symbols file is empty: {production_symbols_path}")
    return production_symbols


def _read_pipe_delimited_directory(directory_url: str) -> pandas.DataFrame:
    """Download one Nasdaq Trader pipe-delimited symbol directory."""

    response = requests.get(directory_url, timeout=30)
    response.raise_for_status()
    directory_frame = pandas.read_csv(
        StringIO(response.text),
        sep="|",
        dtype=str,
        keep_default_na=False,
    )
    directory_frame = directory_frame.loc[
        :,
        ~directory_frame.columns.astype(str).str.startswith("Unnamed"),
    ]
    return directory_frame


def fetch_current_exchange_directory_frames(
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Fetch current Nasdaq and consolidated non-Nasdaq symbol directories."""

    return (
        _read_pipe_delimited_directory(NASDAQ_LISTED_URL),
        _read_pipe_delimited_directory(OTHER_LISTED_URL),
    )


def build_current_exchange_listing_frame(
    nasdaq_listed_frame: pandas.DataFrame,
    other_listed_frame: pandas.DataFrame,
) -> pandas.DataFrame:
    """Normalize official exchange-directory rows into one current listing map."""

    required_nasdaq_columns = {
        "Symbol",
        "Security Name",
        "Test Issue",
        "ETF",
    }
    required_other_columns = {
        "ACT Symbol",
        "Security Name",
        "Exchange",
        "Test Issue",
        "ETF",
    }
    missing_nasdaq_columns = required_nasdaq_columns - set(
        nasdaq_listed_frame.columns
    )
    missing_other_columns = required_other_columns - set(other_listed_frame.columns)
    if missing_nasdaq_columns:
        raise ValueError(
            "Nasdaq listed directory is missing columns: "
            f"{sorted(missing_nasdaq_columns)}"
        )
    if missing_other_columns:
        raise ValueError(
            "other-listed directory is missing columns: "
            f"{sorted(missing_other_columns)}"
        )

    listing_records: list[dict[str, str]] = []
    for listing_record in nasdaq_listed_frame.to_dict("records"):
        raw_symbol = str(listing_record.get("Symbol", "")).strip().upper()
        if not raw_symbol or raw_symbol.startswith("FILE CREATION TIME"):
            continue
        if str(listing_record.get("Test Issue", "")).strip().upper() != "N":
            continue
        listing_records.append(
            {
                "normalized_symbol": normalize_symbol_for_cache(raw_symbol),
                "exchange_symbol": raw_symbol,
                "security_name": str(
                    listing_record.get("Security Name", "")
                ).strip(),
                "exchange": "NASDAQ",
                "is_exchange_traded_fund": str(
                    listing_record.get("ETF", "")
                ).strip().upper(),
            }
        )

    for listing_record in other_listed_frame.to_dict("records"):
        raw_symbol = str(listing_record.get("ACT Symbol", "")).strip().upper()
        if not raw_symbol or raw_symbol.startswith("FILE CREATION TIME"):
            continue
        if str(listing_record.get("Test Issue", "")).strip().upper() != "N":
            continue
        exchange_code = str(listing_record.get("Exchange", "")).strip().upper()
        listing_records.append(
            {
                "normalized_symbol": normalize_symbol_for_cache(raw_symbol),
                "exchange_symbol": raw_symbol,
                "security_name": str(
                    listing_record.get("Security Name", "")
                ).strip(),
                "exchange": EXCHANGE_NAMES_BY_CODE.get(
                    exchange_code,
                    exchange_code,
                ),
                "is_exchange_traded_fund": str(
                    listing_record.get("ETF", "")
                ).strip().upper(),
            }
        )

    listing_frame = pandas.DataFrame(
        listing_records,
        columns=[
            "normalized_symbol",
            "exchange_symbol",
            "security_name",
            "exchange",
            "is_exchange_traded_fund",
        ],
    )
    duplicate_symbols = sorted(
        set(
            listing_frame.loc[
                listing_frame["normalized_symbol"].duplicated(keep=False),
                "normalized_symbol",
            ].astype(str)
        )
    )
    if duplicate_symbols:
        raise ValueError(
            "exchange directories have duplicate normalized symbols: "
            f"{duplicate_symbols[:20]}"
        )
    return listing_frame.sort_values("normalized_symbol").reset_index(drop=True)


def load_price_last_dates(
    production_symbols: list[str],
    price_data_directory: Path,
) -> dict[str, str]:
    """Return the latest valid local price date for each production symbol."""

    price_last_dates: dict[str, str] = {}
    for vendor_symbol in production_symbols:
        price_path = price_data_directory / f"{vendor_symbol}.csv"
        if not price_path.exists():
            price_last_dates[vendor_symbol] = ""
            continue
        try:
            price_date_frame = pandas.read_csv(price_path, usecols=[0])
        except (
            OSError,
            pandas.errors.EmptyDataError,
            pandas.errors.ParserError,
            ValueError,
        ):
            price_last_dates[vendor_symbol] = ""
            continue
        if price_date_frame.empty:
            price_last_dates[vendor_symbol] = ""
            continue
        date_series = pandas.to_datetime(
            price_date_frame.iloc[:, 0],
            errors="coerce",
        ).dropna()
        price_last_dates[vendor_symbol] = (
            pandas.Timestamp(date_series.max()).date().isoformat()
            if not date_series.empty
            else ""
        )
    return price_last_dates


def _load_existing_status_records(status_path: Path) -> dict[str, dict[str, str]]:
    """Load existing status rows for status-change date preservation."""

    if not status_path.exists():
        return {}
    status_frame = pandas.read_csv(status_path, keep_default_na=False, dtype=str)
    missing_columns = set(STATUS_COLUMNS) - set(status_frame.columns)
    if missing_columns:
        raise ValueError(
            "production symbol status file is missing columns: "
            f"{sorted(missing_columns)}"
        )
    duplicate_symbols = status_frame.loc[
        status_frame["symbol"].duplicated(),
        "symbol",
    ].tolist()
    if duplicate_symbols:
        raise ValueError(
            "production symbol status file has duplicate symbols: "
            f"{duplicate_symbols[:20]}"
        )
    return {
        str(status_record["symbol"]).strip().upper(): {
            column_name: str(status_record.get(column_name, ""))
            for column_name in STATUS_COLUMNS
        }
        for status_record in status_frame.to_dict("records")
    }


def build_production_symbol_status_frame(
    *,
    production_symbols: list[str],
    current_listing_frame: pandas.DataFrame,
    price_last_dates: dict[str, str],
    target_price_date: datetime.date,
    status_observed_date: datetime.date,
    existing_status_records: dict[str, dict[str, str]] | None = None,
) -> tuple[pandas.DataFrame, list[str]]:
    """Build current statuses while preserving every production symbol."""

    listing_records_by_symbol = {
        str(listing_record["normalized_symbol"]): listing_record
        for listing_record in current_listing_frame.to_dict("records")
    }
    prior_records = existing_status_records or {}
    status_records: list[dict[str, str]] = []
    changed_symbols: list[str] = []

    for vendor_symbol in production_symbols:
        normalized_symbol = normalize_symbol_for_cache(vendor_symbol)
        listing_record = listing_records_by_symbol.get(normalized_symbol)
        price_last_date = str(price_last_dates.get(vendor_symbol, ""))
        exchange_name = ""
        security_name = ""
        if listing_record is None:
            status_name = INACTIVE_STATUS
            status_reason = "not_in_current_exchange_directories"
        else:
            exchange_name = str(listing_record.get("exchange", ""))
            security_name = str(listing_record.get("security_name", ""))
            is_exchange_traded_fund = (
                str(
                    listing_record.get("is_exchange_traded_fund", "")
                ).strip().upper()
                == "Y"
            )
            non_common_reason = identify_non_common_stock_reason(
                normalized_symbol,
                security_name,
            )
            if is_exchange_traded_fund:
                status_name = INACTIVE_STATUS
                status_reason = "exchange_directory_etf"
            elif non_common_reason is not None:
                status_name = INACTIVE_STATUS
                status_reason = f"non_common_stock:{non_common_reason}"
            elif not price_last_date or (
                datetime.date.fromisoformat(price_last_date) < target_price_date
            ):
                status_name = PRICE_UNAVAILABLE_STATUS
                status_reason = "price_cache_missing_latest_market_date"
            else:
                status_name = ACTIVE_STATUS
                status_reason = "listed_common_stock_with_current_price"

        prior_record = prior_records.get(vendor_symbol)
        status_changed = (
            prior_record is None
            or prior_record.get("status") != status_name
            or prior_record.get("status_reason") != status_reason
        )
        if status_changed:
            status_changed_on = status_observed_date.isoformat()
            changed_symbols.append(vendor_symbol)
        else:
            status_changed_on = str(prior_record.get("status_changed_on", ""))

        status_records.append(
            {
                "symbol": vendor_symbol,
                "status": status_name,
                "status_reason": status_reason,
                "exchange": exchange_name,
                "security_name": security_name,
                "price_last_date": price_last_date,
                "status_changed_on": status_changed_on,
            }
        )

    return pandas.DataFrame(status_records, columns=STATUS_COLUMNS), changed_symbols


def validate_production_symbol_status_frame(
    status_frame: pandas.DataFrame,
    production_symbols: list[str],
) -> None:
    """Validate exact status coverage and supported status values."""

    missing_columns = set(STATUS_COLUMNS) - set(status_frame.columns)
    if missing_columns:
        raise ValueError(
            "production symbol status frame is missing columns: "
            f"{sorted(missing_columns)}"
        )
    status_symbols = status_frame["symbol"].astype(str).str.strip().str.upper()
    duplicate_symbols = status_symbols.loc[status_symbols.duplicated()].tolist()
    if duplicate_symbols:
        raise ValueError(
            "production symbol status frame has duplicate symbols: "
            f"{duplicate_symbols[:20]}"
        )
    expected_symbols = set(production_symbols)
    actual_symbols = set(status_symbols)
    missing_symbols = sorted(expected_symbols - actual_symbols)
    extra_symbols = sorted(actual_symbols - expected_symbols)
    if missing_symbols or extra_symbols:
        raise ValueError(
            "production symbol status does not match production symbols: "
            f"missing={missing_symbols[:20]}, extra={extra_symbols[:20]}"
        )
    invalid_statuses = sorted(
        set(status_frame["status"].astype(str)) - VALID_STATUSES
    )
    if invalid_statuses:
        raise ValueError(
            f"production symbol status has invalid values: {invalid_statuses}"
        )


def load_production_symbol_status_frame(
    status_path: Path,
    production_symbols: list[str],
) -> pandas.DataFrame:
    """Load and validate the current production status contract."""

    if not status_path.exists():
        raise FileNotFoundError(
            f"production symbol status file not found: {status_path}"
        )
    status_frame = pandas.read_csv(status_path, keep_default_na=False, dtype=str)
    validate_production_symbol_status_frame(status_frame, production_symbols)
    return status_frame


def load_symbols_allowed_for_price_refresh(
    production_symbols_path: Path,
    status_path: Path,
) -> list[str]:
    """Return production symbols that should still be retried by Yahoo.

    The download view deliberately admits a newly promoted symbol whose first
    status row has not been generated yet. The post-download status refresh
    must restore exact coverage before live signal loading, which remains
    fail-closed through :func:`load_symbols_blocked_for_new_entries`.
    """

    production_symbols = load_production_symbols_preserving_vendor_format(
        production_symbols_path
    )
    if not status_path.exists():
        LOGGER.warning(
            "Production symbol status file is unavailable during price "
            "refresh; all production symbols will be downloaded before the "
            "post-download status rebuild: %s",
            status_path,
        )
        return production_symbols

    status_frame = pandas.read_csv(status_path, keep_default_na=False, dtype=str)
    missing_columns = set(STATUS_COLUMNS) - set(status_frame.columns)
    if missing_columns:
        raise ValueError(
            "production symbol status frame is missing columns: "
            f"{sorted(missing_columns)}"
        )
    status_symbols = status_frame["symbol"].astype(str).str.strip().str.upper()
    duplicate_symbols = status_symbols.loc[status_symbols.duplicated()].tolist()
    if duplicate_symbols:
        raise ValueError(
            "production symbol status frame has duplicate symbols: "
            f"{duplicate_symbols[:20]}"
        )
    extra_symbols = sorted(set(status_symbols) - set(production_symbols))
    if extra_symbols:
        raise ValueError(
            "production symbol status contains symbols outside production: "
            f"{extra_symbols[:20]}"
        )
    invalid_statuses = sorted(
        set(status_frame["status"].astype(str)) - VALID_STATUSES
    )
    if invalid_statuses:
        raise ValueError(
            f"production symbol status has invalid values: {invalid_statuses}"
        )

    refreshable_statuses = {ACTIVE_STATUS, PRICE_UNAVAILABLE_STATUS}
    statuses_by_symbol = dict(
        zip(status_symbols, status_frame["status"], strict=True)
    )
    return [
        vendor_symbol
        for vendor_symbol in production_symbols
        if statuses_by_symbol.get(vendor_symbol) in refreshable_statuses
        or vendor_symbol not in statuses_by_symbol
    ]


def load_symbols_blocked_for_new_entries(
    production_symbols_path: Path,
    status_path: Path,
) -> set[str]:
    """Return non-active production symbols that cannot open new positions."""

    production_symbols = load_production_symbols_preserving_vendor_format(
        production_symbols_path
    )
    status_frame = load_production_symbol_status_frame(
        status_path,
        production_symbols,
    )
    return set(
        status_frame.loc[
            status_frame["status"] != ACTIVE_STATUS,
            "symbol",
        ].astype(str)
    )


def _read_production_sector_frame(
    paths: ProductionSymbolStatusPaths,
) -> pandas.DataFrame:
    """Read the production FF12 frame, preferring parquet."""

    if paths.production_sector_parquet_path.exists():
        return pandas.read_parquet(paths.production_sector_parquet_path)
    if paths.production_sector_csv_path.exists():
        return pandas.read_csv(
            paths.production_sector_csv_path,
            keep_default_na=False,
        )
    raise FileNotFoundError(
        "production FF12 sector file not found: "
        f"{paths.production_sector_parquet_path} or "
        f"{paths.production_sector_csv_path}"
    )


def validate_related_production_contracts(
    paths: ProductionSymbolStatusPaths,
    production_symbols: list[str],
) -> tuple[int, int]:
    """Validate FF12 coverage, seasoning ownership, and runtime mirror parity."""

    production_sector_frame = _read_production_sector_frame(paths)
    validate_production_ff12_sector_contract(
        production_sector_frame,
        production_symbols,
    )

    if not paths.eligibility_path.exists():
        raise FileNotFoundError(
            f"production symbol eligibility file not found: {paths.eligibility_path}"
        )
    eligibility_frame = pandas.read_csv(
        paths.eligibility_path,
        keep_default_na=False,
        dtype=str,
    )
    if "symbol" not in eligibility_frame.columns:
        raise ValueError("production symbol eligibility file has no symbol column")
    normalized_production_symbols = {
        normalize_symbol_for_cache(symbol_name)
        for symbol_name in production_symbols
    }
    normalized_eligibility_symbols = eligibility_frame["symbol"].map(
        normalize_symbol_for_cache
    )
    duplicate_eligibility_symbols = normalized_eligibility_symbols.loc[
        normalized_eligibility_symbols.duplicated()
    ].tolist()
    if duplicate_eligibility_symbols:
        raise ValueError(
            "production symbol eligibility has duplicate symbols: "
            f"{duplicate_eligibility_symbols[:20]}"
        )
    extra_eligibility_symbols = sorted(
        set(normalized_eligibility_symbols) - normalized_production_symbols
    )
    if extra_eligibility_symbols:
        raise ValueError(
            "production symbol eligibility contains symbols outside production: "
            f"{extra_eligibility_symbols[:20]}"
        )

    runtime_symbols = load_production_symbols_preserving_vendor_format(
        paths.runtime_symbols_path
    )
    if runtime_symbols != production_symbols:
        raise ValueError(
            "symbols.txt compatibility mirror does not match "
            "production_symbols.txt"
        )
    return len(production_sector_frame), len(eligibility_frame)


def _write_status_frame_atomically(
    status_frame: pandas.DataFrame,
    status_path: Path,
) -> None:
    """Atomically publish the production status CSV."""

    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file_descriptor, temporary_file_name = tempfile.mkstemp(
        dir=status_path.parent,
        prefix=f".{status_path.name}.",
        suffix=".tmp",
    )
    os.close(temporary_file_descriptor)
    temporary_path = Path(temporary_file_name)
    try:
        status_frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, status_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def update_production_symbol_status(
    *,
    data_directory: Path,
    price_data_directory: Path,
    target_price_date: datetime.date,
    status_observed_date: datetime.date | None = None,
    publish_outputs: bool = True,
    nasdaq_listed_frame: pandas.DataFrame | None = None,
    other_listed_frame: pandas.DataFrame | None = None,
) -> ProductionSymbolStatusReport:
    """Refresh and optionally publish the production-symbol status contract."""

    paths = ProductionSymbolStatusPaths(Path(data_directory))
    production_symbols = load_production_symbols_preserving_vendor_format(
        paths.production_symbols_path
    )
    group_row_count, eligibility_row_count = validate_related_production_contracts(
        paths,
        production_symbols,
    )

    if nasdaq_listed_frame is None or other_listed_frame is None:
        nasdaq_listed_frame, other_listed_frame = (
            fetch_current_exchange_directory_frames()
        )
    current_listing_frame = build_current_exchange_listing_frame(
        nasdaq_listed_frame,
        other_listed_frame,
    )
    price_last_dates = load_price_last_dates(
        production_symbols,
        Path(price_data_directory),
    )
    existing_status_records = _load_existing_status_records(paths.status_path)
    status_frame, changed_symbols = build_production_symbol_status_frame(
        production_symbols=production_symbols,
        current_listing_frame=current_listing_frame,
        price_last_dates=price_last_dates,
        target_price_date=target_price_date,
        status_observed_date=status_observed_date or datetime.date.today(),
        existing_status_records=existing_status_records,
    )
    validate_production_symbol_status_frame(status_frame, production_symbols)

    if publish_outputs:
        _write_status_frame_atomically(status_frame, paths.status_path)

    status_counts = {
        str(status_name): int(status_count)
        for status_name, status_count in status_frame["status"].value_counts().items()
    }
    LOGGER.info(
        "Production symbol status refresh %s: %d symbols, %s",
        "published" if publish_outputs else "validated",
        len(status_frame),
        status_counts,
    )
    return ProductionSymbolStatusReport(
        published=publish_outputs,
        production_symbol_count=len(production_symbols),
        group_row_count=group_row_count,
        eligibility_row_count=eligibility_row_count,
        status_counts=status_counts,
        changed_symbols=changed_symbols,
        target_price_date=target_price_date.isoformat(),
        status_path=paths.status_path,
    )
