"""Symbol seasoning gate helpers for production universe promotion.

The seasoning gate lets a symbol exist in production data files while blocking
new entries until its audited first eligible trade date. Missing symbols are
intentionally treated as ineligible when the gate is enabled.
"""

# TODO: review

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable


DEFAULT_ELIGIBILITY_PATH = "data/production_symbol_eligibility.csv"
DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS = 365
CSV_ELIGIBILITY_SOURCE = "csv"
PRICE_HISTORY_ELIGIBILITY_SOURCE = "price_history"
VALID_ELIGIBILITY_SOURCES = {
    CSV_ELIGIBILITY_SOURCE,
    PRICE_HISTORY_ELIGIBILITY_SOURCE,
}
PRICE_HISTORY_DATE_COLUMN = "Date"
REQUIRED_ELIGIBILITY_COLUMNS = {
    "symbol",
    "first_eligible_trade_date",
    "source",
    "notes",
}


@dataclass(frozen=True)
class SymbolSeasoningConfig:
    """Configuration for the symbol seasoning entry gate."""

    enabled: bool = False
    eligibility_path: str | None = None
    default_new_symbol_quarantine_days: int = (
        DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS
    )
    eligibility_source: str = CSV_ELIGIBILITY_SOURCE
    quarantine_trading_bars: int | None = None


def _parse_optional_non_negative_integer(
    raw_value: Any,
    field_name: str,
) -> int | None:
    """Parse an optional non-negative integer config field."""

    if raw_value in (None, ""):
        return None
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError) as parse_error:
        raise ValueError(f"{field_name} must be an integer") from parse_error
    if parsed_value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed_value


def parse_symbol_seasoning_config(
    raw_symbol_seasoning_config: Any,
) -> SymbolSeasoningConfig:
    """Parse the optional ``symbol_seasoning`` JSON block.

    ``None`` and ``False`` disable the gate. ``True`` enables it with the
    production defaults. A dictionary may override the default path and
    quarantine length.
    """

    if raw_symbol_seasoning_config in (None, False):
        return SymbolSeasoningConfig()
    if raw_symbol_seasoning_config is True:
        return SymbolSeasoningConfig(
            enabled=True,
            eligibility_path=DEFAULT_ELIGIBILITY_PATH,
        )
    if not isinstance(raw_symbol_seasoning_config, dict):
        raise ValueError("symbol_seasoning must be a JSON object or boolean")

    enabled = bool(raw_symbol_seasoning_config.get("enabled", False))
    raw_eligibility_path = raw_symbol_seasoning_config.get(
        "eligibility_path",
        DEFAULT_ELIGIBILITY_PATH if enabled else None,
    )
    eligibility_path_text = (
        str(raw_eligibility_path) if raw_eligibility_path is not None else None
    )
    try:
        quarantine_days = int(
            raw_symbol_seasoning_config.get(
                "default_new_symbol_quarantine_days",
                DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS,
            )
        )
    except (TypeError, ValueError) as parse_error:
        raise ValueError(
            "symbol_seasoning.default_new_symbol_quarantine_days must be an integer"
        ) from parse_error
    if quarantine_days < 0:
        raise ValueError(
            "symbol_seasoning.default_new_symbol_quarantine_days must be >= 0"
        )
    eligibility_source = str(
        raw_symbol_seasoning_config.get(
            "eligibility_source",
            CSV_ELIGIBILITY_SOURCE,
        )
    ).strip()
    if eligibility_source not in VALID_ELIGIBILITY_SOURCES:
        valid_source_text = ", ".join(sorted(VALID_ELIGIBILITY_SOURCES))
        raise ValueError(
            "symbol_seasoning.eligibility_source must be one of: "
            f"{valid_source_text}"
        )

    raw_quarantine_trading_bars = raw_symbol_seasoning_config.get(
        "quarantine_trading_bars",
        raw_symbol_seasoning_config.get(
            "default_new_symbol_quarantine_bars",
            None,
        ),
    )
    quarantine_trading_bars = _parse_optional_non_negative_integer(
        raw_quarantine_trading_bars,
        "symbol_seasoning.quarantine_trading_bars",
    )

    return SymbolSeasoningConfig(
        enabled=enabled,
        eligibility_path=eligibility_path_text,
        default_new_symbol_quarantine_days=quarantine_days,
        eligibility_source=eligibility_source,
        quarantine_trading_bars=quarantine_trading_bars,
    )


def resolve_eligibility_path(
    seasoning_config: SymbolSeasoningConfig,
    *,
    repository_root: Path,
) -> Path:
    """Resolve the eligibility CSV path relative to the repository root."""

    raw_path_text = seasoning_config.eligibility_path or DEFAULT_ELIGIBILITY_PATH
    eligibility_path = Path(raw_path_text).expanduser()
    if eligibility_path.is_absolute():
        return eligibility_path
    return repository_root / eligibility_path


def load_symbol_first_eligible_trade_dates(
    eligibility_path: Path,
) -> Dict[str, datetime.date]:
    """Load ``symbol -> first_eligible_trade_date`` from the CSV file.

    The loader validates required columns and duplicate symbols so an ambiguous
    production audit file cannot silently allow or block the wrong symbol.
    """

    if not eligibility_path.exists():
        raise FileNotFoundError(
            f"symbol_seasoning.eligibility_path not found: {eligibility_path}"
        )

    symbol_first_eligible_trade_dates: Dict[str, datetime.date] = {}
    with eligibility_path.open("r", encoding="utf-8", newline="") as eligibility_file:
        reader = csv.DictReader(eligibility_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_ELIGIBILITY_COLUMNS - fieldnames
        if missing_columns:
            missing_column_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                "symbol eligibility CSV is missing required columns: "
                f"{missing_column_text}"
            )

        for line_number, eligibility_row in enumerate(reader, start=2):
            symbol_name = eligibility_row.get("symbol", "").strip().upper()
            if not symbol_name:
                raise ValueError(
                    f"symbol eligibility CSV has a blank symbol on line {line_number}"
                )
            if symbol_name in symbol_first_eligible_trade_dates:
                raise ValueError(
                    "symbol eligibility CSV has duplicate symbol: "
                    f"{symbol_name}"
                )
            raw_date_text = eligibility_row.get(
                "first_eligible_trade_date",
                "",
            ).strip()
            try:
                first_eligible_trade_date = datetime.date.fromisoformat(
                    raw_date_text
                )
            except ValueError as parse_error:
                raise ValueError(
                    "symbol eligibility CSV has invalid first_eligible_trade_date "
                    f"for {symbol_name}: {raw_date_text}"
                ) from parse_error
            symbol_first_eligible_trade_dates[symbol_name] = (
                first_eligible_trade_date
            )

    return symbol_first_eligible_trade_dates


def resolve_price_history_path_for_symbol(
    data_directory: Path,
    symbol_name: str,
) -> Path:
    """Return the local price-history CSV path for ``symbol_name``.

    Yahoo class-share files in the historical cache use dots, while the
    production universe stores those symbols with hyphens.  Try the production
    spelling first, then the Yahoo file spelling.
    """

    normalized_symbol_name = symbol_name.strip().upper()
    direct_price_history_path = data_directory / f"{normalized_symbol_name}.csv"
    if direct_price_history_path.exists():
        return direct_price_history_path
    yahoo_symbol_name = normalized_symbol_name.replace("-", ".")
    return data_directory / f"{yahoo_symbol_name}.csv"


def load_trade_dates_from_price_history(
    price_history_path: Path,
) -> list[datetime.date]:
    """Load sorted unique trade dates from one local price-history CSV."""

    with price_history_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as price_history_file:
        reader = csv.DictReader(price_history_file)
        fieldnames = set(reader.fieldnames or [])
        if PRICE_HISTORY_DATE_COLUMN not in fieldnames:
            raise ValueError(
                "price history CSV is missing required Date column: "
                f"{price_history_path}"
            )
        trade_dates: set[datetime.date] = set()
        for row_number, price_history_row in enumerate(reader, start=2):
            raw_trade_date_text = str(
                price_history_row.get(PRICE_HISTORY_DATE_COLUMN, "")
            ).strip()
            if not raw_trade_date_text:
                continue
            try:
                trade_dates.add(
                    datetime.date.fromisoformat(raw_trade_date_text[:10])
                )
            except ValueError as parse_error:
                raise ValueError(
                    "price history CSV has invalid Date on line "
                    f"{row_number} in {price_history_path}: "
                    f"{raw_trade_date_text}"
                ) from parse_error
    return sorted(trade_dates)


def calculate_first_eligible_trade_date_from_history(
    trade_dates: Iterable[datetime.date],
    *,
    quarantine_calendar_days: int = DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS,
    quarantine_trading_bars: int | None = None,
) -> datetime.date | None:
    """Return the first date after first-bar quarantine constraints.

    When both calendar-day and trading-bar quarantine settings are supplied,
    the later date wins.  That blocks the first ``quarantine_trading_bars``
    observed bars and also blocks entries before ``quarantine_calendar_days``
    calendar days have elapsed from the first observed local price bar.
    """

    if quarantine_calendar_days < 0:
        raise ValueError("quarantine_calendar_days must be >= 0")
    if quarantine_trading_bars is not None and quarantine_trading_bars < 0:
        raise ValueError("quarantine_trading_bars must be >= 0")

    sorted_trade_dates = sorted(set(trade_dates))
    if not sorted_trade_dates:
        return None

    first_trade_date = sorted_trade_dates[0]
    first_eligible_trade_date = first_trade_date + datetime.timedelta(
        days=quarantine_calendar_days
    )
    if quarantine_trading_bars is None:
        return first_eligible_trade_date
    if len(sorted_trade_dates) <= quarantine_trading_bars:
        return None

    first_date_after_bar_quarantine = sorted_trade_dates[quarantine_trading_bars]
    if first_date_after_bar_quarantine > first_eligible_trade_date:
        return first_date_after_bar_quarantine
    return first_eligible_trade_date


def build_symbol_first_eligible_trade_dates_from_price_history(
    data_directory: Path,
    *,
    allowed_symbols: set[str] | None = None,
    quarantine_calendar_days: int = DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS,
    quarantine_trading_bars: int | None = None,
) -> Dict[str, datetime.date]:
    """Build ``symbol -> first eligible date`` from local price-history files."""

    if not data_directory.exists():
        raise FileNotFoundError(
            f"price history data directory not found: {data_directory}"
        )

    if allowed_symbols is None:
        symbol_names = sorted(
            price_history_path.stem.replace(".", "-").upper()
            for price_history_path in data_directory.glob("*.csv")
        )
    else:
        symbol_names = sorted(
            symbol_name.strip().upper()
            for symbol_name in allowed_symbols
            if symbol_name.strip()
        )

    symbol_first_eligible_trade_dates: Dict[str, datetime.date] = {}
    for symbol_name in symbol_names:
        price_history_path = resolve_price_history_path_for_symbol(
            data_directory,
            symbol_name,
        )
        if not price_history_path.exists():
            continue
        trade_dates = load_trade_dates_from_price_history(price_history_path)
        first_eligible_trade_date = calculate_first_eligible_trade_date_from_history(
            trade_dates,
            quarantine_calendar_days=quarantine_calendar_days,
            quarantine_trading_bars=quarantine_trading_bars,
        )
        if first_eligible_trade_date is None:
            continue
        symbol_first_eligible_trade_dates[symbol_name] = first_eligible_trade_date

    return symbol_first_eligible_trade_dates


def write_symbol_eligibility_csv_from_price_history(
    output_path: Path,
    data_directory: Path,
    *,
    allowed_symbols: set[str] | None = None,
    quarantine_calendar_days: int = DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS,
    quarantine_trading_bars: int | None = None,
) -> Dict[str, datetime.date]:
    """Write an auditable eligibility CSV derived from local price history."""

    symbol_first_eligible_trade_dates = (
        build_symbol_first_eligible_trade_dates_from_price_history(
            data_directory,
            allowed_symbols=allowed_symbols,
            quarantine_calendar_days=quarantine_calendar_days,
            quarantine_trading_bars=quarantine_trading_bars,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "symbol",
                "first_eligible_trade_date",
                "source",
                "notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for symbol_name, first_eligible_trade_date in sorted(
            symbol_first_eligible_trade_dates.items()
        ):
            writer.writerow(
                {
                    "symbol": symbol_name,
                    "first_eligible_trade_date": first_eligible_trade_date.isoformat(),
                    "source": "price_history_first_bar_quarantine",
                    "notes": (
                        "First local price bar plus "
                        f"{quarantine_calendar_days} calendar days"
                        + (
                            f" and {quarantine_trading_bars} trading bars"
                            if quarantine_trading_bars is not None
                            else ""
                        )
                    ),
                }
            )
    return symbol_first_eligible_trade_dates


def normalize_trade_date(raw_trade_date: Any) -> datetime.date:
    """Normalize a date-like value to ``datetime.date``."""

    if isinstance(raw_trade_date, datetime.datetime):
        return raw_trade_date.date()
    if isinstance(raw_trade_date, datetime.date):
        return raw_trade_date
    if hasattr(raw_trade_date, "date"):
        possible_date = raw_trade_date.date()
        if isinstance(possible_date, datetime.date):
            return possible_date
    return datetime.date.fromisoformat(str(raw_trade_date))


def is_symbol_eligible_on(
    symbol_name: str,
    trade_date: Any,
    symbol_first_eligible_trade_dates: Dict[str, datetime.date],
) -> bool:
    """Return whether ``symbol_name`` may open a new trade on ``trade_date``.

    Missing symbols are blocked. Date comparison is inclusive.
    """

    normalized_symbol_name = symbol_name.strip().upper()
    first_eligible_trade_date = symbol_first_eligible_trade_dates.get(
        normalized_symbol_name
    )
    if first_eligible_trade_date is None:
        return False
    return normalize_trade_date(trade_date) >= first_eligible_trade_date


def calculate_first_eligible_trade_date(
    promotion_date: Any,
    quarantine_days: int = DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS,
) -> datetime.date:
    """Return the default first eligible date for a production promotion."""

    if quarantine_days < 0:
        raise ValueError("quarantine_days must be >= 0")
    return normalize_trade_date(promotion_date) + datetime.timedelta(
        days=quarantine_days
    )
