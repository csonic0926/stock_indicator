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
from typing import Any, Dict


DEFAULT_ELIGIBILITY_PATH = "data/production_symbol_eligibility.csv"
DEFAULT_NEW_SYMBOL_QUARANTINE_DAYS = 365
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

    return SymbolSeasoningConfig(
        enabled=enabled,
        eligibility_path=eligibility_path_text,
        default_new_symbol_quarantine_days=quarantine_days,
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
