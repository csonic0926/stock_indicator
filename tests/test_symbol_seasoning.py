"""Tests for production symbol seasoning eligibility."""

from __future__ import annotations

import datetime
from pathlib import Path

from stock_indicator import symbol_seasoning


DATA_DIRECTORY = Path("data")
PRODUCTION_SYMBOLS_PATH = DATA_DIRECTORY / "production_symbols.txt"
PRODUCTION_ELIGIBILITY_PATH = DATA_DIRECTORY / "production_symbol_eligibility.csv"


def _load_production_symbols() -> set[str]:
    """Return normalized production-old symbols from the committed baseline."""

    return {
        line_text.strip().upper()
        for line_text in PRODUCTION_SYMBOLS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line_text.strip()
    }


def test_production_symbols_are_baseline_eligible() -> None:
    """Every baseline production symbol should be eligible from 2010-01-01."""

    eligibility_dates = symbol_seasoning.load_symbol_first_eligible_trade_dates(
        PRODUCTION_ELIGIBILITY_PATH
    )

    assert set(eligibility_dates) == _load_production_symbols()
    assert set(eligibility_dates.values()) == {datetime.date(2010, 1, 1)}


def test_missing_symbol_is_not_eligible() -> None:
    """Missing eligibility rows should block entries instead of allowing them."""

    eligibility_dates = {"AAA": datetime.date(2020, 1, 1)}

    assert not symbol_seasoning.is_symbol_eligible_on(
        "MISSING",
        "2026-01-01",
        eligibility_dates,
    )


def test_symbol_eligibility_date_is_inclusive() -> None:
    """A symbol may trade on its first eligible date, but not before it."""

    eligibility_dates = {"AAA": datetime.date(2024, 1, 5)}

    assert not symbol_seasoning.is_symbol_eligible_on(
        "AAA",
        "2024-01-04",
        eligibility_dates,
    )
    assert symbol_seasoning.is_symbol_eligible_on(
        "AAA",
        "2024-01-05",
        eligibility_dates,
    )
