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
    """Production eligibility should be data-derived, not a 2010 hard floor."""

    eligibility_dates = symbol_seasoning.load_symbol_first_eligible_trade_dates(
        PRODUCTION_ELIGIBILITY_PATH
    )
    production_symbols = _load_production_symbols()

    assert set(eligibility_dates).issubset(production_symbols)
    assert len(eligibility_dates) > 4_000
    assert datetime.date(1995, 1, 3) in set(eligibility_dates.values())
    assert datetime.date(2010, 1, 1) not in set(eligibility_dates.values())
    assert eligibility_dates["AAPL"] == datetime.date(1995, 1, 3)
    assert eligibility_dates["AMZN"] == datetime.date(1998, 5, 15)
    assert eligibility_dates["ABNB"] == datetime.date(2021, 12, 10)


def test_price_history_eligibility_blocks_first_bars_and_calendar_days(
    tmp_path: Path,
) -> None:
    """Price-history seasoning should use the later bar/date quarantine."""

    price_history_path = tmp_path / "AAA.csv"
    price_history_path.write_text(
        "\n".join(
            [
                "Date,open,high,low,close,volume",
                "2020-01-01,1,1,1,1,100",
                "2020-01-02,1,1,1,1,100",
                "2020-01-03,1,1,1,1,100",
                "2020-01-06,1,1,1,1,100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    eligibility_dates = (
        symbol_seasoning.build_symbol_first_eligible_trade_dates_from_price_history(
            tmp_path,
            allowed_symbols={"AAA"},
            quarantine_calendar_days=3,
            quarantine_trading_bars=2,
        )
    )

    assert eligibility_dates == {"AAA": datetime.date(2020, 1, 4)}


def test_missing_symbol_is_not_eligible() -> None:
    """Missing eligibility rows should block entries instead of allowing them."""

    eligibility_dates = {"AAA": datetime.date(2020, 1, 1)}

    assert not symbol_seasoning.is_symbol_eligible_on(
        "MISSING",
        "2026-01-01",
        eligibility_dates,
    )


def test_parse_price_history_seasoning_config() -> None:
    """Config should preserve data-driven quarantine settings."""

    config = symbol_seasoning.parse_symbol_seasoning_config(
        {
            "enabled": True,
            "eligibility_source": "price_history",
            "default_new_symbol_quarantine_days": 365,
            "quarantine_trading_bars": 252,
        }
    )

    assert config.enabled
    assert config.eligibility_source == "price_history"
    assert config.default_new_symbol_quarantine_days == 365
    assert config.quarantine_trading_bars == 252


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
