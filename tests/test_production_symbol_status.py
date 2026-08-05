"""Tests for the non-destructive production-symbol status contract."""

# TODO: review

from __future__ import annotations

import datetime
from pathlib import Path

import pandas

from stock_indicator import production_symbol_status


def _build_nasdaq_directory_frame() -> pandas.DataFrame:
    """Return a compact Nasdaq directory fixture."""

    return pandas.DataFrame(
        [
            {
                "Symbol": "AAA",
                "Security Name": "Alpha Corporation - Common Stock",
                "Test Issue": "N",
                "ETF": "N",
            },
            {
                "Symbol": "BBB",
                "Security Name": "Beta Index ETF",
                "Test Issue": "N",
                "ETF": "Y",
            },
            {
                "Symbol": "CCC.W",
                "Security Name": "Charlie Corporation - Warrant",
                "Test Issue": "N",
                "ETF": "N",
            },
            {
                "Symbol": "STALE",
                "Security Name": "Stale Corporation - Common Stock",
                "Test Issue": "N",
                "ETF": "N",
            },
        ]
    )


def _build_empty_other_directory_frame() -> pandas.DataFrame:
    """Return an empty consolidated-directory fixture with its schema."""

    return pandas.DataFrame(
        columns=[
            "ACT Symbol",
            "Security Name",
            "Exchange",
            "Test Issue",
            "ETF",
        ]
    )


def test_status_builder_retains_symbols_and_separates_price_recovery() -> None:
    """Inactive rows stay recorded while stale prices remain refreshable."""

    production_symbols = ["AAA", "BBB", "CCC-W", "STALE", "GONE"]
    listing_frame = production_symbol_status.build_current_exchange_listing_frame(
        _build_nasdaq_directory_frame(),
        _build_empty_other_directory_frame(),
    )
    existing_status_records = {
        "AAA": {
            "symbol": "AAA",
            "status": "active",
            "status_reason": "listed_common_stock_with_current_price",
            "status_changed_on": "2026-07-01",
        },
        "BBB": {
            "symbol": "BBB",
            "status": "active",
            "status_reason": "listed_common_stock_with_current_price",
            "status_changed_on": "2026-07-01",
        },
    }

    status_frame, changed_symbols = (
        production_symbol_status.build_production_symbol_status_frame(
            production_symbols=production_symbols,
            current_listing_frame=listing_frame,
            price_last_dates={
                "AAA": "2026-08-04",
                "BBB": "2026-08-04",
                "CCC-W": "2026-08-04",
                "STALE": "2026-07-31",
                "GONE": "2026-07-01",
            },
            target_price_date=datetime.date(2026, 8, 4),
            status_observed_date=datetime.date(2026, 8, 5),
            existing_status_records=existing_status_records,
        )
    )

    statuses_by_symbol = status_frame.set_index("symbol")["status"].to_dict()
    assert statuses_by_symbol == {
        "AAA": "active",
        "BBB": "inactive",
        "CCC-W": "inactive",
        "STALE": "price_unavailable",
        "GONE": "inactive",
    }
    reasons_by_symbol = status_frame.set_index("symbol")[
        "status_reason"
    ].to_dict()
    assert reasons_by_symbol["BBB"] == "exchange_directory_etf"
    assert reasons_by_symbol["CCC-W"].startswith("non_common_stock:")
    assert reasons_by_symbol["GONE"] == "not_in_current_exchange_directories"
    assert status_frame.set_index("symbol").loc[
        "AAA", "status_changed_on"
    ] == "2026-07-01"
    assert set(changed_symbols) == {"BBB", "CCC-W", "STALE", "GONE"}


def test_status_loaders_use_distinct_download_and_entry_views(
    tmp_path: Path,
) -> None:
    """Price-unavailable symbols retry downloads but cannot open entries."""

    production_symbols_path = tmp_path / "production_symbols.txt"
    production_symbols_path.write_text(
        "AAA\nSTALE\nGONE\n",
        encoding="utf-8",
    )
    status_path = tmp_path / "production_symbol_status.csv"
    pandas.DataFrame(
        [
            {
                "symbol": "AAA",
                "status": "active",
                "status_reason": "listed_common_stock_with_current_price",
                "exchange": "NASDAQ",
                "security_name": "Alpha Corporation",
                "price_last_date": "2026-08-04",
                "status_changed_on": "2026-08-05",
            },
            {
                "symbol": "STALE",
                "status": "price_unavailable",
                "status_reason": "price_cache_missing_latest_market_date",
                "exchange": "NASDAQ",
                "security_name": "Stale Corporation",
                "price_last_date": "2026-07-31",
                "status_changed_on": "2026-08-05",
            },
            {
                "symbol": "GONE",
                "status": "inactive",
                "status_reason": "not_in_current_exchange_directories",
                "exchange": "",
                "security_name": "",
                "price_last_date": "2026-07-01",
                "status_changed_on": "2026-08-05",
            },
        ],
        columns=production_symbol_status.STATUS_COLUMNS,
    ).to_csv(status_path, index=False)

    refresh_symbols = (
        production_symbol_status.load_symbols_allowed_for_price_refresh(
            production_symbols_path,
            status_path,
        )
    )
    blocked_symbols = (
        production_symbol_status.load_symbols_blocked_for_new_entries(
            production_symbols_path,
            status_path,
        )
    )

    assert refresh_symbols == ["AAA", "STALE"]
    assert blocked_symbols == {"STALE", "GONE"}


def test_download_view_bootstraps_new_symbol_before_total_status_refresh(
    tmp_path: Path,
) -> None:
    """A newly promoted statusless symbol downloads before signal validation."""

    production_symbols_path = tmp_path / "production_symbols.txt"
    production_symbols_path.write_text("AAA\nNEW\n", encoding="utf-8")
    status_path = tmp_path / "production_symbol_status.csv"
    pandas.DataFrame(
        [
            {
                "symbol": "AAA",
                "status": "active",
                "status_reason": "listed_common_stock_with_current_price",
                "exchange": "NASDAQ",
                "security_name": "Alpha Corporation",
                "price_last_date": "2026-08-04",
                "status_changed_on": "2026-08-05",
            }
        ],
        columns=production_symbol_status.STATUS_COLUMNS,
    ).to_csv(status_path, index=False)

    refresh_symbols = (
        production_symbol_status.load_symbols_allowed_for_price_refresh(
            production_symbols_path,
            status_path,
        )
    )

    assert refresh_symbols == ["AAA", "NEW"]


def test_status_update_validates_group_and_seasoning_contracts(
    tmp_path: Path,
) -> None:
    """A refresh publishes status without removing any production contract row."""

    data_directory = tmp_path / "data"
    price_data_directory = data_directory / "stock_data"
    price_data_directory.mkdir(parents=True)
    production_symbols = ["AAA", "STALE"]
    (data_directory / "production_symbols.txt").write_text(
        "AAA\nSTALE\n",
        encoding="utf-8",
    )
    (data_directory / "symbols.txt").write_text(
        "AAA\nSTALE\n",
        encoding="utf-8",
    )
    pandas.DataFrame(
        [
            {
                "ticker": symbol_name,
                "ff12": 6,
                "ff12_source": "legacy_backtest",
                "classification_confidence": "high",
            }
            for symbol_name in production_symbols
        ]
    ).to_parquet(
        data_directory / "production_symbols_with_sector.parquet",
        index=False,
    )
    pandas.DataFrame(
        [
            {
                "symbol": "AAA",
                "first_eligible_trade_date": "2025-01-01",
                "source": "test",
                "notes": "",
            }
        ]
    ).to_csv(
        data_directory / "production_symbol_eligibility.csv",
        index=False,
    )
    for symbol_name, last_date in [
        ("AAA", "2026-08-04"),
        ("STALE", "2026-07-31"),
    ]:
        pandas.DataFrame(
            {"close": [1.0]},
            index=pandas.to_datetime([last_date]),
        ).to_csv(price_data_directory / f"{symbol_name}.csv")

    report = production_symbol_status.update_production_symbol_status(
        data_directory=data_directory,
        price_data_directory=price_data_directory,
        target_price_date=datetime.date(2026, 8, 4),
        status_observed_date=datetime.date(2026, 8, 5),
        nasdaq_listed_frame=_build_nasdaq_directory_frame(),
        other_listed_frame=_build_empty_other_directory_frame(),
    )

    published_frame = pandas.read_csv(
        data_directory / "production_symbol_status.csv",
        keep_default_na=False,
    )
    assert report.production_symbol_count == 2
    assert report.group_row_count == 2
    assert report.eligibility_row_count == 1
    assert published_frame["symbol"].tolist() == production_symbols
    assert report.status_counts == {"active": 1, "price_unavailable": 1}
