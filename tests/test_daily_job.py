"""Tests for daily job utilities that do not depend on cron helpers."""

# TODO: review

from pathlib import Path
import os
import sys

import logging
import pandas
import pytest
import yfinance.exceptions as yfinance_exceptions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator import daily_job


def test_update_all_data_from_yf_deduplicates_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``update_all_data_from_yf`` should remove duplicate rows."""

    data_directory = tmp_path
    csv_path = data_directory / "AAA.csv"
    csv_path.write_text(
        "Date,open,close\n2024-01-01,1,1\n2024-01-02,1,1\n",
        encoding="utf-8",
    )

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        existing_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
        new_frame = pandas.DataFrame(
            {"open": [1, 1], "close": [1, 1]},
            index=pandas.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        combined_frame = pandas.concat([existing_frame, new_frame])
        combined_frame.to_csv(cache_path)
        return combined_frame

    monkeypatch.setattr(daily_job, "download_history", fake_download_history)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["AAA"])
    monkeypatch.setattr(
        daily_job.strategy, "load_ff12_groups_by_symbol", lambda: {"AAA": 1}
    )
    monkeypatch.setattr(
        daily_job.strategy, "load_symbols_excluded_by_industry", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_asset_metadata", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_listing_name", lambda: set()
    )
    daily_job.update_all_data_from_yf(
        "2024-01-01", "2024-01-04", data_directory
    )

    result_frame = pandas.read_csv(csv_path, index_col=0, parse_dates=True)
    assert list(result_frame.index.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert not result_frame.index.duplicated().any()


def test_update_all_data_from_yf_treats_end_as_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``update_all_data_from_yf`` should treat the ``end_date`` as inclusive."""

    recorded_end_dates: list[str] = []

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        recorded_end_dates.append(end)
        data_frame = pandas.DataFrame(
            {"close": [1.0]}, index=pandas.to_datetime([start])
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            data_frame.to_csv(cache_path)
        return data_frame

    monkeypatch.setattr(daily_job, "download_history", fake_download_history)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["AAA"])
    monkeypatch.setattr(
        daily_job.strategy, "load_ff12_groups_by_symbol", lambda: {"AAA": 1}
    )
    monkeypatch.setattr(
        daily_job.strategy, "load_symbols_excluded_by_industry", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_asset_metadata", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_listing_name", lambda: set()
    )

    daily_job.update_all_data_from_yf("2024-01-01", "2024-01-01", tmp_path)

    assert recorded_end_dates == ["2024-01-02", "2024-01-02"]


def test_update_all_data_from_yf_preserves_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing rows should remain intact after a refresh."""

    data_directory = tmp_path
    csv_file_path = data_directory / "AAA.csv"
    csv_file_path.write_text(
        (
            "Date,open,close\n"
            "2024-01-01,1,1\n"
            "2024-01-02,1,1\n"
            "2024-01-03,1,1\n"
        ),
        encoding="utf-8",
    )

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        existing_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
        new_frame = pandas.DataFrame(
            {"open": [1, 1], "close": [1, 1]},
            index=pandas.to_datetime(["2024-01-03", "2024-01-04"]),
        )
        combined_frame = pandas.concat([existing_frame, new_frame])
        combined_frame.to_csv(cache_path)
        return combined_frame

    monkeypatch.setattr(daily_job, "download_history", fake_download_history)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["AAA"])
    monkeypatch.setattr(
        daily_job.strategy, "load_ff12_groups_by_symbol", lambda: {"AAA": 1}
    )
    monkeypatch.setattr(
        daily_job.strategy, "load_symbols_excluded_by_industry", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_asset_metadata", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_listing_name", lambda: set()
    )

    daily_job.update_all_data_from_yf(
        "2024-01-01", "2024-01-05", data_directory
    )

    result_frame = pandas.read_csv(csv_file_path, index_col=0, parse_dates=True)
    assert list(result_frame.index.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert not result_frame.index.duplicated().any()


def test_update_all_data_from_yf_logs_warning_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Errors during download should be logged and not raised."""

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        if symbol_name == "BBB":
            raise yfinance_exceptions.YFException("bad symbol")
        frame = pandas.DataFrame({"close": [1.0]}, index=pandas.to_datetime([start]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path)
        return frame

    monkeypatch.setattr(daily_job, "download_history", fake_download_history)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(
        daily_job.strategy,
        "load_ff12_groups_by_symbol",
        lambda: {"AAA": 1, "BBB": 2},
    )
    monkeypatch.setattr(
        daily_job.strategy, "load_symbols_excluded_by_industry", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_asset_metadata", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_listing_name", lambda: set()
    )

    with caplog.at_level(logging.WARNING):
        daily_job.update_all_data_from_yf(
            "2024-01-01", "2024-01-05", tmp_path
        )

    assert any("BBB" in record.message for record in caplog.records)



def test_load_runtime_download_symbols_skips_non_stock_and_missing_sector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime refresh should only include sector-classified common stocks."""

    monkeypatch.setattr(
        daily_job,
        "load_symbols",
        lambda: ["AAA", "BBB", "CCC", daily_job.SP500_SYMBOL],
    )
    monkeypatch.setattr(
        daily_job.strategy,
        "load_ff12_groups_by_symbol",
        lambda: {"AAA": 1, "CCC": 6},
    )
    monkeypatch.setattr(
        daily_job.strategy,
        "load_symbols_excluded_by_industry",
        lambda: {"CCC"},
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_asset_metadata", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_listing_name", lambda: set()
    )

    assert daily_job.load_runtime_download_symbols() == [
        "AAA",
        daily_job.SP500_SYMBOL,
    ]


def test_load_runtime_download_symbols_skips_asset_and_listing_name_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime refresh should remove known ETFs, warrants and units."""

    monkeypatch.setattr(
        daily_job,
        "load_symbols",
        lambda: ["AAA", "ETF", "UNKNOWN", "WARRANT"],
    )
    monkeypatch.setattr(
        daily_job.strategy,
        "load_ff12_groups_by_symbol",
        lambda: {"AAA": 1, "ETF": 1, "UNKNOWN": 1, "WARRANT": 1},
    )
    monkeypatch.setattr(
        daily_job.strategy, "load_symbols_excluded_by_industry", lambda: set()
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_asset_metadata", lambda: {"ETF"}
    )
    monkeypatch.setattr(
        daily_job, "load_symbols_rejected_by_listing_name", lambda: {"WARRANT"}
    )

    assert daily_job.load_runtime_download_symbols() == [
        "AAA",
        "UNKNOWN",
        daily_job.SP500_SYMBOL,
    ]
