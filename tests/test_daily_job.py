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

    daily_job.update_all_data_from_yf("2024-01-01", "2024-01-01", tmp_path)

    assert recorded_end_dates == ["2024-01-02"]


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

    with caplog.at_level(logging.WARNING):
        daily_job.update_all_data_from_yf(
            "2024-01-01", "2024-01-05", tmp_path
        )

    assert any("BBB" in record.message for record in caplog.records)


def test_find_history_signal_uses_shifted_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``find_history_signal`` should evaluate signals on the trading day."""

    csv_path = tmp_path / "AAA.csv"
    csv_path.write_text(
        "Date,open,close,volume\n2024-01-10,1,1,1000000\n",
        encoding="utf-8",
    )

    recorded_arguments: dict[str, object] = {}

    def fake_parse_arguments(argument_line: str) -> tuple[
        float | None,
        int | None,
        int,
        str,
        str,
        float,
        set[int] | None,
    ]:
        recorded_arguments["argument_line"] = argument_line
        return (None, None, 1, "buy", "sell", 1.0, None)

    def fake_run_daily_tasks(
        *,
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        symbol_list: list[str] | None,
        data_download_function,
        data_directory: Path,
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        allowed_fama_french_groups,
        maximum_symbols_per_group: int,
        use_unshifted_signals: bool = False,
        **_: object,
    ) -> dict[str, list[str] | list[tuple[str, int | None]]]:
        recorded_arguments["start_date"] = start_date
        recorded_arguments["end_date"] = end_date
        recorded_arguments["use_unshifted_signals"] = use_unshifted_signals
        assert buy_strategy_name == "buy"
        assert sell_strategy_name == "sell"
        return {
            "filtered_symbols": [("AAA", None)],
            "entry_signals": ["AAA"],
            "exit_signals": [],
        }

    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", tmp_path)
    monkeypatch.setattr(daily_job, "determine_start_date", lambda _path: "2024-01-01")
    monkeypatch.setattr(daily_job, "parse_daily_task_arguments", fake_parse_arguments)
    monkeypatch.setattr(daily_job, "run_daily_tasks", fake_run_daily_tasks)

    result = daily_job.find_history_signal(
        "2024-01-10", "dollar_volume>1", "buy", "sell", 1.0
    )

    assert result["entry_signals"] == ["AAA"]
    assert recorded_arguments["use_unshifted_signals"] is False
    assert recorded_arguments["end_date"] == "2024-01-11"
    assert recorded_arguments["start_date"] == "2024-01-01"

