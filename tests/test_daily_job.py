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

    assert recorded_end_dates
    assert set(recorded_end_dates) == {"2024-01-02"}


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


def test_find_history_signal_uses_last_available_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symbols with data before ``date_string`` should still be evaluated."""

    csv_file_path = tmp_path / "KO.csv"
    csv_file_path.write_text(
        "Date,open,close\n2025-10-08,60,61\n", encoding="utf-8"
    )

    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", tmp_path)

    def fake_parse_daily_task_arguments(argument_line: str) -> tuple[
        float | None,
        int | None,
        int,
        str,
        str,
        float,
        set[int] | None,
    ]:
        return (None, None, 1, "fake_buy", "fake_sell", 1.0, None)

    monkeypatch.setattr(
        daily_job, "parse_daily_task_arguments", fake_parse_daily_task_arguments
    )

    captured_symbol_list: dict[str, list[str] | None] = {"values": None}

    def fake_run_daily_tasks(
        *,
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        symbol_list: list[str] | None,
        **_: object,
    ) -> dict[str, list[str] | list[tuple[str, int | None]]]:
        captured_symbol_list["values"] = list(symbol_list) if symbol_list else []
        return {
            "filtered_symbols": []
            if not symbol_list
            else [(symbol_list[0], None)],
            "entry_signals": [] if not symbol_list else list(symbol_list),
            "exit_signals": [],
        }

    monkeypatch.setattr(daily_job, "run_daily_tasks", fake_run_daily_tasks)

    result = daily_job.find_history_signal(
        "2025-10-09", "filter", "fake_buy", "fake_sell", 1.0
    )

    assert captured_symbol_list["values"] == ["KO"]
    assert result["entry_signals"] == ["KO"]


def test_filter_debug_values_returns_last_available_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``filter_debug_values`` should sample the most recent available bar."""

    csv_file_path = tmp_path / "KO.csv"
    csv_file_path.write_text(
        (
            "Date,open,close\n"
            "2025-10-07,10,10\n"
            "2025-10-08,11,11\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", tmp_path)

    def fake_parse_strategy_name(strategy_name: str) -> tuple[
        str, int | None, tuple[float, float] | None, tuple[float, float] | None, tuple[float, float] | None
    ]:
        return (strategy_name, None, None, None, None)

    monkeypatch.setattr(daily_job.strategy, "parse_strategy_name", fake_parse_strategy_name)

    def fake_buy_strategy(frame: pandas.DataFrame, **_: object) -> None:
        frame["sma_angle"] = pandas.Series([0.1, 0.2], index=frame.index)
        frame["near_price_volume_ratio"] = pandas.Series([0.3, 0.4], index=frame.index)
        frame["above_price_volume_ratio"] = pandas.Series([0.5, 0.6], index=frame.index)
        frame["fake_buy_entry_signal"] = pandas.Series([False, True], index=frame.index)

    def fake_sell_strategy(frame: pandas.DataFrame, **_: object) -> None:
        frame["fake_sell_exit_signal"] = pandas.Series([False, True], index=frame.index)

    monkeypatch.setitem(daily_job.strategy.BUY_STRATEGIES, "fake_buy", fake_buy_strategy)
    monkeypatch.setitem(daily_job.strategy.SELL_STRATEGIES, "fake_sell", fake_sell_strategy)

    result = daily_job.filter_debug_values("KO", "2025-10-09", "fake_buy", "fake_sell")

    assert result["sma_angle"] == pytest.approx(0.2)
    assert result["near_price_volume_ratio"] == pytest.approx(0.4)
    assert result["above_price_volume_ratio"] == pytest.approx(0.6)
    assert result["entry"] is True
    assert result["exit"] is True

