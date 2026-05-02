"""Tests for historical signal helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas
import pytest

# Ensure the src directory is importable without package installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from stock_indicator import daily_job, strategy


@pytest.fixture
def temporary_data_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a temporary directory patched as the stock data location."""

    data_directory = tmp_path / "stock_data"
    data_directory.mkdir()
    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(strategy, "DOLLAR_VOLUME_SMA_WINDOW", 1)
    return data_directory


def _register_test_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register lightweight buy and sell strategies for testing."""

    def fake_buy_strategy(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        index = price_frame.index
        entry_series = pandas.Series([False, False, True], index=index)
        price_frame["test_strategy_entry_signal"] = entry_series
        if include_raw_signals:
            raw_series = pandas.Series([False, False, True], index=index)
            price_frame["test_strategy_raw_entry_signal"] = raw_series
        price_frame["sma_angle"] = pandas.Series([0.0, 0.0, 0.0], index=index)
        price_frame["near_price_volume_ratio"] = pandas.Series(
            [0.5, 0.5, 0.5], index=index
        )
        price_frame["above_price_volume_ratio"] = pandas.Series(
            [0.2, 0.2, 0.2], index=index
        )

    def fake_sell_strategy(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        index = price_frame.index
        exit_series = pandas.Series([False, False, False], index=index)
        price_frame["test_strategy_exit_signal"] = exit_series
        if include_raw_signals:
            raw_exit_series = pandas.Series([False, False, False], index=index)
            price_frame["test_strategy_raw_exit_signal"] = raw_exit_series

    monkeypatch.setitem(strategy.BUY_STRATEGIES, "test_strategy", fake_buy_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "test_strategy", fake_sell_strategy)


def test_find_history_signal_includes_shifted_entries(
    temporary_data_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signals should include entries when only shifted columns fire."""

    _register_test_strategy(monkeypatch)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["KO"])
    csv_path = temporary_data_directory / "KO.csv"
    frame = pandas.DataFrame(
        {
            "Date": pandas.to_datetime(
                ["2025-10-08", "2025-10-09", "2025-10-10"]
            ),
            "Open": [10.0, 10.0, 10.0],
            "High": [11.0, 11.0, 11.0],
            "Low": [9.0, 9.0, 9.0],
            "Close": [10.0, 10.0, 10.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000],
        }
    )
    frame.to_csv(csv_path, index=False)

    result = daily_job.find_history_signal(
        "2025-10-10",
        "dollar_volume>0",
        "test_strategy",
        "test_strategy",
        1.0,
    )

    assert "KO" in result.get("entry_signals", [])


def test_find_history_signal_ignores_stale_cached_symbols(
    temporary_data_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cron signal checks should use the current symbol list, not stale CSVs."""

    _register_test_strategy(monkeypatch)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["KO"])
    frame = pandas.DataFrame(
        {
            "Date": pandas.to_datetime(
                ["2025-10-08", "2025-10-09", "2025-10-10"]
            ),
            "Open": [10.0, 10.0, 10.0],
            "High": [11.0, 11.0, 11.0],
            "Low": [9.0, 9.0, 9.0],
            "Close": [10.0, 10.0, 10.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000],
        }
    )
    frame.to_csv(temporary_data_directory / "KO.csv", index=False)
    frame.to_csv(temporary_data_directory / "STALE.csv", index=False)

    result = daily_job.find_history_signal(
        "2025-10-10",
        "dollar_volume>0",
        "test_strategy",
        "test_strategy",
        1.0,
    )

    assert "KO" in result.get("entry_signals", [])
    assert "STALE" not in result.get("entry_signals", [])
    assert {
        symbol_name for symbol_name, _ in result.get("filtered_symbols", [])
    } == {"KO"}


def test_filter_debug_values_uses_latest_available_row(
    temporary_data_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Debug values should fall back to the latest row on or before the date."""

    def buy_strategy_with_metrics(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        index = price_frame.index
        length = len(index)
        price_frame["test_strategy_entry_signal"] = pandas.Series(
            [False] * length, index=index
        )
        price_frame["sma_angle"] = pandas.Series([1.0] * length, index=index)
        price_frame["near_price_volume_ratio"] = pandas.Series(
            [0.11] * length, index=index
        )
        price_frame["above_price_volume_ratio"] = pandas.Series(
            [0.33] * length, index=index
        )
        if include_raw_signals:
            price_frame["test_strategy_raw_entry_signal"] = pandas.Series(
                [False] * length, index=index
            )

    def sell_strategy_placeholder(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        length = len(price_frame.index)
        price_frame["test_strategy_exit_signal"] = pandas.Series(
            [False] * length, index=price_frame.index
        )
        if include_raw_signals:
            price_frame["test_strategy_raw_exit_signal"] = pandas.Series(
                [False] * length, index=price_frame.index
            )

    monkeypatch.setitem(strategy.BUY_STRATEGIES, "test_strategy", buy_strategy_with_metrics)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "test_strategy", sell_strategy_placeholder)

    csv_path = temporary_data_directory / "KO.csv"
    frame = pandas.DataFrame(
        {
            "Date": pandas.to_datetime(["2025-10-08", "2025-10-10"]),
            "Open": [12.0, 13.0],
            "High": [12.5, 13.5],
            "Low": [11.5, 12.5],
            "Close": [12.0, 13.0],
            "Volume": [2_000_000, 2_000_000],
        }
    )
    frame.to_csv(csv_path, index=False)

    debug_values = daily_job.filter_debug_values(
        "KO", "2025-10-09", "test_strategy", "test_strategy"
    )

    assert debug_values["sma_angle"] == pytest.approx(1.0)
    assert debug_values["sma_angle_previous"] is None
    assert debug_values["near_price_volume_ratio"] == pytest.approx(0.11)
    assert debug_values["near_price_volume_ratio_previous"] is None
    assert debug_values["above_price_volume_ratio"] == pytest.approx(0.33)
    assert debug_values["above_price_volume_ratio_previous"] is None
    assert debug_values["entry"] is False
    assert debug_values["exit"] is False


def test_filter_debug_values_reports_raw_entry_signals(
    temporary_data_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw entry signals should be reflected in the debug output."""

    def buy_strategy_with_raw_signals(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        index = price_frame.index
        length = len(index)
        price_frame["test_strategy_entry_signal"] = pandas.Series(
            [False] * length, index=index
        )
        price_frame["sma_angle"] = pandas.Series([2.0] * length, index=index)
        price_frame["near_price_volume_ratio"] = pandas.Series(
            [0.2] * length, index=index
        )
        price_frame["above_price_volume_ratio"] = pandas.Series(
            [0.4] * length, index=index
        )
        if include_raw_signals:
            price_frame["test_strategy_raw_entry_signal"] = pandas.Series(
                [False, False, True], index=index
            )

    def sell_strategy_without_signals(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        index = price_frame.index
        length = len(index)
        price_frame["test_strategy_exit_signal"] = pandas.Series(
            [False] * length, index=index
        )
        if include_raw_signals:
            price_frame["test_strategy_raw_exit_signal"] = pandas.Series(
                [False] * length, index=index
            )

    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "test_strategy", buy_strategy_with_raw_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "test_strategy", sell_strategy_without_signals
    )

    csv_path = temporary_data_directory / "KO.csv"
    frame = pandas.DataFrame(
        {
            "Date": pandas.to_datetime(
                ["2025-10-08", "2025-10-09", "2025-10-10"]
            ),
            "Open": [12.0, 12.5, 13.0],
            "High": [12.5, 13.0, 13.5],
            "Low": [11.5, 12.0, 12.5],
            "Close": [12.0, 12.5, 13.0],
            "Volume": [1_500_000, 1_500_000, 1_500_000],
        }
    )
    frame.to_csv(csv_path, index=False)

    debug_values = daily_job.filter_debug_values(
        "KO", "2025-10-10", "test_strategy", "test_strategy"
    )

    assert debug_values["entry"] is True
    assert debug_values["exit"] is False
    assert debug_values["sma_angle_previous"] == pytest.approx(2.0)
    assert debug_values["near_price_volume_ratio_previous"] is None
    assert debug_values["above_price_volume_ratio_previous"] is None


def test_filter_debug_values_includes_shifted_entry_signals(
    temporary_data_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shifted entry signals should remain visible for the evaluation date."""

    def buy_strategy_with_shifted_signal(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        index = price_frame.index
        price_frame["test_strategy_entry_signal"] = pandas.Series(
            [False, False, True], index=index
        )
        price_frame["sma_angle"] = pandas.Series([3.0, 4.0, 5.0], index=index)
        price_frame["near_price_volume_ratio"] = pandas.Series(
            [0.25, 0.35, 0.45], index=index
        )
        price_frame["above_price_volume_ratio"] = pandas.Series(
            [0.55, 0.65, 0.75], index=index
        )
        if include_raw_signals:
            price_frame["test_strategy_raw_entry_signal"] = pandas.Series(
                [False, False, False], index=index
            )

    def sell_strategy_placeholder(
        price_frame: pandas.DataFrame,
        *,
        include_raw_signals: bool = False,
        **_: float,
    ) -> None:
        index = price_frame.index
        length = len(index)
        price_frame["test_strategy_exit_signal"] = pandas.Series(
            [False] * length, index=index
        )
        if include_raw_signals:
            price_frame["test_strategy_raw_exit_signal"] = pandas.Series(
                [False] * length, index=index
            )

    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "test_strategy", buy_strategy_with_shifted_signal
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "test_strategy", sell_strategy_placeholder
    )

    csv_path = temporary_data_directory / "KO.csv"
    frame = pandas.DataFrame(
        {
            "Date": pandas.to_datetime(
                ["2025-10-08", "2025-10-09", "2025-10-10"]
            ),
            "Open": [12.0, 12.5, 13.0],
            "High": [12.5, 13.0, 13.5],
            "Low": [11.5, 12.0, 12.5],
            "Close": [12.0, 12.5, 13.0],
            "Volume": [1_800_000, 1_800_000, 1_800_000],
        }
    )
    frame.to_csv(csv_path, index=False)

    debug_values = daily_job.filter_debug_values(
        "KO", "2025-10-09", "test_strategy", "test_strategy"
    )

    assert debug_values["entry"] is True
    assert debug_values["exit"] is False
    assert debug_values["sma_angle"] == pytest.approx(4.0)
    assert debug_values["sma_angle_previous"] == pytest.approx(3.0)
    assert debug_values["near_price_volume_ratio"] == pytest.approx(0.35)
    assert debug_values["above_price_volume_ratio"] == pytest.approx(0.65)
