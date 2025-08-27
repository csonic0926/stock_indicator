"""Tests for strategy evaluation utilities."""
# TODO: review

import os
import sys
from pathlib import Path
from typing import Iterable

import pandas
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import stock_indicator.strategy as strategy
from stock_indicator.simulator import SimulationResult, Trade, calc_commission

from stock_indicator.strategy import (
    evaluate_ema_sma_cross_strategy,
    evaluate_kalman_channel_strategy,
    evaluate_combined_strategy,
    parse_strategy_name,
)


def test_evaluate_ema_sma_cross_strategy_computes_win_rate(tmp_path: Path) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
    ]
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "test.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_normalizes_headers(tmp_path: Path) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
    ]
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "OPEN": price_values,
            "CLOSE": price_values,
        }
    )
    csv_path = tmp_path / "test_uppercase.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_removes_ticker_suffix(tmp_path: Path) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
    ]
    price_value_list = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "Open RIV": price_value_list,
            "Close RIV": price_value_list,
        }
    )
    csv_path = tmp_path / "ticker_suffix.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_strips_leading_underscore(
    tmp_path: Path,
) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
    ]
    price_value_list = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "_Open RIV": price_value_list,
            "_Close RIV": price_value_list,
        }
    )
    csv_path = tmp_path / "leading_underscore.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_raises_value_error_for_missing_columns(
    tmp_path: Path,
) -> None:
    price_values = [10.0, 10.0, 10.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "Opening Price": price_values, "Closing Price": price_values}
    )
    csv_path = tmp_path / "test_missing.csv"
    price_data_frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)


def test_evaluate_ema_sma_cross_strategy_handles_multiindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_price_values = [10.0] * 150
    pattern_price_values = [
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
    ]
    price_value_list = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_value_list), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            ("Date", ""): date_index,
            ("OPEN", "ignore"): price_value_list,
            ("CLOSE", "ignore"): price_value_list,
        }
    )
    csv_path = tmp_path / "multi_index.csv"
    price_data_frame.to_csv(csv_path, index=False)

    original_read_csv = pandas.read_csv

    def patched_read_csv(*args: object, **kwargs: object) -> pandas.DataFrame:
        return original_read_csv(
            *args,
            header=[0, 1],
            index_col=0,
            parse_dates=[0],
            **{key: value for key, value in kwargs.items() if key not in {"header", "index_col", "parse_dates"}},
        )

    monkeypatch.setattr(pandas, "read_csv", patched_read_csv)
    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_requires_close_above_long_term_sma(
    tmp_path: Path,
) -> None:
    initial_price_values = [20.0] * 150
    pattern_price_values = [
        20.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
    ]
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "below_long_sma.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 0
    assert result.win_rate == 0.0


def test_evaluate_ema_sma_cross_strategy_ignores_missing_relative_strength(
    tmp_path: Path,
) -> None:
    price_values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "missing_rs.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert (result.total_trades, result.win_rate) == (0, 0.0)
    updated_data_frame = pandas.read_csv(csv_path)
    assert "rs" not in updated_data_frame.columns


def test_evaluate_ema_sma_cross_strategy_computes_profit_and_loss_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_values = [10.0] * 160
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "statistics.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trades = [
        Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=12.0,
            profit=2.0,
            holding_period=1,
        ),
        Trade(
            entry_date=date_index[2],
            exit_date=date_index[3],
            entry_price=10.0,
            exit_price=9.0,
            profit=-1.0,
            holding_period=1,
        ),
    ]
    simulation_result = SimulationResult(trades=trades, total_profit=1.0)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_ema_sma_cross_strategy(tmp_path, window_size=3)

    assert result.total_trades == 2
    assert result.win_rate == 0.5
    assert result.mean_profit_percentage == pytest.approx(0.2)
    assert result.profit_percentage_standard_deviation == 0.0
    assert result.mean_loss_percentage == pytest.approx(0.1)
    assert result.loss_percentage_standard_deviation == 0.0
    assert result.mean_holding_period == pytest.approx(1.0)
    assert result.holding_period_standard_deviation == 0.0


def test_evaluate_kalman_channel_strategy_generates_trade(tmp_path: Path) -> None:
    initial_price_values = [20.0] * 20
    pattern_price_values = [
        10.0,
        20.0,
        20.0,
        20.0,
        10.0,
        10.0,
    ]
    price_values = initial_price_values + pattern_price_values
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "kalman.csv"
    price_data_frame.to_csv(csv_path, index=False)

    result = evaluate_kalman_channel_strategy(tmp_path)

    assert result.total_trades == 1
    assert result.win_rate == 0.0


def test_evaluate_combined_strategy_different_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should aggregate results for mixed strategies."""
    price_values = [10.0] * 160
    date_index = pandas.date_range("2020-01-01", periods=len(price_values), freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
        }
    )
    csv_path = tmp_path / "combined.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trades = [
        Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=12.0,
            profit=2.0,
            holding_period=1,
        ),
        Trade(
            entry_date=date_index[2],
            exit_date=date_index[3],
            entry_price=10.0,
            exit_price=9.0,
            profit=-1.0,
            holding_period=1,
        ),
    ]
    simulation_result = SimulationResult(trades=trades, total_profit=1.0)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_combined_strategy(
        tmp_path, "ema_sma_cross", "kalman_filtering"
    )

    assert result.total_trades == 2
    assert result.win_rate == 0.5


def test_evaluate_combined_strategy_calculates_compound_annual_growth_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should compute CAGR based on final balance and duration."""

    date_index = pandas.date_range("2020-01-01", periods=370, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0] * 370, "close": [10.0] * 370}
    )
    csv_path = tmp_path / "cagr_test.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trade = Trade(
        entry_date=date_index[0],
        exit_date=date_index[365],
        entry_price=10.0,
        exit_price=11.0,
        profit=1.0,
        holding_period=365,
    )
    simulation_result = SimulationResult(trades=[trade], total_profit=1.0)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    def fake_simulate_portfolio_balance(*args: object, **kwargs: object) -> float:
        return 3300.0

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(
        strategy, "simulate_portfolio_balance", fake_simulate_portfolio_balance
    )
    monkeypatch.setattr(strategy, "calculate_annual_returns", lambda *a, **k: {})
    monkeypatch.setattr(strategy, "calculate_annual_trade_counts", lambda *a, **k: {})

    result = evaluate_combined_strategy(tmp_path, "ema_sma_cross", "ema_sma_cross")

    duration_years = (trade.exit_date - trade.entry_date).days / 365.25
    expected_growth_rate = (3300.0 / 3000.0) ** (1 / duration_years) - 1
    assert result.compound_annual_growth_rate == pytest.approx(
        expected_growth_rate
    )


def test_evaluate_combined_strategy_reports_max_drawdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should report the largest portfolio decline."""

    date_index = pandas.date_range("2020-01-01", periods=200, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0] * 200, "close": [10.0] * 200}
    )
    csv_path = tmp_path / "AAA.csv"
    price_data_frame.to_csv(csv_path, index=False)

    trades = [
        Trade(
            entry_date=pandas.Timestamp("2020-01-01"),
            exit_date=pandas.Timestamp("2020-01-02"),
            entry_price=10.0,
            exit_price=20.0,
            profit=10.0,
            holding_period=1,
        ),
        Trade(
            entry_date=pandas.Timestamp("2020-01-03"),
            exit_date=pandas.Timestamp("2020-01-04"),
            entry_price=20.0,
            exit_price=10.0,
            profit=-10.0,
            holding_period=1,
        ),
    ]
    simulation_result = SimulationResult(trades=trades, total_profit=0.0)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return simulation_result

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy, "calculate_annual_returns", lambda *a, **k: {})
    monkeypatch.setattr(strategy, "calculate_annual_trade_counts", lambda *a, **k: {})
    monkeypatch.setattr(strategy, "simulate_portfolio_balance", lambda *a, **k: 1008.0)

    result = evaluate_combined_strategy(
        tmp_path, "ema_sma_cross", "ema_sma_cross", starting_cash=1000.0
    )

    peak_value = (
        1000.0 * (20.0 / 10.0)
        - calc_commission(100, 10.0)
        - calc_commission(100, 20.0)
    )
    final_value = (
        peak_value
        - 99 * 20.0
        - calc_commission(99, 20.0)
        + 99 * 10.0
        - calc_commission(99, 10.0)
    )
    expected_drawdown = (peak_value - final_value) / peak_value
    assert result.maximum_drawdown == pytest.approx(expected_drawdown)


def test_evaluate_combined_strategy_unsupported_name(tmp_path: Path) -> None:
    """evaluate_combined_strategy should raise for unknown strategies."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        evaluate_combined_strategy(tmp_path, "unknown", "ema_sma_cross")


def test_evaluate_combined_strategy_rejects_sell_only_buy(tmp_path: Path) -> None:
    """evaluate_combined_strategy should reject sell-only strategies used for buying."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        evaluate_combined_strategy(tmp_path, "kalman_filtering", "ema_sma_cross")


def test_evaluate_combined_strategy_passes_window_size_and_renames_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should pass the window size and append it to signal names."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "window.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, int | None] = {"window_size": None}
    captured_column_names: list[str] = []

    def fake_attach_signals(frame: pandas.DataFrame, window_size: int = 50) -> None:
        captured_arguments["window_size"] = window_size
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        captured_column_names.extend(passed_frame.columns)
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    try:
        evaluate_combined_strategy(
            tmp_path, "ema_sma_cross_with_slope_40", "ema_sma_cross_with_slope_40"
        )
    except ValueError as error:
        pytest.fail(f"Unexpected ValueError: {error}")

    assert captured_arguments["window_size"] == 40
    assert "ema_sma_cross_with_slope_40_entry_signal" in captured_column_names
    assert "ema_sma_cross_with_slope_40_exit_signal" in captured_column_names


def test_evaluate_combined_strategy_passes_slope_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should pass the slope range to the strategy function."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1.0, 1.0],
        }
    )
    csv_path = tmp_path / "slope.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, tuple[float, float] | None] = {
        "slope_range": None
    }

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        slope_range: tuple[float, float] = (-0.3, 1.0),
    ) -> None:
        captured_arguments["slope_range"] = slope_range
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_-0.5_0.5",
        "ema_sma_cross_with_slope_-0.5_0.5",
    )

    assert captured_arguments["slope_range"] == (-0.5, 0.5)


def test_evaluate_combined_strategy_passes_slope_range_with_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should pass slope range for strategies using volume."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1.0, 1.0],
        }
    )
    csv_path = tmp_path / "slope_volume.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, tuple[float, float] | None] = {
        "slope_range": None
    }

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        slope_range: tuple[float, float] = (-0.3, 1.0),
    ) -> None:
        captured_arguments["slope_range"] = slope_range
        frame["ema_sma_cross_with_slope_and_volume_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_and_volume_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy,
        "attach_ema_sma_cross_with_slope_and_volume_signals",
        fake_attach_signals,
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES,
        "ema_sma_cross_with_slope_and_volume",
        fake_attach_signals,
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES,
        "ema_sma_cross_with_slope_and_volume",
        fake_attach_signals,
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_and_volume_-0.5_0.5",
        "ema_sma_cross_with_slope_and_volume_-0.5_0.5",
    )

    assert captured_arguments["slope_range"] == (-0.5, 0.5)


def test_evaluate_combined_strategy_renames_columns_with_slope_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signal column names should include the slope range suffix."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "slope_rename.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_column_names: list[str] = []

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        slope_range: tuple[float, float] = (-0.3, 1.0),
    ) -> None:
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        captured_column_names.extend(passed_frame.columns)
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_-0.5_0.5",
        "ema_sma_cross_with_slope_-0.5_0.5",
    )

    assert (
        "ema_sma_cross_with_slope_-0.5_0.5_entry_signal" in captured_column_names
    )
    assert (
        "ema_sma_cross_with_slope_-0.5_0.5_exit_signal" in captured_column_names
    )


def test_evaluate_combined_strategy_renames_columns_negative_positive_slope_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should include negative-to-positive slope values in names."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "slope_negative_positive.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_column_names: list[str] = []

    def fake_attach_signals(
        frame: pandas.DataFrame,
        window_size: int = 40,
        slope_range: tuple[float, float] = (-0.3, 1.0),
    ) -> None:
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        captured_column_names.extend(passed_frame.columns)
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=10.0,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_-0.1_1.2",
        "ema_sma_cross_with_slope_-0.1_1.2",
    )

    assert (
        "ema_sma_cross_with_slope_-0.1_1.2_entry_signal" in captured_column_names
    )
    assert (
        "ema_sma_cross_with_slope_-0.1_1.2_exit_signal" in captured_column_names
    )

def test_evaluate_combined_strategy_dollar_volume_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should skip symbols below the dollar volume threshold."""

    price_values = [10.0] * 60
    volume_values = [1_000_000] * 60
    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": price_values,
            "close": price_values,
            "volume": volume_values,
        }
    )
    csv_path = tmp_path / "filtered.csv"
    price_data_frame.to_csv(csv_path, index=False)

    simulate_called: dict[str, bool] = {"called": False}

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        simulate_called["called"] = True
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)


def test_evaluate_combined_strategy_dollar_volume_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should skip symbols below the dollar volume ratio."""

    volumes_by_symbol = {"AAA": 100_000_000, "BBB": 900_000_000}
    for symbol_name, volume_value in volumes_by_symbol.items():
        date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
        pandas.DataFrame(
            {
                "Date": date_index,
                "open": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": [volume_value] * 60,
                "symbol": [symbol_name] * 60,
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    processed_symbols: list[str] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbols.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume_ratio=0.2,
    )
    assert processed_symbols == ["BBB"]

    processed_symbols.clear()
    evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume_ratio=0.05,
    )
    assert set(processed_symbols) == {"AAA", "BBB"}


def test_evaluate_combined_strategy_trade_details_use_latest_average_dollar_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TradeDetail should store date-specific 50-day average dollar volumes."""

    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0] * 60,
            "close": [10.0] * 60,
            "volume": [1_000_000] * 59 + [2_000_000],
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [20.0] * 60,
            "close": [20.0] * 60,
            "volume": [2_000_000] * 59 + [4_000_000],
        }
    ).to_csv(tmp_path / "BBB.csv", index=False)

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(
        strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda frame: None}
    )

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        price_data_frame: pandas.DataFrame = kwargs["data"]
        entry_date = price_data_frame.index[58]
        exit_date = price_data_frame.index[59]
        entry_price = float(price_data_frame.iloc[58]["open"])
        exit_price = float(price_data_frame.iloc[59]["open"])
        trade = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            profit=0.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)

    result = strategy_module.evaluate_combined_strategy(tmp_path, "noop", "noop")
    trade_details = result.trade_details_by_year[2020]
    open_details = {
        detail.symbol: detail for detail in trade_details if detail.action == "open"
    }
    close_details = {
        detail.symbol: detail for detail in trade_details if detail.action == "close"
    }
    aaa_open = open_details["AAA"]
    bbb_open = open_details["BBB"]
    aaa_close = close_details["AAA"]
    bbb_close = close_details["BBB"]

    assert aaa_open.simple_moving_average_dollar_volume == pytest.approx(10_000_000.0)
    assert bbb_open.simple_moving_average_dollar_volume == pytest.approx(40_000_000.0)
    assert (
        aaa_open.total_simple_moving_average_dollar_volume
        == pytest.approx(50_000_000.0)
    )
    assert (
        bbb_open.total_simple_moving_average_dollar_volume
        == pytest.approx(50_000_000.0)
    )
    assert aaa_open.simple_moving_average_dollar_volume_ratio == pytest.approx(0.2)
    assert bbb_open.simple_moving_average_dollar_volume_ratio == pytest.approx(0.8)

    assert aaa_close.simple_moving_average_dollar_volume == pytest.approx(10_200_000.0)
    assert bbb_close.simple_moving_average_dollar_volume == pytest.approx(40_800_000.0)
    assert (
        aaa_close.total_simple_moving_average_dollar_volume
        == pytest.approx(51_000_000.0)
    )
    assert (
        bbb_close.total_simple_moving_average_dollar_volume
        == pytest.approx(51_000_000.0)
    )
    assert aaa_close.simple_moving_average_dollar_volume_ratio == pytest.approx(0.2)
    assert bbb_close.simple_moving_average_dollar_volume_ratio == pytest.approx(0.8)


def test_evaluate_combined_strategy_handles_empty_csv(tmp_path: Path) -> None:
    """evaluate_combined_strategy should skip empty CSV files and return zero trades."""
    empty_data_frame = pandas.DataFrame(
        columns=["Date", "open", "close", "volume"]
    )
    csv_file_path = tmp_path / "empty.csv"
    empty_data_frame.to_csv(csv_file_path, index=False)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
    )

    assert result.total_trades == 0


def test_evaluate_combined_strategy_reports_maximum_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should return the highest number of overlapping trades."""
    import stock_indicator.strategy as strategy_module
    from stock_indicator.simulator import SimulationResult, Trade

    for symbol_name in ["AAA", "BBB"]:
        pandas.DataFrame(
            {
                "Date": ["2020-01-01"],
                "open": [1.0],
                "close": [1.0],
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    simulation_results = [
        SimulationResult(
            trades=[
                Trade(
                    entry_date=pandas.Timestamp("2020-01-01"),
                    exit_date=pandas.Timestamp("2020-01-03"),
                    entry_price=1.0,
                    exit_price=1.0,
                    profit=0.0,
                    holding_period=2,
                )
            ],
            total_profit=0.0,
        ),
        SimulationResult(
            trades=[
                Trade(
                    entry_date=pandas.Timestamp("2020-01-02"),
                    exit_date=pandas.Timestamp("2020-01-04"),
                    entry_price=1.0,
                    exit_price=1.0,
                    profit=0.0,
                    holding_period=2,
                )
            ],
            total_profit=0.0,
        ),
    ]
    simulation_iterator = iter(simulation_results)

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        return next(simulation_iterator)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)

    result = strategy_module.evaluate_combined_strategy(tmp_path, "noop", "noop")
    assert result.maximum_concurrent_positions == 2


def test_evaluate_combined_strategy_skips_sp500_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should ignore the S&P 500 index symbol."""

    import stock_indicator.strategy as strategy_module
    from stock_indicator.symbols import SP500_SYMBOL

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [1.0] * 60,
            "close": [1.0] * 60,
            "symbol": ["AAA"] * 60,
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [1.0] * 60,
            "close": [1.0] * 60,
            "symbol": [SP500_SYMBOL] * 60,
        }
    ).to_csv(tmp_path / f"{SP500_SYMBOL}.csv", index=False)

    processed_symbol_names: list[str] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbol_names.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda frame: None})

    strategy_module.evaluate_combined_strategy(tmp_path, "noop", "noop")
    assert processed_symbol_names == ["AAA"]


def test_evaluate_combined_strategy_ignores_trades_before_start_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_combined_strategy should ignore trades before the start date."""

    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2020-01-01", periods=5, freq="D")
    pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0, 10.0, 10.0, 10.0, 12.0],
            "close": [10.0, 10.0, 10.0, 10.0, 12.0],
            "volume": [1_000_000] * 5,
        }
    ).to_csv(tmp_path / "AAA.csv", index=False)

    def fake_strategy(frame: pandas.DataFrame) -> None:
        frame["test_entry_signal"] = False
        frame["test_exit_signal"] = False
        if pandas.Timestamp("2020-01-01") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-01"), "test_entry_signal"] = True
        if pandas.Timestamp("2020-01-02") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-02"), "test_exit_signal"] = True
        if pandas.Timestamp("2020-01-04") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-04"), "test_entry_signal"] = True
        if pandas.Timestamp("2020-01-05") in frame.index:
            frame.loc[pandas.Timestamp("2020-01-05"), "test_exit_signal"] = True

    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"test": fake_strategy})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"test": fake_strategy})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"test": fake_strategy})

    result_without_filter = strategy_module.evaluate_combined_strategy(
        tmp_path, "test", "test"
    )
    result_with_filter = strategy_module.evaluate_combined_strategy(
        tmp_path,
        "test",
        "test",
        start_date=pandas.Timestamp("2020-01-03"),
    )

    assert result_without_filter.total_trades == 2
    assert result_with_filter.total_trades == 1
    for trade_details in result_with_filter.trade_details_by_year.values():
        for detail in trade_details:
            assert detail.date >= pandas.Timestamp("2020-01-03")


def test_attach_20_50_sma_cross_signals_mark_crosses() -> None:
    """The 20/50 SMA cross signals should mark upward and downward crosses."""
    # TODO: review

    import stock_indicator.strategy as strategy_module
    from stock_indicator.indicators import sma

    close_values = (
        [100.0] * 50
        + list(range(100, 50, -1))
        + list(range(50, 200))
        + list(range(200, 50, -1))
    )
    price_data_frame = pandas.DataFrame({"close": close_values})

    strategy_module.attach_20_50_sma_cross_signals(price_data_frame)

    sma_20_series = sma(price_data_frame["close"], 20)
    sma_50_series = sma(price_data_frame["close"], 50)
    expected_entry_series = (
        (sma_20_series.shift(1) <= sma_50_series.shift(1))
        & (sma_20_series > sma_50_series)
    ).shift(1, fill_value=False)
    expected_exit_series = (
        (sma_20_series.shift(1) >= sma_50_series.shift(1))
        & (sma_20_series < sma_50_series)
    ).shift(1, fill_value=False)

    assert price_data_frame["20_50_sma_cross_entry_signal"].equals(expected_entry_series)
    assert price_data_frame["20_50_sma_cross_exit_signal"].equals(expected_exit_series)
    assert price_data_frame["20_50_sma_cross_entry_signal"].any()
    assert price_data_frame["20_50_sma_cross_exit_signal"].any()


def test_attach_ema_sma_cross_and_rsi_signals_filters_by_rsi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EMA/SMA cross entry signal should require RSI to be 40 or below."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0]}
    )

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
    ) -> None:
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, True]
        )

    def fake_rsi(
        price_series: pandas.Series, window_size: int = 14
    ) -> pandas.Series:
        return pandas.Series([50, 30, 50])

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(strategy_module, "rsi", fake_rsi)

    strategy_module.attach_ema_sma_cross_and_rsi_signals(price_data_frame)

    assert list(price_data_frame["ema_sma_cross_and_rsi_entry_signal"]) == [
        False,
        True,
        False,
    ]
    assert list(price_data_frame["ema_sma_cross_and_rsi_exit_signal"]) == [
        False,
        False,
        True,
    ]


def test_attach_ema_sma_cross_signals_requires_positive_long_term_sma_slope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EMA/SMA cross entry signal should require a positive long-term SMA slope."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0] * 5, "close": [1.0, 1.0, 4.0, 4.0, 4.0]}
    )

    def fake_ema(
        price_series: pandas.Series, window_size: int
    ) -> pandas.Series:
        return pandas.Series([1.0, 1.0, 2.0, 3.0, 4.0])

    def fake_sma_negative(
        price_series: pandas.Series, window_size: int
    ) -> pandas.Series:
        if window_size == strategy_module.LONG_TERM_SMA_WINDOW:
            return pandas.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        return pandas.Series([1.0, 1.0, 1.5, 2.5, 3.5])

    monkeypatch.setattr(strategy_module, "ema", fake_ema)
    monkeypatch.setattr(strategy_module, "sma", fake_sma_negative)

    strategy_module.attach_ema_sma_cross_signals(price_data_frame, window_size=2)

    assert list(price_data_frame["ema_sma_cross_entry_signal"]) == [
        False,
        False,
        False,
        False,
        False,
    ]

    def fake_sma_positive(
        price_series: pandas.Series, window_size: int
    ) -> pandas.Series:
        if window_size == strategy_module.LONG_TERM_SMA_WINDOW:
            return pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        return pandas.Series([1.0, 1.0, 1.5, 2.5, 3.5])

    monkeypatch.setattr(strategy_module, "sma", fake_sma_positive)

    price_data_frame = pandas.DataFrame(
        {"open": [1.0] * 5, "close": [1.0, 1.0, 4.0, 4.0, 4.0]}
    )
    strategy_module.attach_ema_sma_cross_signals(price_data_frame, window_size=2)

    assert list(price_data_frame["ema_sma_cross_entry_signal"]) == [
        False,
        False,
        False,
        True,
        False,
    ]


def test_attach_ftd_ema_sma_cross_signals_requires_recent_ftd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FTD/EMA-SMA cross entry signal should require a recent FTD."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0] * 7, "close": [1.0] * 7}
    )

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
    ) -> None:
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, False, False, False, True, False, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, False, False, False, False, False]
        )

    def fake_ftd(
        data_frame: pandas.DataFrame, buy_mark_day: int, tolerance: float = 1e-8
    ) -> bool:
        return len(data_frame) - 1 == 1

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(strategy_module, "ftd", fake_ftd)

    strategy_module.attach_ftd_ema_sma_cross_signals(price_data_frame)

    assert list(price_data_frame["ftd_ema_sma_cross_entry_signal"]) == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert list(price_data_frame["ftd_ema_sma_cross_exit_signal"]) == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_attach_ema_sma_cross_with_slope_requires_close_above_long_term_sma_and_slope_in_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EMA/SMA cross entry should require the closing price to be above the long-term SMA and a slope in range."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    recorded_require_close_above_long_term_sma: bool | None = None

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
    ) -> None:
        nonlocal recorded_require_close_above_long_term_sma
        recorded_require_close_above_long_term_sma = require_close_above_long_term_sma
        data_frame["sma_value"] = pandas.Series(
            [1.0, 0.8, 0.9, 1.2, 0.7]
        )
        data_frame["sma_previous"] = data_frame["sma_value"].shift(1)
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, False, False, True]
        )

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )

    strategy_module.attach_ema_sma_cross_with_slope_signals(price_data_frame)

    assert recorded_require_close_above_long_term_sma is True
    assert list(price_data_frame["ema_sma_cross_with_slope_entry_signal"]) == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert list(price_data_frame["ema_sma_cross_with_slope_exit_signal"]) == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_attach_ema_sma_cross_with_slope_signals_raises_value_error_for_invalid_slope_range() -> None:
    """``attach_ema_sma_cross_with_slope_signals`` should validate the slope range."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame({"open": [1.0], "close": [1.0]})

    with pytest.raises(
        ValueError, match="lower bound cannot exceed upper bound"
    ):
        strategy_module.attach_ema_sma_cross_with_slope_signals(
            price_data_frame, slope_range=(1.0, -1.0)
        )


def test_attach_ema_sma_cross_with_slope_and_volume_requires_higher_ema_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry requires EMA dollar volume to exceed the SMA value."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0],
            "volume": [1.0, 2.0, 3.0, 2.0, 1.0],
        }
    )

    def fake_attach_ema_sma_cross_with_slope_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        slope_range: tuple[float, float] = (-0.3, 1.0),
    ) -> None:
        data_frame["ema_sma_cross_with_slope_entry_signal"] = pandas.Series(
            [False, True, True, True, True]
        )
        data_frame["ema_sma_cross_with_slope_exit_signal"] = pandas.Series(
            [False, False, False, False, True]
        )

    monkeypatch.setattr(
        strategy_module,
        "attach_ema_sma_cross_with_slope_signals",
        fake_attach_ema_sma_cross_with_slope_signals,
    )

    strategy_module.attach_ema_sma_cross_with_slope_and_volume_signals(
        price_data_frame, window_size=3
    )

    assert list(
        price_data_frame["ema_sma_cross_with_slope_and_volume_entry_signal"]
    ) == [False, False, True, False, False]
    assert list(
        price_data_frame["ema_sma_cross_with_slope_and_volume_exit_signal"]
    ) == [False, False, False, False, True]


def test_attach_ema_sma_cross_with_slope_and_volume_signals_raises_value_error_for_invalid_slope_range() -> None:
    """``attach_ema_sma_cross_with_slope_and_volume_signals`` should validate the slope range."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0], "close": [1.0], "volume": [1.0]}
    )

    with pytest.raises(
        ValueError, match="lower bound cannot exceed upper bound"
    ):
        strategy_module.attach_ema_sma_cross_with_slope_and_volume_signals(
            price_data_frame, slope_range=(1.0, -1.0)
        )


def test_attach_ema_sma_double_cross_requires_long_term_ema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The double cross entry should require the long-term EMA above the SMA."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    price_data_frame = pandas.DataFrame(
        {"open": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0]}
    )

    recorded_require_close: bool | None = None

    def fake_attach_ema_sma_cross_signals(
        data_frame: pandas.DataFrame,
        window_size: int = 50,
        require_close_above_long_term_sma: bool = True,
    ) -> None:
        nonlocal recorded_require_close
        recorded_require_close = require_close_above_long_term_sma
        data_frame["ema_sma_cross_entry_signal"] = pandas.Series(
            [False, True, True]
        )
        data_frame["ema_sma_cross_exit_signal"] = pandas.Series(
            [False, False, True]
        )
        data_frame["long_term_sma_previous"] = pandas.Series([1.0, 1.0, 1.0])

    def fake_ema(
        price_series: pandas.Series, window_size: int
    ) -> pandas.Series:
        if window_size == strategy_module.LONG_TERM_SMA_WINDOW:
            return pandas.Series([1.0, 2.0, 0.5])
        return pandas.Series([0.0, 0.0, 0.0])

    monkeypatch.setattr(
        strategy_module, "attach_ema_sma_cross_signals", fake_attach_ema_sma_cross_signals
    )
    monkeypatch.setattr(strategy_module, "ema", fake_ema)

    strategy_module.attach_ema_sma_double_cross_signals(price_data_frame)

    assert recorded_require_close is False
    assert list(price_data_frame["ema_sma_double_cross_entry_signal"]) == [
        False,
        False,
        True,
    ]
    assert list(price_data_frame["ema_sma_double_cross_exit_signal"]) == [
        False,
        False,
        True,
    ]


def test_supported_strategies_includes_20_50_sma_cross() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the 20/50 SMA cross strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_20_50_sma_cross_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["20_50_sma_cross"]
        is attach_20_50_sma_cross_signals
    )


def test_supported_strategies_includes_ftd_ema_sma_cross() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the FTD/EMA-SMA cross strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ftd_ema_sma_cross_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ftd_ema_sma_cross"]
        is attach_ftd_ema_sma_cross_signals
    )


def test_supported_strategies_includes_ema_sma_cross_with_slope() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the EMA/SMA cross with slope strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ema_sma_cross_with_slope_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ema_sma_cross_with_slope"]
        is attach_ema_sma_cross_with_slope_signals
    )


def test_supported_strategies_includes_ema_sma_cross_with_slope_and_volume() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the slope and volume strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ema_sma_cross_with_slope_and_volume_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ema_sma_cross_with_slope_and_volume"]
        is attach_ema_sma_cross_with_slope_and_volume_signals
    )


def test_supported_strategies_includes_ema_sma_double_cross() -> None:
    """``SUPPORTED_STRATEGIES`` should expose the EMA/SMA double cross strategy."""
    # TODO: review

    from stock_indicator.strategy import (
        SUPPORTED_STRATEGIES,
        attach_ema_sma_double_cross_signals,
    )

    assert (
        SUPPORTED_STRATEGIES["ema_sma_double_cross"]
        is attach_ema_sma_double_cross_signals
    )


def test_parse_strategy_name_with_window_size() -> None:
    """``parse_strategy_name`` should parse the window size suffix."""

    base_name, window_size, slope_range = parse_strategy_name(
        "ema_sma_cross_with_slope_40"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size == 40
    assert slope_range is None


def test_parse_strategy_name_with_window_and_slope_range() -> None:
    """The parser should extract both window size and slope range."""

    base_name, window_size, slope_range = parse_strategy_name(
        "ema_sma_cross_with_slope_40_-0.5_0.5"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size == 40
    assert slope_range == (-0.5, 0.5)


def test_parse_strategy_name_with_slope_range_only() -> None:
    """The parser should handle slope range without window size."""

    base_name, window_size, slope_range = parse_strategy_name(
        "ema_sma_cross_with_slope_-0.5_0.5"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size is None
    assert slope_range == (-0.5, 0.5)


def test_parse_strategy_name_with_integer_slope_values() -> None:
    """``parse_strategy_name`` should convert integer slope bounds to floats."""

    base_name, window_size, slope_range = parse_strategy_name(
        "ema_sma_cross_with_slope_-1_2"
    )
    assert base_name == "ema_sma_cross_with_slope"
    assert window_size is None
    assert slope_range == (-1.0, 2.0)


def test_parse_strategy_name_without_suffix() -> None:
    """``parse_strategy_name`` should return ``None`` when no suffix is given."""

    base_name, window_size, slope_range = parse_strategy_name("ema_sma_cross")
    assert base_name == "ema_sma_cross"
    assert window_size is None
    assert slope_range is None


def test_parse_strategy_name_rejects_malformed_suffix() -> None:
    """``parse_strategy_name`` should raise ``ValueError`` for invalid suffixes."""

    with pytest.raises(ValueError, match="Malformed strategy name"):
        parse_strategy_name("ema_sma_cross_with_slope_")
    with pytest.raises(ValueError, match="positive integer"):
        parse_strategy_name("ema_sma_cross_with_slope_0")


def test_evaluate_combined_strategy_uses_window_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should parse window suffixes and rename signal columns."""

    date_index = pandas.date_range("2020-01-01", periods=2, freq="D")
    price_data_frame = pandas.DataFrame(
        {"Date": date_index, "open": [10.0, 10.0], "close": [10.0, 10.0]}
    )
    csv_path = tmp_path / "suffix.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_arguments: dict[str, int | None] = {"window_size": None}

    def fake_attach_signals(frame: pandas.DataFrame, window_size: int = 50) -> None:
        captured_arguments["window_size"] = window_size
        frame["ema_sma_cross_with_slope_entry_signal"] = [True, False]
        frame["ema_sma_cross_with_slope_exit_signal"] = [False, True]

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        assert "ema_sma_cross_with_slope_40_entry_signal" in passed_frame.columns
        assert "ema_sma_cross_with_slope_40_exit_signal" in passed_frame.columns
        trade = Trade(
            entry_date=date_index[0],
            exit_date=date_index[1],
            entry_price=10.0,
            exit_price=11.0,
            profit=1.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=1.0)

    monkeypatch.setattr(
        strategy, "attach_ema_sma_cross_with_slope_signals", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.BUY_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setitem(
        strategy.SELL_STRATEGIES, "ema_sma_cross_with_slope", fake_attach_signals
    )
    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross_with_slope_40",
        "ema_sma_cross_with_slope_40",
    )

    assert captured_arguments["window_size"] == 40
    assert result.total_trades == 1
