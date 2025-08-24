"""Tests for strategy evaluation utilities."""
# TODO: review

import os
import sys
from pathlib import Path

import pandas
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import stock_indicator.strategy as strategy
from stock_indicator.simulator import SimulationResult, Trade

from stock_indicator.strategy import (
    evaluate_ema_sma_cross_strategy,
    evaluate_kalman_channel_strategy,
    evaluate_combined_strategy,
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


def test_evaluate_combined_strategy_unsupported_name(tmp_path: Path) -> None:
    """evaluate_combined_strategy should raise for unknown strategies."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        evaluate_combined_strategy(tmp_path, "unknown", "ema_sma_cross")


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

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume=20,
    )
    assert result.total_trades == 0
    assert simulate_called["called"] is False

    simulate_called["called"] = False
    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume=5,
    )
    assert simulate_called["called"] is True


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


def test_evaluate_combined_strategy_handles_adjusted_close_column(
    tmp_path: Path,
) -> None:
    """The dollar volume filter should handle files with an Adj Close column."""

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    price_value_list = [100.0] * 60
    volume_value_list = [1_000_000] * 60
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "Open": price_value_list,
            "High": price_value_list,
            "Low": price_value_list,
            "Close": price_value_list,
            "Adj Close": price_value_list,
            "Volume": volume_value_list,
        }
    )
    csv_file_path = tmp_path / "adjusted_close.csv"
    price_data_frame.to_csv(csv_file_path, index=False)

    result = evaluate_combined_strategy(
        tmp_path,
        "ema_sma_cross",
        "ema_sma_cross",
        minimum_average_dollar_volume=1.0,
    )

    assert result.total_trades == 0


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
        data_frame: pandas.DataFrame, window_size: int = 50
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
