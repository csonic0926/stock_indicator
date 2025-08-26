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


def test_evaluate_combined_strategy_rejects_sell_only_buy(tmp_path: Path) -> None:
    """evaluate_combined_strategy should reject sell-only strategies used for buying."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        evaluate_combined_strategy(tmp_path, "kalman_filtering", "ema_sma_cross")


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


def test_evaluate_combined_strategy_dollar_volume_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function should process only the top symbols by 50-day average dollar volume."""

    import stock_indicator.strategy as strategy_module

    for symbol_name, volume_list in {
        "AAA": [1000.0] * 59 + [1.0],
        "BBB": [10.0] * 59 + [200.0],
        "CCC": [10.0] * 59 + [300.0],
    }.items():
        date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
        pandas.DataFrame(
            {
                "Date": date_index,
                "open": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": volume_list,
                "symbol": [symbol_name] * 60,
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    processed_symbols: list[str] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbols.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    strategy_module.evaluate_combined_strategy(
        tmp_path, "noop", "noop", top_dollar_volume_rank=2
    )
    assert set(processed_symbols) == {"AAA", "CCC"}


def test_evaluate_combined_strategy_dollar_volume_filter_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only symbols above the dollar volume threshold should be ranked."""

    import stock_indicator.strategy as strategy_module

    volumes_by_symbol = {
        "AAA": 100_000_000,
        "BBB": 200_000_000,
        "CCC": 300_000_000,
        "DDD": 250_000_000,
    }
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
    captured_counts: list[int] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbols.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    def fake_simulate_portfolio_balance(
        trades: Iterable[Trade],
        starting_cash: float,
        eligible_symbol_count: int,
        withdraw_amount: float = 0.0,
    ) -> float:
        captured_counts.append(eligible_symbol_count)
        assert withdraw_amount == 0.0
        return starting_cash

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(
        strategy_module, "simulate_portfolio_balance", fake_simulate_portfolio_balance
    )
    monkeypatch.setattr(strategy_module, "calculate_annual_returns", lambda *a, **k: {})
    monkeypatch.setattr(
        strategy_module, "calculate_annual_trade_counts", lambda *a, **k: {}
    )
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    strategy_module.evaluate_combined_strategy(
        tmp_path,
        "noop",
        "noop",
        minimum_average_dollar_volume=150,
        top_dollar_volume_rank=2,
    )
    assert set(processed_symbols) == {"CCC", "DDD"}
    assert captured_counts == [2]


def test_evaluate_combined_strategy_large_threshold_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the top six symbols should be processed when many exceed the threshold."""
    # TODO: review

    import stock_indicator.strategy as strategy_module

    volumes_by_symbol_name = {
        "AAA": 11_000_000_000,
        "BBB": 12_000_000_000,
        "CCC": 13_000_000_000,
        "DDD": 14_000_000_000,
        "EEE": 15_000_000_000,
        "FFF": 16_000_000_000,
        "GGG": 17_000_000_000,
    }
    for symbol_name, volume_value in volumes_by_symbol_name.items():
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

    processed_symbol_names: list[str] = []
    captured_counts: list[int] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> SimulationResult:
        processed_symbol_names.append(data["symbol"].iloc[0])
        return SimulationResult(trades=[], total_profit=0.0)

    def fake_simulate_portfolio_balance(
        trades: Iterable[Trade], starting_cash: float, eligible_symbol_count: int
    ) -> float:
        captured_counts.append(eligible_symbol_count)
        return starting_cash

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(
        strategy_module, "simulate_portfolio_balance", fake_simulate_portfolio_balance
    )
    monkeypatch.setattr(strategy_module, "calculate_annual_returns", lambda *a, **k: {})
    monkeypatch.setattr(
        strategy_module, "calculate_annual_trade_counts", lambda *a, **k: {}
    )
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda df: None})
    monkeypatch.setattr(strategy_module, "SUPPORTED_STRATEGIES", {"noop": lambda df: None})

    strategy_module.evaluate_combined_strategy(
        tmp_path,
        "noop",
        "noop",
        minimum_average_dollar_volume=10000,
        top_dollar_volume_rank=6,
    )

    assert set(processed_symbol_names) == {
        "BBB",
        "CCC",
        "DDD",
        "EEE",
        "FFF",
        "GGG",
    }
    assert captured_counts == [6]


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
