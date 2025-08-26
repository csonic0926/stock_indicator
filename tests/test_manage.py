"""Tests for the interactive management shell."""

# TODO: review

import io
import os
import sys
from pathlib import Path

import pandas
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def test_update_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should invoke the symbol cache update."""
    import stock_indicator.manage as manage_module

    call_record = {"called": False}

    def fake_update_symbol_cache() -> None:
        call_record["called"] = True

    monkeypatch.setattr(
        manage_module.symbols, "update_symbol_cache", fake_update_symbol_cache
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_symbols")
    assert call_record["called"] is True


def test_update_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should download data and write it to a CSV file."""
    import stock_indicator.manage as manage_module

    recorded_arguments: dict[str, str] = {}

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        recorded_arguments["symbol"] = symbol
        recorded_arguments["start"] = start
        recorded_arguments["end"] = end
        return pandas.DataFrame(
            {"close": [1.0]}, index=pandas.to_datetime(["2023-01-01"])
        ).rename_axis("Date")

    monkeypatch.setattr(
        manage_module.data_loader, "download_history", fake_download_history
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_data TEST 2023-01-01 2023-01-02")
    output_file = tmp_path / "TEST.csv"
    assert output_file.exists()
    csv_contents = pandas.read_csv(output_file)
    assert "Date" in csv_contents.columns
    assert recorded_arguments == {
        "symbol": "TEST",
        "start": "2023-01-01",
        "end": "2023-01-02",
    }


def test_update_all_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should download data for every symbol in the cache."""
    import stock_indicator.manage as manage_module

    symbol_list = ["AAA", "BBB", manage_module.SP500_SYMBOL]

    def fake_load_symbols() -> list[str]:
        return symbol_list

    download_calls: list[str] = []

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        download_calls.append(symbol)
        return pandas.DataFrame(
            {"close": [1.0]}, index=pandas.to_datetime(["2023-01-01"])
        ).rename_axis("Date")

    monkeypatch.setattr(manage_module.symbols, "load_symbols", fake_load_symbols)
    monkeypatch.setattr(
        manage_module.data_loader, "download_history", fake_download_history
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    expected_symbols = symbol_list
    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_all_data 2023-01-01 2023-01-02")
    for symbol in expected_symbols:
        csv_path = tmp_path / f"{symbol}.csv"
        assert csv_path.exists()
        csv_contents = pandas.read_csv(csv_path)
        assert "Date" in csv_contents.columns
    assert download_calls == expected_symbols


# TODO: review
def test_count_symbols_with_average_dollar_volume_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should report how many symbols exceed a dollar volume threshold."""
    import stock_indicator.manage as manage_module

    call_arguments: dict[str, float] = {}

    def fake_counter(data_directory: Path, minimum_average_dollar_volume: float) -> int:
        call_arguments["threshold"] = minimum_average_dollar_volume
        assert data_directory == manage_module.DATA_DIRECTORY
        return 7

    monkeypatch.setattr(
        manage_module.volume,
        "count_symbols_with_average_dollar_volume_above",
        fake_counter,
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("count_symbols_with_average_dollar_volume_above 10")
    assert call_arguments["threshold"] == 10.0
    assert output_buffer.getvalue().strip() == "7"


def test_start_simulate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should evaluate strategies and display metrics."""
    import stock_indicator.manage as manage_module

    call_record: dict[str, tuple[str, str]] = {}
    volume_record: dict[str, float] = {}
    stop_loss_record: dict[str, float] = {}

    from stock_indicator.strategy import StrategyMetrics, TradeDetail

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        call_record["strategies"] = (buy_strategy_name, sell_strategy_name)
        volume_record["threshold"] = minimum_average_dollar_volume
        stop_loss_record["value"] = stop_loss_percentage
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        assert data_directory == manage_module.DATA_DIRECTORY
        trade_details_by_year = {
            2023: [
                TradeDetail(
                    date=pandas.Timestamp("2023-01-02"),
                    symbol="AAA",
                    action="open",
                    price=10.0,
                    simple_moving_average_dollar_volume_ratio=0.1,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2023-01-05"),
                    symbol="AAA",
                    action="close",
                    price=11.0,
                    simple_moving_average_dollar_volume_ratio=0.1,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2023-02-10"),
                    symbol="BBB",
                    action="open",
                    price=20.0,
                    simple_moving_average_dollar_volume_ratio=0.2,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2023-02-15"),
                    symbol="BBB",
                    action="close",
                    price=21.0,
                    simple_moving_average_dollar_volume_ratio=0.2,
                ),
            ],
            2024: [
                TradeDetail(
                    date=pandas.Timestamp("2024-03-01"),
                    symbol="CCC",
                    action="open",
                    price=30.0,
                    simple_moving_average_dollar_volume_ratio=0.3,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2024-03-05"),
                    symbol="CCC",
                    action="close",
                    price=29.0,
                    simple_moving_average_dollar_volume_ratio=0.3,
                ),
            ],
        }
        return StrategyMetrics(
            total_trades=3,
            win_rate=0.5,
            mean_profit_percentage=0.1,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.05,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=2.0,
            holding_period_standard_deviation=1.0,
            maximum_concurrent_positions=2,
            final_balance=123.45,
            annual_returns={2023: 0.1, 2024: -0.05},
            annual_trade_counts={2023: 2, 2024: 1},
            trade_details_by_year=trade_details_by_year,
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate dollar_volume>500 ema_sma_cross ema_sma_cross")
    assert call_record["strategies"] == ("ema_sma_cross", "ema_sma_cross")
    assert volume_record["threshold"] == 500.0
    assert stop_loss_record["value"] == 1.0
    assert "Simulation start date: 2019-01-01" in output_buffer.getvalue()
    assert (
        "Trades: 3, Win rate: 50.00%, Mean profit %: 10.00%, Profit % Std Dev: 0.00%, "
        "Mean loss %: 5.00%, Loss % Std Dev: 0.00%, Mean holding period: 2.00 bars, "
        "Holding period Std Dev: 1.00 bars, Max concurrent positions: 2, Final balance: 123.45" in output_buffer.getvalue()
    )
    assert "Year 2023: 10.00%, trade: 2" in output_buffer.getvalue()
    assert "Year 2024: -5.00%, trade: 1" in output_buffer.getvalue()
    assert "  2023-01-02 AAA open 10.00 0.1000" in output_buffer.getvalue()
    assert "  2023-01-05 AAA close 11.00 0.1000" in output_buffer.getvalue()
    assert "  2024-03-01 CCC open 30.00 0.3000" in output_buffer.getvalue()
    assert "  2024-03-05 CCC close 29.00 0.3000" in output_buffer.getvalue()


def test_start_simulate_different_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should support different buy and sell strategies."""
    import stock_indicator.manage as manage_module

    call_arguments: dict[str, tuple[str, str]] = {}
    threshold_record: dict[str, float] = {}
    stop_loss_record: dict[str, float] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        call_arguments["strategies"] = (buy_strategy_name, sell_strategy_name)
        threshold_record["threshold"] = minimum_average_dollar_volume
        stop_loss_record["value"] = stop_loss_percentage
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("start_simulate dollar_volume>0 ema_sma_cross kalman_filtering")
    assert call_arguments["strategies"] == (
        "ema_sma_cross",
        "kalman_filtering",
    )
    assert threshold_record["threshold"] == 0.0
    assert stop_loss_record["value"] == 1.0


def test_start_simulate_dollar_volume_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should forward the ranking filter to evaluation."""
    import stock_indicator.manage as manage_module

    rank_record: dict[str, int | None] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        rank_record["rank"] = top_dollar_volume_rank
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("start_simulate dollar_volume=6th ema_sma_cross ema_sma_cross")
    assert rank_record["rank"] == 6


def test_start_simulate_dollar_volume_threshold_and_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should forward both threshold and ranking filters."""
    import stock_indicator.manage as manage_module

    recorded_values: dict[str, float | int | None] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        recorded_values["threshold"] = minimum_average_dollar_volume
        recorded_values["rank"] = top_dollar_volume_rank
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("start_simulate dollar_volume>100,6th ema_sma_cross ema_sma_cross")
    assert recorded_values["threshold"] == 100.0
    assert recorded_values["rank"] == 6


def test_start_simulate_dollar_volume_large_threshold_and_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should parse and forward a large threshold with ranking."""
    # TODO: review
    import stock_indicator.manage as manage_module

    recorded_values: dict[str, float | int | None] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        recorded_values["threshold"] = minimum_average_dollar_volume
        recorded_values["rank"] = top_dollar_volume_rank
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy, "evaluate_combined_strategy", fake_evaluate
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate dollar_volume>10000,6th ema_sma_cross ema_sma_cross"
    )
    assert recorded_values["threshold"] == 10000.0
    assert recorded_values["rank"] == 6


def test_start_simulate_supports_rsi_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should recognize the EMA/SMA cross with RSI strategy."""
    # TODO: review

    import stock_indicator.manage as manage_module

    call_arguments: dict[str, tuple[str, str]] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        call_arguments["strategies"] = (buy_strategy_name, sell_strategy_name)
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate dollar_volume>0 "
        "ema_sma_cross_and_rsi ema_sma_cross_and_rsi"
    )
    assert call_arguments["strategies"] == (
        "ema_sma_cross_and_rsi",
        "ema_sma_cross_and_rsi",
    )


def test_start_simulate_supports_slope_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should recognize the EMA/SMA cross with slope strategy."""
    # TODO: review

    import stock_indicator.manage as manage_module

    call_arguments: dict[str, tuple[str, str]] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        call_arguments["strategies"] = (buy_strategy_name, sell_strategy_name)
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate dollar_volume>0 "
        "ema_sma_cross_with_slope ema_sma_cross_with_slope"
    )
    assert call_arguments["strategies"] == (
        "ema_sma_cross_with_slope",
        "ema_sma_cross_with_slope",
    )


def test_start_simulate_supports_slope_and_volume_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should recognize the slope and volume strategy."""
    # TODO: review

    import stock_indicator.manage as manage_module

    call_arguments: dict[str, tuple[str, str]] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        call_arguments["strategies"] = (buy_strategy_name, sell_strategy_name)
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate dollar_volume>0 "
        "ema_sma_cross_with_slope_and_volume "
        "ema_sma_cross_with_slope_and_volume"
    )
    assert call_arguments["strategies"] == (
        "ema_sma_cross_with_slope_and_volume",
        "ema_sma_cross_with_slope_and_volume",
    )


def test_start_simulate_supports_20_50_sma_cross_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should recognize the 20/50 SMA cross strategy."""
    # TODO: review

    import stock_indicator.manage as manage_module

    call_arguments: dict[str, tuple[str, str]] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        call_arguments["strategies"] = (buy_strategy_name, sell_strategy_name)
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate dollar_volume>0 20_50_sma_cross 20_50_sma_cross"
    )
    assert call_arguments["strategies"] == (
        "20_50_sma_cross",
        "20_50_sma_cross",
    )


def test_start_simulate_accepts_stop_loss_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should forward the stop loss argument to evaluation."""
    import stock_indicator.manage as manage_module

    stop_loss_record: dict[str, float] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        stop_loss_record["value"] = stop_loss_percentage
        assert starting_cash == 3000.0
        assert withdraw_amount == 0.0
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate dollar_volume>100 ema_sma_cross ema_sma_cross 0.5"
    )
    assert stop_loss_record["value"] == 0.5


def test_start_simulate_accepts_cash_and_withdraw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should forward cash and withdraw arguments."""
    import stock_indicator.manage as manage_module

    recorded_values: dict[str, float] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
    ) -> StrategyMetrics:
        recorded_values["cash"] = starting_cash
        recorded_values["withdraw"] = withdraw_amount
        return StrategyMetrics(
            total_trades=0,
            win_rate=0.0,
            mean_profit_percentage=0.0,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=0.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=0,
            final_balance=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate starting_cash=5000 withdraw=1000 dollar_volume>0 ema_sma_cross ema_sma_cross"
    )
    assert recorded_values["cash"] == 5000.0
    assert recorded_values["withdraw"] == 1000.0


def test_start_simulate_unsupported_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should report unsupported strategy names."""
    import stock_indicator.manage as manage_module

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate dollar_volume>0 unknown ema_sma_cross")
    assert "unsupported strategies" in output_buffer.getvalue()


def test_start_simulate_rejects_sell_only_buy_strategy() -> None:
    """The command should reject strategies that are sell only when used for buying."""
    import stock_indicator.manage as manage_module

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "start_simulate dollar_volume>0 kalman_filtering ema_sma_cross"
    )
    assert "unsupported strategies" in output_buffer.getvalue()
