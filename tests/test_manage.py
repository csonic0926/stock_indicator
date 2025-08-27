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
def test_find_signal_prints_recalculated_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should display recalculated entry and exit signals for a date."""
    import stock_indicator.manage as manage_module

    recorded_arguments: dict[str, object] = {}

    def fake_find_signal(
        date_string: str,
        dollar_volume_filter: str,
        buy_strategy: str,
        sell_strategy: str,
        stop_loss: float,
    ) -> dict[str, list[str]]:
        recorded_arguments["date"] = date_string
        recorded_arguments["filter"] = dollar_volume_filter
        recorded_arguments["buy"] = buy_strategy
        recorded_arguments["sell"] = sell_strategy
        recorded_arguments["stop"] = stop_loss
        return {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    monkeypatch.setattr(manage_module.daily_job, "find_signal", fake_find_signal)

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "find_signal 2024-01-10 dollar_volume>1%,-0.2% ema_sma_cross ema_sma_cross 1.0"
    )

    assert recorded_arguments == {
        "date": "2024-01-10",
        "filter": "dollar_volume>1%,-0.2%",
        "buy": "ema_sma_cross",
        "sell": "ema_sma_cross",
        "stop": 1.0,
    }
    assert output_buffer.getvalue().splitlines() == ["['AAA']", "['BBB']"]


# TODO: review
def test_find_signal_invalid_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should require five arguments."""
    import stock_indicator.manage as manage_module

    call_record = {"called": False}

    def fake_find_signal(
        date_string: str,
        dollar_volume_filter: str,
        buy_strategy: str,
        sell_strategy: str,
        stop_loss: float,
    ) -> dict[str, list[str]]:
        call_record["called"] = True
        return {"entry_signals": [], "exit_signals": []}

    monkeypatch.setattr(manage_module.daily_job, "find_signal", fake_find_signal)

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("find_signal invalid-date")

    assert call_record["called"] is False
    assert (
        output_buffer.getvalue()
        == "usage: find_signal DATE DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS\n"
    )


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
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        call_record["strategies"] = (buy_strategy_name, sell_strategy_name)
        volume_record["ratio"] = minimum_average_dollar_volume_ratio
        volume_record["increment"] = dollar_volume_ratio_increment
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
                    simple_moving_average_dollar_volume=100_000_000.0,
                    total_simple_moving_average_dollar_volume=1_000_000_000.0,
                    simple_moving_average_dollar_volume_ratio=0.1,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2023-01-05"),
                    symbol="AAA",
                    action="close",
                    price=11.0,
                    simple_moving_average_dollar_volume=100_000_000.0,
                    total_simple_moving_average_dollar_volume=1_000_000_000.0,
                    simple_moving_average_dollar_volume_ratio=0.1,
                    result="win",
                    percentage_change=0.1,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2023-02-10"),
                    symbol="BBB",
                    action="open",
                    price=20.0,
                    simple_moving_average_dollar_volume=200_000_000.0,
                    total_simple_moving_average_dollar_volume=1_000_000_000.0,
                    simple_moving_average_dollar_volume_ratio=0.2,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2023-02-15"),
                    symbol="BBB",
                    action="close",
                    price=21.0,
                    simple_moving_average_dollar_volume=200_000_000.0,
                    total_simple_moving_average_dollar_volume=1_000_000_000.0,
                    simple_moving_average_dollar_volume_ratio=0.2,
                    result="win",
                    percentage_change=0.05,
                ),
            ],
            2024: [
                TradeDetail(
                    date=pandas.Timestamp("2024-03-01"),
                    symbol="CCC",
                    action="open",
                    price=30.0,
                    simple_moving_average_dollar_volume=300_000_000.0,
                    total_simple_moving_average_dollar_volume=1_000_000_000.0,
                    simple_moving_average_dollar_volume_ratio=0.3,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2024-03-05"),
                    symbol="CCC",
                    action="close",
                    price=29.0,
                    simple_moving_average_dollar_volume=300_000_000.0,
                    total_simple_moving_average_dollar_volume=1_000_000_000.0,
                    simple_moving_average_dollar_volume_ratio=0.3,
                    result="lose",
                    percentage_change=-1.0 / 30.0,
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
            maximum_drawdown=0.25,
            final_balance=123.45,
            compound_annual_growth_rate=0.1,
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
    shell.onecmd("start_simulate dollar_volume>2.41%,-0.2% ema_sma_cross ema_sma_cross")
    assert call_record["strategies"] == ("ema_sma_cross", "ema_sma_cross")
    assert volume_record["ratio"] == pytest.approx(0.0241)
    assert volume_record["increment"] == pytest.approx(-0.002)
    assert stop_loss_record["value"] == 1.0
    assert "Simulation start date: 2019-01-01" in output_buffer.getvalue()
    assert (
        "Trades: 3, Win rate: 50.00%, Mean profit %: 10.00%, Profit % Std Dev: 0.00%, "
        "Mean loss %: 5.00%, Loss % Std Dev: 0.00%, Mean holding period: 2.00 bars, "
        "Holding period Std Dev: 1.00 bars, Max concurrent positions: 2, Final balance: 123.45, CAGR: 10.00%, Max drawdown: 25.00%"
        in output_buffer.getvalue()
    )
    assert "Year 2023: 10.00%, trade: 2" in output_buffer.getvalue()
    assert "Year 2024: -5.00%, trade: 1" in output_buffer.getvalue()
    assert (
        "  2023-01-02 AAA open 10.00 0.1000 100.00M 1000.00M"
        in output_buffer.getvalue()
    )
    assert (
        "  2023-01-05 AAA close 11.00 0.1000 100.00M 1000.00M win 10.00%"
        in output_buffer.getvalue()
    )
    assert (
        "  2024-03-01 CCC open 30.00 0.3000 300.00M 1000.00M"
        in output_buffer.getvalue()
    )
    assert (
        "  2024-03-05 CCC close 29.00 0.3000 300.00M 1000.00M lose -3.33%"
        in output_buffer.getvalue()
    )


def test_start_simulate_filters_early_googl_trades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should drop early GOOGL trades from reported metrics."""
    import stock_indicator.manage as manage_module

    from stock_indicator.strategy import StrategyMetrics, TradeDetail

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        trade_details_by_year = {
            2013: [
                TradeDetail(
                    date=pandas.Timestamp("2013-01-02"),
                    symbol="GOOGL",
                    action="open",
                    price=10.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2013-01-10"),
                    symbol="GOOGL",
                    action="close",
                    price=12.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                    result="win",
                    percentage_change=0.2,
                ),
            ],
            2015: [
                TradeDetail(
                    date=pandas.Timestamp("2015-06-01"),
                    symbol="XYZ",
                    action="open",
                    price=20.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2015-06-10"),
                    symbol="XYZ",
                    action="close",
                    price=19.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                    result="lose",
                    percentage_change=-0.05,
                ),
            ],
        }
        return StrategyMetrics(
            total_trades=2,
            win_rate=0.5,
            mean_profit_percentage=0.2,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.05,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=4.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=1,
            maximum_drawdown=0.1,
            final_balance=100.0,
            compound_annual_growth_rate=0.1,
            annual_returns={2013: 0.05, 2015: -0.1},
            annual_trade_counts={2013: 1, 2015: 1},
            trade_details_by_year=trade_details_by_year,
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate dollar_volume>0%,0% ema_sma_cross ema_sma_cross")
    output_string = output_buffer.getvalue()
    assert "GOOGL" not in output_string
    assert "Year 2013" not in output_string
    assert "Year 2015: -10.00%, trade: 1" in output_string
    assert "Trades: 1," in output_string


def test_start_simulate_different_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should support different buy and sell strategies."""
    import stock_indicator.manage as manage_module

    call_arguments: dict[str, tuple[str, str]] = {}
    stop_loss_record: dict[str, float] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        call_arguments["strategies"] = (buy_strategy_name, sell_strategy_name)
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("start_simulate dollar_volume>0%,0% ema_sma_cross kalman_filtering")
    assert call_arguments["strategies"] == (
        "ema_sma_cross",
        "kalman_filtering",
    )
    assert stop_loss_record["value"] == 1.0


def test_start_simulate_accepts_start_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should forward the start date to evaluation."""
    import stock_indicator.manage as manage_module

    recorded_arguments: dict[str, pandas.Timestamp | None] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        recorded_arguments["start_date"] = start_date
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "start_simulate start=2018-01-01 dollar_volume>0%,0% ema_sma_cross ema_sma_cross"
    )
    assert recorded_arguments["start_date"] == pandas.Timestamp("2018-01-01")
    assert "Simulation start date: 2018-01-01" in output_buffer.getvalue()


def test_start_simulate_dollar_volume_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should forward the ratio filter to evaluation."""
    import stock_indicator.manage as manage_module

    ratio_record: dict[str, float | None] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        ratio_record["ratio"] = minimum_average_dollar_volume_ratio
        ratio_record["increment"] = dollar_volume_ratio_increment
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("start_simulate dollar_volume>1%,-0.2% ema_sma_cross ema_sma_cross")
    assert ratio_record["ratio"] == 0.01
    assert ratio_record["increment"] == pytest.approx(-0.002)




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
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
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
        "start_simulate dollar_volume>0%,0% "
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
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
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
        "start_simulate dollar_volume>0%,0% "
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
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
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
        "start_simulate dollar_volume>0%,0% "
        "ema_sma_cross_with_slope_and_volume "
        "ema_sma_cross_with_slope_and_volume"
    )
    assert call_arguments["strategies"] == (
        "ema_sma_cross_with_slope_and_volume",
        "ema_sma_cross_with_slope_and_volume",
    )


def test_start_simulate_accepts_slope_range_strategy_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should forward slope-range strategy names for evaluation."""

    import stock_indicator.manage as manage_module

    recorded_arguments: dict[str, tuple[str, str]] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        recorded_arguments["strategies"] = (buy_strategy_name, sell_strategy_name)
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
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
        "start_simulate dollar_volume>0%,0% "
        "ema_sma_cross_with_slope_-0.5_0.5 "
        "ema_sma_cross_with_slope_-0.5_0.5"
    )

    assert recorded_arguments["strategies"] == (
        "ema_sma_cross_with_slope_-0.5_0.5",
        "ema_sma_cross_with_slope_-0.5_0.5",
    )


def test_start_simulate_reports_missing_slope_bound() -> None:
    """The command should report a missing slope bound in strategy names."""

    import stock_indicator.manage as manage_module

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "start_simulate dollar_volume>0%,0% "
        "ema_sma_cross_with_slope_-0.5 ema_sma_cross"
    )
    assert (
        output_buffer.getvalue()
        == "Malformed strategy name: expected two numeric segments for slope range but found 1 in 'ema_sma_cross_with_slope_-0.5'\n"
    )


def test_start_simulate_reports_extra_slope_bound() -> None:
    """The command should report when too many slope bounds are provided."""

    import stock_indicator.manage as manage_module

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "start_simulate dollar_volume>0%,0% "
        "ema_sma_cross_with_slope_-0.5_0.5_1.0 ema_sma_cross"
    )
    assert (
        output_buffer.getvalue()
        == "Malformed strategy name: expected two numeric segments for slope range but found 3 in 'ema_sma_cross_with_slope_-0.5_0.5_1.0'\n"
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
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
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
        "start_simulate dollar_volume>0%,0% 20_50_sma_cross 20_50_sma_cross"
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
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
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
        "start_simulate dollar_volume>1%,-0.2% ema_sma_cross ema_sma_cross 0.5"
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
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
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
        "start_simulate starting_cash=5000 withdraw=1000 dollar_volume>0%,0% ema_sma_cross ema_sma_cross"
    )
    assert recorded_values["cash"] == 5000.0
    assert recorded_values["withdraw"] == 1000.0


def test_start_simulate_unsupported_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should report unsupported strategy names."""
    import stock_indicator.manage as manage_module

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate dollar_volume>0%,0% unknown ema_sma_cross")
    assert "unsupported strategies" in output_buffer.getvalue()


def test_start_simulate_rejects_sell_only_buy_strategy() -> None:
    """The command should reject strategies that are sell only when used for buying."""
    import stock_indicator.manage as manage_module

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "start_simulate dollar_volume>0%,0% kalman_filtering ema_sma_cross"
    )
    assert "unsupported strategies" in output_buffer.getvalue()

def test_start_simulate_accepts_windowed_strategy_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should accept strategies with numeric window suffixes."""
    # TODO: review

    import stock_indicator.manage as manage_module
    import stock_indicator.strategy as strategy_module

    output_buffer = io.StringIO()

    monkeypatch.setattr(
        strategy_module,
        "BUY_STRATEGIES",
        {"noop": lambda frame: None},
    )
    monkeypatch.setattr(
        strategy_module,
        "SELL_STRATEGIES",
        {"noop": lambda frame: None},
    )

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume_ratio: float | None = None,
        dollar_volume_ratio_increment: float = 0.0,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
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
            maximum_drawdown=0.0,
            final_balance=0.0,
            compound_annual_growth_rate=0.0,
            annual_returns={},
            annual_trade_counts={},
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate dollar_volume>0%,0% noop_5 noop_10")
    assert "unsupported strategies" not in output_buffer.getvalue()
