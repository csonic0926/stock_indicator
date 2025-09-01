"""Tests for the interactive management shell."""

# TODO: review

import io
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import pandas
import pytest

from stock_indicator.simulator import SimulationResult
from stock_indicator.strategy import StrategyMetrics


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


def test_update_data_from_yf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should download data from Yahoo Finance and write CSV."""
    import stock_indicator.manage as manage_module

    call_symbols: list[str] = []

    def fake_download_history(symbol: str, start: str, end: str) -> pandas.DataFrame:
        call_symbols.append(symbol)
        return pandas.DataFrame(
            {"close": [1.0]}, index=pandas.to_datetime(["2023-01-01"])
        ).rename_axis("Date")

    monkeypatch.setattr(
        manage_module.data_loader, "download_history", fake_download_history
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("update_data_from_yf TEST 2023-01-01 2023-01-02")
    output_file = tmp_path / "TEST.csv"
    assert output_file.exists()
    csv_contents = pandas.read_csv(output_file)
    assert "Date" in csv_contents.columns
    # Expect two downloads: requested symbol and ^GSPC
    assert call_symbols[0] == "TEST"
    assert "^GSPC" in call_symbols


def test_update_all_data_from_yf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should download data from Yahoo Finance for each cached symbol."""
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
    shell.onecmd("update_all_data_from_yf 2023-01-01 2023-01-02")
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
        allowed_group_identifiers: set[int] | None = None,
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
        "find_signal 2024-01-10 dollar_volume>1 ema_sma_cross ema_sma_cross 1.0"
    )

    assert recorded_arguments == {
        "date": "2024-01-10",
        "filter": "dollar_volume>1",
        "buy": "ema_sma_cross",
        "sell": "ema_sma_cross",
        "stop": 1.0,
    }
    assert output_buffer.getvalue().splitlines() == [
        "entry signals: ['AAA']",
        "exit signals: ['BBB']",
    ]


# TODO: review
def test_find_signal_invalid_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should reject incomplete arguments."""
    import stock_indicator.manage as manage_module

    call_record = {"called": False}

    def fake_find_signal(
        date_string: str,
        dollar_volume_filter: str,
        buy_strategy: str,
        sell_strategy: str,
        stop_loss: float,
        allowed_group_identifiers: set[int] | None = None,
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
        ==
        "usage: find_signal DATE DOLLAR_VOLUME_FILTER (BUY SELL STOP_LOSS | STOP_LOSS strategy=ID) [group=1,2,...]\n"
    )


# TODO: review
def test_find_signal_with_strategy_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should map a strategy id to buy and sell strategies."""
    import stock_indicator.manage as manage_module

    recorded_arguments: dict[str, object] = {}

    def fake_find_signal(
        date_string: str,
        dollar_volume_filter: str,
        buy_strategy: str,
        sell_strategy: str,
        stop_loss: float,
        allowed_group_identifiers: set[int] | None = None,
    ) -> dict[str, list[str]]:
        recorded_arguments["date"] = date_string
        recorded_arguments["filter"] = dollar_volume_filter
        recorded_arguments["buy"] = buy_strategy
        recorded_arguments["sell"] = sell_strategy
        recorded_arguments["stop"] = stop_loss
        recorded_arguments["group"] = allowed_group_identifiers
        return {"entry_signals": [], "exit_signals": []}

    monkeypatch.setattr(manage_module.daily_job, "find_signal", fake_find_signal)

    def fake_load_mapping() -> dict[str, tuple[str, str]]:
        return {"ID": ("ema_sma_cross", "ema_sma_cross")}

    monkeypatch.setattr(manage_module, "load_strategy_set_mapping", fake_load_mapping)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd("find_signal 2024-01-10 dollar_volume>1 1.0 strategy=ID")

    assert recorded_arguments == {
        "date": "2024-01-10",
        "filter": "dollar_volume>1",
        "buy": "ema_sma_cross",
        "sell": "ema_sma_cross",
        "stop": 1.0,
        "group": None,
    }


    


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
        minimum_average_dollar_volume_ratio: float | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        call_record["strategies"] = (buy_strategy_name, sell_strategy_name)
        volume_record["threshold"] = minimum_average_dollar_volume
        if minimum_average_dollar_volume_ratio is not None:
            volume_record["ratio"] = minimum_average_dollar_volume_ratio
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
    shell.onecmd("start_simulate dollar_volume>500 ema_sma_cross ema_sma_cross")
    assert call_record["strategies"] == ("ema_sma_cross", "ema_sma_cross")
    assert volume_record["threshold"] == 500.0
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
        "AAA open 10.00 0.1000 100.00M 1000.00M price_score="
        in output_buffer.getvalue()
    )
    assert (
        "  2023-01-05 AAA close 11.00 0.1000 100.00M 1000.00M win 10.00%"
        in output_buffer.getvalue()
    )
    assert (
        "CCC open 30.00 0.3000 300.00M 1000.00M price_score="
        in output_buffer.getvalue()
    )
    assert (
        "  2024-03-05 CCC close 29.00 0.3000 300.00M 1000.00M lose -3.33%"
        in output_buffer.getvalue()
    )


def test_start_simulate_suppresses_trade_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command should hide trade details when the flag is False."""
    import stock_indicator.manage as manage_module

    from stock_indicator.strategy import StrategyMetrics, TradeDetail

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        minimum_average_dollar_volume_ratio: float | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        trade_details_by_year = {
            2023: [
                TradeDetail(
                    date=pandas.Timestamp("2023-01-02"),
                    symbol="AAA",
                    action="open",
                    price=10.0,
                    simple_moving_average_dollar_volume=100.0,
                    total_simple_moving_average_dollar_volume=1000.0,
                    simple_moving_average_dollar_volume_ratio=0.1,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2023-01-05"),
                    symbol="AAA",
                    action="close",
                    price=11.0,
                    simple_moving_average_dollar_volume=100.0,
                    total_simple_moving_average_dollar_volume=1000.0,
                    simple_moving_average_dollar_volume_ratio=0.1,
                    result="win",
                    percentage_change=0.1,
                ),
            ]
        }
        return StrategyMetrics(
            total_trades=1,
            win_rate=1.0,
            mean_profit_percentage=0.1,
            profit_percentage_standard_deviation=0.0,
            mean_loss_percentage=0.0,
            loss_percentage_standard_deviation=0.0,
            mean_holding_period=1.0,
            holding_period_standard_deviation=0.0,
            maximum_concurrent_positions=1,
            maximum_drawdown=0.0,
            final_balance=100.0,
            compound_annual_growth_rate=0.1,
            annual_returns={2023: 0.1},
            annual_trade_counts={2023: 1},
            trade_details_by_year=trade_details_by_year,
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        fake_evaluate,
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "start_simulate dollar_volume>0 ema_sma_cross ema_sma_cross 1 false"
    )
    output_string = output_buffer.getvalue()
    assert "Year 2023: 10.00%, trade: 1" in output_string
    assert "AAA open" not in output_string
    assert "AAA close" not in output_string


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
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        minimum_average_dollar_volume_ratio: float | None = None,
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
    shell.onecmd("start_simulate dollar_volume>0 ema_sma_cross ema_sma_cross")
    output_string = output_buffer.getvalue()
    assert "GOOGL" not in output_string
    assert "Year 2013" not in output_string
    assert "Year 2015: -10.00%, trade: 1" in output_string
    assert "Trades: 1," in output_string


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
        minimum_average_dollar_volume_ratio: float | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
    shell.onecmd("start_simulate dollar_volume>0 ema_sma_cross kalman_filtering")
    assert call_arguments["strategies"] == (
        "ema_sma_cross",
        "kalman_filtering",
    )
    assert threshold_record["threshold"] == 0.0
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
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        minimum_average_dollar_volume_ratio: float | None = None,
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
        "start_simulate start=2018-01-01 dollar_volume>0 ema_sma_cross ema_sma_cross"
    )
    assert recorded_arguments["start_date"] == pandas.Timestamp("2018-01-01")
    assert "Simulation start date: 2018-01-01" in output_buffer.getvalue()


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
        minimum_average_dollar_volume_ratio: float | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
    shell.onecmd("start_simulate dollar_volume=6th ema_sma_cross ema_sma_cross")
    assert rank_record["rank"] == 6


def test_start_simulate_dollar_volume_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should forward the ratio filter to evaluation."""
    import stock_indicator.manage as manage_module

    ratio_record: dict[str, float | None] = {}

    from stock_indicator.strategy import StrategyMetrics

    def fake_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        minimum_average_dollar_volume_ratio: float | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
    ) -> StrategyMetrics:
        ratio_record["ratio"] = minimum_average_dollar_volume_ratio
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
    shell.onecmd("start_simulate dollar_volume>1% ema_sma_cross ema_sma_cross")
    assert ratio_record["ratio"] == 0.01


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
        minimum_average_dollar_volume_ratio: float | None = None,
        starting_cash: float = 3000.0,
        withdraw_amount: float = 0.0,
        stop_loss_percentage: float = 1.0,
        start_date: pandas.Timestamp | None = None,
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
    shell.onecmd("start_simulate dollar_volume>100,6th ema_sma_cross ema_sma_cross")
    assert recorded_values["threshold"] == 100.0
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
        minimum_average_dollar_volume_ratio: float | None = None,
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
        minimum_average_dollar_volume_ratio: float | None = None,
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
        minimum_average_dollar_volume_ratio: float | None = None,
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
        "start_simulate dollar_volume>0 "
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
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        minimum_average_dollar_volume_ratio: float | None = None,
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
        "start_simulate dollar_volume>0 "
        "ema_sma_cross_with_slope_-0.5_0.5 "
        "ema_sma_cross_with_slope_-0.5_0.5"
    )

    assert recorded_arguments["strategies"] == (
        "ema_sma_cross_with_slope_-0.5_0.5",
        "ema_sma_cross_with_slope_-0.5_0.5",
    )


def test_start_simulate_passes_distinct_window_sizes_and_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command should forward distinct window sizes and keep signals separate."""

    import stock_indicator.manage as manage_module
    import stock_indicator.strategy as strategy_module

    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [10.0] * 60,
            "close": [10.0] * 60,
            "volume": [1.0] * 60,
        }
    )
    csv_path = tmp_path / "sample.csv"
    price_data_frame.to_csv(csv_path, index=False)

    captured_details: dict[str, object] = {"strategies": (), "columns": []}

    def fake_attach_signals(
        frame: pandas.DataFrame, window_size: int = 40
    ) -> None:
        entry_series = [True] + [False] * 59
        exit_series = [False] * 59 + [True]
        frame["ema_sma_cross_with_slope_entry_signal"] = entry_series
        frame["ema_sma_cross_with_slope_exit_signal"] = exit_series

    monkeypatch.setattr(
        strategy_module,
        "attach_ema_sma_cross_with_slope_signals",
        fake_attach_signals,
    )
    monkeypatch.setitem(
        strategy_module.BUY_STRATEGIES,
        "ema_sma_cross_with_slope",
        fake_attach_signals,
    )
    monkeypatch.setitem(
        strategy_module.SELL_STRATEGIES,
        "ema_sma_cross_with_slope",
        fake_attach_signals,
    )

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        passed_frame: pandas.DataFrame = kwargs["data"]
        captured_details["columns"] = list(passed_frame.columns)
        return SimulationResult(trades=[], total_profit=0.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)

    original_evaluate = strategy_module.evaluate_combined_strategy

    def wrapped_evaluate(
        data_directory: Path,
        buy_strategy_name: str,
        sell_strategy_name: str,
        **kwargs: object,
    ) -> StrategyMetrics:
        captured_details["strategies"] = (buy_strategy_name, sell_strategy_name)
        return original_evaluate(
            data_directory, buy_strategy_name, sell_strategy_name, **kwargs
        )

    monkeypatch.setattr(
        manage_module.strategy,
        "evaluate_combined_strategy",
        wrapped_evaluate,
    )
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)
    monkeypatch.setattr(manage_module, "STOCK_DATA_DIRECTORY", tmp_path)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate start=2020-01-01 dollar_volume>0 "
        "ema_sma_cross_with_slope_40 "
        "ema_sma_cross_with_slope_50"
    )

    assert captured_details["strategies"] == (
        "ema_sma_cross_with_slope_40",
        "ema_sma_cross_with_slope_50",
    )
    assert "ema_sma_cross_with_slope_40_entry_signal" in captured_details["columns"]
    assert "ema_sma_cross_with_slope_50_exit_signal" in captured_details["columns"]


def test_start_simulate_reports_missing_slope_bound() -> None:
    """The command should report a missing slope bound in strategy names."""

    import stock_indicator.manage as manage_module

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        "start_simulate dollar_volume>0 "
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
        "start_simulate dollar_volume>0 "
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
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        minimum_average_dollar_volume_ratio: float | None = None,
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
        minimum_average_dollar_volume_ratio: float | None = None,
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
        minimum_average_dollar_volume_ratio: float | None = None,
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
        minimum_average_dollar_volume: float | None,
        top_dollar_volume_rank: int | None = None,
        minimum_average_dollar_volume_ratio: float | None = None,
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
    shell.onecmd("start_simulate dollar_volume>0 noop_5 noop_10")
    assert "unsupported strategies" not in output_buffer.getvalue()


def test_update_sector_data_without_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command should rebuild data using the last configuration."""
    import stock_indicator.manage as manage_module

    call_flags = {"update_called": False, "report_called": False}

    def fake_update_latest_dataset() -> pandas.DataFrame:
        call_flags["update_called"] = True
        return pandas.DataFrame({"ticker": ["AAA"], "ff48": [1]})

    def fake_generate_coverage_report(data_frame: pandas.DataFrame) -> str:
        call_flags["report_called"] = True
        return "report"

    monkeypatch.setattr(
        manage_module.pipeline, "update_latest_dataset", fake_update_latest_dataset
    )
    monkeypatch.setattr(
        manage_module.pipeline, "generate_coverage_report", fake_generate_coverage_report
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("update_sector_data")

    assert call_flags == {"update_called": True, "report_called": True}
    assert output_buffer.getvalue() == "report\n"


def test_update_sector_data_with_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command should build data when given explicit sources."""
    import stock_indicator.manage as manage_module

    call_arguments: dict[str, object] = {}

    def fake_build_sector_classification_dataset(
        mapping_url: str, output_path: Path
    ) -> pandas.DataFrame:
        call_arguments["mapping"] = mapping_url
        call_arguments["output"] = output_path
        return pandas.DataFrame({"ticker": ["AAA"], "ff48": [1]})

    def fake_generate_coverage_report(data_frame: pandas.DataFrame) -> str:
        return "ok"

    monkeypatch.setattr(
        manage_module.pipeline,
        "build_sector_classification_dataset",
        fake_build_sector_classification_dataset,
    )
    monkeypatch.setattr(
        manage_module.pipeline, "generate_coverage_report", fake_generate_coverage_report
    )

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd(
        f"update_sector_data --ff-map-url=http://map {tmp_path/'out.parquet'}"
    )

    assert call_arguments == {
        "mapping": "http://map",
        "output": tmp_path / "out.parquet",
    }
    assert output_buffer.getvalue() == "ok\n"


def test_start_simulate_creates_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command should write completed trades to a CSV file."""
    import stock_indicator.manage as manage_module

    from stock_indicator.strategy import StrategyMetrics, TradeDetail

    open_trade_detail = TradeDetail(
        date=pandas.Timestamp("2024-01-02"),
        symbol="AAA",
        action="open",
        price=10.0,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
        price_concentration_score=1.0,
        near_price_volume_ratio=0.5,
        above_price_volume_ratio=0.3,
        histogram_node_count=2,
    )
    close_trade_detail = TradeDetail(
        date=pandas.Timestamp("2024-01-05"),
        symbol="AAA",
        action="close",
        price=12.0,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
        result="win",
        percentage_change=0.2,
    )
    metrics = StrategyMetrics(
        total_trades=0,
        win_rate=0.0,
        mean_profit_percentage=0.0,
        profit_percentage_standard_deviation=0.0,
        mean_loss_percentage=0.0,
        loss_percentage_standard_deviation=0.0,
        mean_holding_period=0.0,
        holding_period_standard_deviation=0.0,
        maximum_concurrent_positions=1,
        maximum_drawdown=0.0,
        final_balance=0.0,
        compound_annual_growth_rate=0.0,
        annual_returns={2024: 0.0},
        annual_trade_counts={2024: 0},
        trade_details_by_year={2024: [open_trade_detail, close_trade_detail]},
    )

    def fake_evaluate(*_: object, **__: object) -> StrategyMetrics:
        return metrics

    monkeypatch.setattr(
        manage_module.strategy, "evaluate_combined_strategy", fake_evaluate
    )
    monkeypatch.chdir(tmp_path)

    shell = manage_module.StockShell(stdout=io.StringIO())
    shell.onecmd(
        "start_simulate start=2024-01-01 dollar_volume>1 ema_sma_cross ema_sma_cross"
    )

    result_directory = tmp_path / "logs" / "simulate_result"
    csv_files = list(result_directory.glob("simulation_*.csv"))
    assert len(csv_files) == 1
    data_frame = pandas.read_csv(csv_files[0])
    assert list(data_frame.columns) == [
        "year",
        "entry_date",
        "concurrent_position_index",
        "symbol",
        "price_concentration_score",
        "near_price_volume_ratio",
        "above_price_volume_ratio",
        "histogram_node_count",
        "exit_date",
        "result",
        "percentage_change",
    ]
