from pathlib import Path

import pandas
import pytest

from stock_indicator import cron
from stock_indicator import strategy


def test_run_daily_job_uses_production_universe_without_refresh() -> None:
    """The shell cron wrapper must not refresh candidate universe files."""

    script_text = Path("run_daily_job.sh").read_text(encoding="utf-8")
    yahoo_update_position = script_text.index("update_all_data_from_yf")
    daily_signal_position = script_text.index("multi_bucket_daily_signal")
    assert "update_universe_pipeline" not in script_text
    assert yahoo_update_position < daily_signal_position
    assert "YAHOO_CACHE_REFRESH_LOOKBACK_DAYS" in script_text
    assert "determine_latest_cached_market_date" in script_text
    assert "timedelta(days=183)" not in script_text


def test_run_daily_tasks_detects_signals(tmp_path, monkeypatch):
    """run_daily_tasks should return symbols with entry and exit signals."""
    symbol_list = ["TEST"]

    def fake_update_symbol_cache() -> None:
        return None

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        frame = pandas.DataFrame({"open": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0]})
        if cache_path is not None:
            frame.to_csv(cache_path)
        return frame

    def fake_strategy(
        price_history_frame: pandas.DataFrame, *, include_raw_signals: bool = False
    ) -> None:
        price_history_frame["fake_strategy_entry_signal"] = [False, False, True]
        price_history_frame["fake_strategy_exit_signal"] = [False, False, False]

    monkeypatch.setattr(cron, "update_symbol_cache", fake_update_symbol_cache)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(strategy, "load_ff12_groups_by_symbol", lambda: {"TEST": 12})
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)

    result = cron.run_daily_tasks(
        "fake_strategy",
        "fake_strategy",
        "2024-01-01",
        "2024-01-10",
        symbol_list=symbol_list,
        data_download_function=fake_download_history,
        data_directory=tmp_path,
    )

    assert result["entry_signals"] == symbol_list
    assert result["exit_signals"] == []
    saved_file_path = Path(tmp_path) / "TEST.csv"
    assert saved_file_path.exists()


def test_parse_daily_task_arguments_returns_expected_values():
    """parse_daily_task_arguments should extract values from argument string."""
    argument_line = "dollar_volume>10000 ema_sma_cross ema_sma_cross 1.0"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        _,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == 10000.0
    assert top_dollar_volume_rank is None
    assert maximum_symbols_per_group == 1
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_rank() -> None:
    """The parser should accept ranking-based dollar volume filters."""
    argument_line = "dollar_volume=5th ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        _,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume is None
    assert top_dollar_volume_rank == 5
    assert maximum_symbols_per_group == 1
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_threshold_and_rank() -> None:
    """The parser should accept combined threshold and ranking filters."""
    argument_line = "dollar_volume>100,5th ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        _,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == 100.0
    assert top_dollar_volume_rank == 5
    assert maximum_symbols_per_group == 1
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_percentage() -> None:
    """The parser should accept percentage-based dollar volume filters."""
    argument_line = "dollar_volume>2.41% ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        _,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == pytest.approx(0.0241)
    assert top_dollar_volume_rank is None
    assert maximum_symbols_per_group == 1
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_percentage_and_rank() -> None:
    """The parser should accept percentage filters combined with ranking."""
    argument_line = "dollar_volume>2.41%,5th ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        _,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == pytest.approx(0.0241)
    assert top_dollar_volume_rank == 5
    assert maximum_symbols_per_group == 1
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_pick_parameter() -> None:
    """The parser should extract the per-group pick count."""
    argument_line = "dollar_volume>2.41%,Top3,Pick2 ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        _,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == pytest.approx(0.0241)
    assert top_dollar_volume_rank == 3
    assert maximum_symbols_per_group == 2
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_ff12_other_group() -> None:
    """Group 12 is a valid Fama-French bucket."""
    argument_line = "group=12 dollar_volume>2.41%,Top3,Pick2 ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        allowed_fama_french_groups,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == pytest.approx(0.0241)
    assert top_dollar_volume_rank == 3
    assert maximum_symbols_per_group == 2
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0
    assert allowed_fama_french_groups == {12}


def test_run_daily_tasks_skips_symbol_update_errors(tmp_path, monkeypatch):
    """run_daily_tasks should continue when the symbol cache update fails."""

    def failing_update() -> None:
        raise RuntimeError("network down")

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        frame = pandas.DataFrame(
            {
                "Date": [pandas.Timestamp("2024-01-01")],
                "open": [1.0],
                "close": [1.0],
                "volume": [100.0],
            }
        )
        if cache_path is not None:
            frame.to_csv(cache_path, index=False)
        return frame

    def fake_strategy(
        price_history_frame: pandas.DataFrame, *, include_raw_signals: bool = False
    ) -> None:
        price_history_frame["fake_strategy_entry_signal"] = [True]
        price_history_frame["fake_strategy_exit_signal"] = [False]

    monkeypatch.setattr(cron, "update_symbol_cache", failing_update)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(strategy, "load_ff12_groups_by_symbol", lambda: {"TEST": 12})

    result = cron.run_daily_tasks(
        "fake_strategy",
        "fake_strategy",
        "2024-01-01",
        "2024-01-02",
        symbol_list=["TEST"],
        data_download_function=fake_download_history,
        data_directory=tmp_path,
    )

    assert result["entry_signals"] == ["TEST"]


def test_run_daily_tasks_honors_dollar_volume_rank(tmp_path, monkeypatch):
    """run_daily_tasks should process only the highest-ranked symbol."""

    symbol_list = ["AAA", "BBB"]

    def fake_update_symbol_cache() -> None:
        return None

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        if symbol == "AAA":
            volume_values = [1000.0] * 59 + [1.0]
        else:
            volume_values = [10.0] * 59 + [200.0]
        frame = pandas.DataFrame(
            {"open": [1.0] * 60, "close": [1.0] * 60, "volume": volume_values}
        )
        if cache_path is not None:
            frame.to_csv(cache_path)
        return frame

    def fake_strategy(
        price_history_frame: pandas.DataFrame, *, include_raw_signals: bool = False
    ) -> None:
        entry_signals = [False] * 59 + [True]
        price_history_frame["fake_strategy_entry_signal"] = entry_signals
        price_history_frame["fake_strategy_exit_signal"] = [False] * 60

    monkeypatch.setattr(cron, "update_symbol_cache", fake_update_symbol_cache)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(
        strategy,
        "load_ff12_groups_by_symbol",
        lambda: {"AAA": 1, "BBB": 2, "CCC": 3},
    )
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)

    result = cron.run_daily_tasks(
        "fake_strategy",
        "fake_strategy",
        "2024-01-01",
        "2024-01-10",
        symbol_list=symbol_list,
        data_download_function=fake_download_history,
        data_directory=tmp_path,
        top_dollar_volume_rank=1,
    )

    assert result["entry_signals"] == ["AAA"]
    assert result["exit_signals"] == []


def test_run_daily_tasks_honors_group_ratio_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_daily_tasks`` should honor group ratio and ranking."""

    volume_by_symbol = {"AAA": 2_000_000, "BBB": 100_000, "CCC": 5_000_000}
    date_index = pandas.date_range("2020-01-01", periods=60, freq="D")
    for symbol_name, volume_value in volume_by_symbol.items():
        pandas.DataFrame(
            {
                "Date": date_index,
                "open": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": [volume_value] * 60,
                "symbol": [symbol_name] * 60,
            }
        ).to_csv(tmp_path / f"{symbol_name}.csv", index=False)

    def fake_strategy(
        price_history_frame: pandas.DataFrame, *, include_raw_signals: bool = False
    ) -> None:
        entry_signals = [False] * 59 + [True]
        price_history_frame["fake_strategy_entry_signal"] = entry_signals
        price_history_frame["fake_strategy_exit_signal"] = [False] * 60

    monkeypatch.setattr(
        strategy, "BUY_STRATEGIES", {"fake_strategy": fake_strategy}
    )
    monkeypatch.setattr(
        strategy, "SELL_STRATEGIES", {"fake_strategy": fake_strategy}
    )
    monkeypatch.setattr(
        strategy, "SUPPORTED_STRATEGIES", {"fake_strategy": fake_strategy}
    )

    group_map = {"AAA": 1, "BBB": 1, "CCC": 2}
    monkeypatch.setattr(strategy, "load_ff12_groups_by_symbol", lambda: group_map)
    monkeypatch.setattr(cron, "load_ff12_groups_by_symbol", lambda: group_map)
    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())

    argument_line = "dollar_volume>1.6%,Top4 fake_strategy fake_strategy"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        _,
        allowed_groups,
    ) = cron.parse_daily_task_arguments(argument_line)
    signal_result = cron.run_daily_tasks(
        buy_strategy_name,
        sell_strategy_name,
        "2020-01-01",
        "2020-03-01",
        symbol_list=["AAA", "BBB", "CCC"],
        data_directory=tmp_path,
        minimum_average_dollar_volume=minimum_average_dollar_volume,
        top_dollar_volume_rank=top_dollar_volume_rank,
        allowed_fama_french_groups=allowed_groups,
        maximum_symbols_per_group=maximum_symbols_per_group,
    )

    entry_signal_symbols = signal_result["entry_signals"]
    assert "AAA" in entry_signal_symbols
    assert "BBB" not in entry_signal_symbols

    processed_symbol_names: list[str] = []

    def fake_simulate_trades(
        data: pandas.DataFrame, *args: object, **kwargs: object
    ) -> strategy.SimulationResult:
        processed_symbol_names.append(data["symbol"].iloc[0])
        trade = strategy.Trade(
            entry_date=data.index[-1],
            exit_date=data.index[-1],
            entry_price=float(data["open"].iloc[-1]),
            exit_price=float(data["close"].iloc[-1]),
            profit=0.0,
            holding_period=0,
        )
        return strategy.SimulationResult(trades=[trade], total_profit=0.0)

    monkeypatch.setattr(strategy, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(
        strategy,
        "simulate_portfolio_balance",
        lambda trades, starting_cash, maximum_position_count, withdraw_amount=0.0, **kwargs: starting_cash,
    )
    monkeypatch.setattr(
        strategy, "calculate_annual_returns", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        strategy, "calculate_annual_trade_counts", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        strategy, "calculate_max_drawdown", lambda *args, **kwargs: 0.0
    )

    strategy.evaluate_combined_strategy(
        tmp_path,
        "fake_strategy",
        "fake_strategy",
        minimum_average_dollar_volume_ratio=0.016,
        top_dollar_volume_rank=4,
        allowed_symbols={"AAA", "BBB", "CCC"},
    )

    assert set(entry_signal_symbols) == set(processed_symbol_names)


def test_run_daily_tasks_applies_combined_filters(tmp_path, monkeypatch):
    """run_daily_tasks should respect both threshold and rank filters."""
    symbol_list = ["AAA", "BBB", "CCC"]

    def fake_update_symbol_cache() -> None:
        return None

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        if symbol == "AAA":
            volume_values = [200_000_000.0] * 60
        elif symbol == "BBB":
            volume_values = [150_000_000.0] * 60
        else:
            volume_values = [50_000_000.0] * 60
        frame = pandas.DataFrame(
            {"open": [1.0] * 60, "close": [1.0] * 60, "volume": volume_values}
        )
        if cache_path is not None:
            frame.to_csv(cache_path)
        return frame

    def fake_strategy(
        price_history_frame: pandas.DataFrame, *, include_raw_signals: bool = False
    ) -> None:
        entry_signals = [False] * 59 + [True]
        price_history_frame["fake_strategy_entry_signal"] = entry_signals
        price_history_frame["fake_strategy_exit_signal"] = [False] * 60

    monkeypatch.setattr(cron, "update_symbol_cache", fake_update_symbol_cache)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.BUY_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setitem(strategy.SELL_STRATEGIES, "fake_strategy", fake_strategy)
    monkeypatch.setattr(strategy, "load_symbols_excluded_by_industry", lambda: set())
    monkeypatch.setattr(
        strategy,
        "load_ff12_groups_by_symbol",
        lambda: {"AAA": 1, "BBB": 2, "CCC": 3},
    )

    result = cron.run_daily_tasks(
        "fake_strategy",
        "fake_strategy",
        "2024-01-01",
        "2024-01-10",
        symbol_list=symbol_list,
        data_download_function=fake_download_history,
        data_directory=tmp_path,
        minimum_average_dollar_volume=100,
        top_dollar_volume_rank=1,
    )

    assert result["entry_signals"] == ["AAA"]
    assert result["exit_signals"] == []
