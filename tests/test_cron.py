from pathlib import Path

import pandas
import pytest

from stock_indicator import cron
from stock_indicator import strategy


def test_run_daily_tasks_detects_signals(tmp_path, monkeypatch):
    """run_daily_tasks should return symbols with entry and exit signals."""
    symbol_list = ["TEST"]

    def fake_update_symbol_cache() -> None:
        return None

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        frame = pandas.DataFrame({"close": [1.0, 2.0, 3.0]})
        if cache_path is not None:
            frame.to_csv(cache_path)
        return frame

    def fake_strategy(price_history_frame: pandas.DataFrame) -> None:
        price_history_frame["fake_strategy_entry_signal"] = [False, False, True]
        price_history_frame["fake_strategy_exit_signal"] = [False, False, False]

    monkeypatch.setattr(cron, "update_symbol_cache", fake_update_symbol_cache)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)

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
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == 10000.0
    assert top_dollar_volume_rank is None
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_rank() -> None:
    """The parser should accept ranking-based dollar volume filters."""
    argument_line = "dollar_volume=5th ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume is None
    assert top_dollar_volume_rank == 5
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_accepts_threshold_and_rank() -> None:
    """The parser should accept combined threshold and ranking filters."""
    argument_line = "dollar_volume>100,5th ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == 100.0
    assert top_dollar_volume_rank == 5
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_parse_daily_task_arguments_parses_large_threshold_and_rank() -> None:
    """The parser should handle a large threshold combined with ranking."""
    # TODO: review
    argument_line = "dollar_volume>10000,6th ema_sma_cross ema_sma_cross"
    (
        minimum_average_dollar_volume,
        top_dollar_volume_rank,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert minimum_average_dollar_volume == 10000.0
    assert top_dollar_volume_rank == 6
    assert buy_strategy_name == "ema_sma_cross"
    assert sell_strategy_name == "ema_sma_cross"
    assert stop_loss_percentage == 1.0


def test_run_daily_tasks_skips_symbol_update_errors(monkeypatch):
    """run_daily_tasks should continue when the symbol cache update fails."""

    def failing_update() -> None:
        raise RuntimeError("network down")

    def fake_download_history(
        symbol: str, start: str, end: str
    ) -> pandas.DataFrame:
        return pandas.DataFrame({"close": [1.0], "volume": [100.0]})

    def fake_strategy(price_history_frame: pandas.DataFrame) -> None:
        price_history_frame["fake_strategy_entry_signal"] = [True]
        price_history_frame["fake_strategy_exit_signal"] = [False]

    monkeypatch.setattr(cron, "update_symbol_cache", failing_update)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)

    result = cron.run_daily_tasks(
        "fake_strategy",
        "fake_strategy",
        "2024-01-01",
        "2024-01-02",
        symbol_list=["TEST"],
        data_download_function=fake_download_history,
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
        frame = pandas.DataFrame({"close": [1.0] * 60, "volume": volume_values})
        if cache_path is not None:
            frame.to_csv(cache_path)
        return frame

    def fake_strategy(price_history_frame: pandas.DataFrame) -> None:
        entry_signals = [False] * 59 + [True]
        price_history_frame["fake_strategy_entry_signal"] = entry_signals
        price_history_frame["fake_strategy_exit_signal"] = [False] * 60

    monkeypatch.setattr(cron, "update_symbol_cache", fake_update_symbol_cache)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)

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
        frame = pandas.DataFrame({"close": [1.0] * 60, "volume": volume_values})
        if cache_path is not None:
            frame.to_csv(cache_path)
        return frame

    def fake_strategy(price_history_frame: pandas.DataFrame) -> None:
        entry_signals = [False] * 59 + [True]
        price_history_frame["fake_strategy_entry_signal"] = entry_signals
        price_history_frame["fake_strategy_exit_signal"] = [False] * 60

    monkeypatch.setattr(cron, "update_symbol_cache", fake_update_symbol_cache)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)

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


def test_run_daily_tasks_limits_symbols_by_large_threshold_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_daily_tasks should process only the top six symbols above the threshold."""
    # TODO: review
    symbol_name_list = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]

    def fake_update_symbol_cache() -> None:
        return None

    volume_by_symbol_name = {
        "AAA": 11_000_000_000.0,
        "BBB": 12_000_000_000.0,
        "CCC": 13_000_000_000.0,
        "DDD": 14_000_000_000.0,
        "EEE": 15_000_000_000.0,
        "FFF": 16_000_000_000.0,
        "GGG": 17_000_000_000.0,
    }

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        volume_value = volume_by_symbol_name[symbol]
        data_frame = pandas.DataFrame(
            {"close": [1.0] * 60, "volume": [volume_value] * 60}
        )
        if cache_path is not None:
            data_frame.to_csv(cache_path)
        return data_frame

    def fake_strategy(price_history_frame: pandas.DataFrame) -> None:
        entry_signal_values = [False] * 59 + [True]
        price_history_frame["fake_strategy_entry_signal"] = entry_signal_values
        price_history_frame["fake_strategy_exit_signal"] = [False] * 60

    monkeypatch.setattr(cron, "update_symbol_cache", fake_update_symbol_cache)
    monkeypatch.setitem(strategy.SUPPORTED_STRATEGIES, "fake_strategy", fake_strategy)

    result = cron.run_daily_tasks(
        "fake_strategy",
        "fake_strategy",
        "2024-01-01",
        "2024-01-10",
        symbol_list=symbol_name_list,
        data_download_function=fake_download_history,
        data_directory=tmp_path,
        minimum_average_dollar_volume=10000,
        top_dollar_volume_rank=6,
    )

    assert len(result["entry_signals"]) == 6
    assert set(result["entry_signals"]) == {
        "BBB",
        "CCC",
        "DDD",
        "EEE",
        "FFF",
        "GGG",
    }
    assert result["exit_signals"] == []
