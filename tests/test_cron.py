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
    argument_line = "dollar_volume>2.4%,-0.2% ema_sma_cross ema_sma_cross 1.0"
    (
        base_ratio,
        increment_ratio,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
    ) = cron.parse_daily_task_arguments(argument_line)
    assert base_ratio == pytest.approx(0.024)
    assert increment_ratio == pytest.approx(-0.002)
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





