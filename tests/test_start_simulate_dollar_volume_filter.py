"""Tests for ``start_simulate`` dollar volume filtering."""
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

import stock_indicator.manage as manage_module
import stock_indicator.strategy as strategy_module
from stock_indicator.strategy import SimulationResult, Trade


def test_start_simulate_retains_trade_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``start_simulate`` should keep trades when the dollar volume threshold rises."""
    # TODO: review

    date_index = pandas.date_range("2018-10-15", periods=61, freq="D")
    price_data_frame = pandas.DataFrame(
        {
            "Date": date_index,
            "open": [100.0] * len(date_index),
            "close": [100.0] * len(date_index),
            "volume": [40_000_000] * len(date_index),
        }
    )
    csv_path = tmp_path / "MSFT.csv"
    price_data_frame.to_csv(csv_path, index=False)

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

    def fake_simulate_trades(*args: object, **kwargs: object) -> SimulationResult:
        entry_date = pandas.Timestamp("2018-12-13")
        exit_date = pandas.Timestamp("2018-12-14")
        trade = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=100.0,
            exit_price=110.0,
            profit=10.0,
            holding_period=1,
        )
        return SimulationResult(trades=[trade], total_profit=10.0)

    monkeypatch.setattr(strategy_module, "simulate_trades", fake_simulate_trades)
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate dollar_volume>1%,-0.2% noop noop")
    first_output = output_buffer.getvalue()

    output_buffer.truncate(0)
    output_buffer.seek(0)
    shell.onecmd("start_simulate dollar_volume>2%,-0.2% noop noop")
    second_output = output_buffer.getvalue()

    expected_entry = "2018-12-13 MSFT open"
    assert expected_entry in first_output
    assert expected_entry in second_output
