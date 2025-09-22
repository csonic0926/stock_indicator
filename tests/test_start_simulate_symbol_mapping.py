"""Tests for symbol mapping in the ``start_simulate`` command."""

# TODO: review

from __future__ import annotations

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
from stock_indicator.strategy import StrategyMetrics, TradeDetail


def test_start_simulate_filters_pre_2014_googl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``start_simulate`` should drop GOOGL trades before 2014 while keeping GOOG trades."""

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
        **kwargs: object,
    ) -> StrategyMetrics:
        trade_details_by_year = {
            2013: [
                TradeDetail(
                    date=pandas.Timestamp("2013-01-02"),
                    symbol="GOOG",
                    action="open",
                    price=10.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2013-01-05"),
                    symbol="GOOG",
                    action="close",
                    price=11.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                    result="win",
                    percentage_change=0.1,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2013-02-02"),
                    symbol="GOOGL",
                    action="open",
                    price=20.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                ),
                TradeDetail(
                    date=pandas.Timestamp("2013-02-10"),
                    symbol="GOOGL",
                    action="close",
                    price=18.0,
                    simple_moving_average_dollar_volume=1.0,
                    total_simple_moving_average_dollar_volume=1.0,
                    simple_moving_average_dollar_volume_ratio=1.0,
                    result="lose",
                    percentage_change=-0.1,
                ),
            ]
        }
        return StrategyMetrics(
            total_trades=2,
            win_rate=0.5,
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
            annual_returns={2013: 0.1},
            annual_trade_counts={2013: 2},
            annual_profit_totals={2013: 0.0},
            trade_details_by_year=trade_details_by_year,
        )

    monkeypatch.setattr(strategy_module, "evaluate_combined_strategy", fake_evaluate)
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate dollar_volume>0 noop noop")
    output_string = output_buffer.getvalue()

    assert "GOOGL" not in output_string
    assert "GOOG" in output_string
    assert "Trades: 1," in output_string
    assert "Year 2013: 10.00%, profit: 0.00, trade: 1" in output_string
