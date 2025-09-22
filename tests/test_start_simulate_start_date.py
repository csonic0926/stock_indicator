"""Tests for start date handling in the ``start_simulate`` command."""

# TODO: review

from __future__ import annotations

import io
from pathlib import Path

import pandas
import pytest

import stock_indicator.manage as manage_module
import stock_indicator.strategy as strategy_module
from stock_indicator.strategy import StrategyMetrics


def test_start_simulate_accepts_start_date(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``start_simulate`` should pass the supplied start date to evaluation."""

    recorded_arguments: dict[str, pandas.Timestamp | None] = {"start_date": None}
    start_date_called: dict[str, bool] = {"called": False}

    def fake_determine_start_date(data_directory: Path) -> str:  # pragma: no cover - defensive
        start_date_called["called"] = True
        return "1900-01-01"

    def fake_evaluate_combined_strategy(
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
            annual_profit_totals={},
        )

    monkeypatch.setattr(manage_module, "determine_start_date", fake_determine_start_date)
    monkeypatch.setattr(strategy_module, "evaluate_combined_strategy", fake_evaluate_combined_strategy)
    monkeypatch.setattr(strategy_module, "BUY_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(strategy_module, "SELL_STRATEGIES", {"noop": lambda frame: None})
    monkeypatch.setattr(manage_module, "DATA_DIRECTORY", tmp_path)

    output_buffer = io.StringIO()
    shell = manage_module.StockShell(stdout=output_buffer)
    shell.onecmd("start_simulate start=2020-05-06 dollar_volume>0 noop noop")

    assert recorded_arguments["start_date"] == pandas.Timestamp("2020-05-06")
    assert start_date_called["called"] is False
    assert "Simulation start date: 2020-05-06" in output_buffer.getvalue()

