"""Tests for complex simulation shared position management."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas
import pytest

from stock_indicator import strategy


def _build_trade(
    entry_date: str,
    exit_date: str,
    *,
    entry_price: float = 10.0,
    exit_price: float = 11.0,
    profit: float = 1.0,
    symbol: str = "AAA",
) -> tuple[strategy.Trade, tuple[strategy.TradeDetail, strategy.TradeDetail]]:
    """Create a trade and associated detail records for testing."""

    entry_timestamp = pandas.Timestamp(entry_date)
    exit_timestamp = pandas.Timestamp(exit_date)
    holding_period = (exit_timestamp - entry_timestamp).days
    trade = strategy.Trade(
        entry_date=entry_timestamp,
        exit_date=exit_timestamp,
        entry_price=entry_price,
        exit_price=exit_price,
        profit=profit,
        holding_period=holding_period,
        exit_reason="signal",
    )
    entry_detail = strategy.TradeDetail(
        date=entry_timestamp,
        symbol=symbol,
        action="open",
        price=entry_price,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
    )
    exit_detail = strategy.TradeDetail(
        date=exit_timestamp,
        symbol=symbol,
        action="close",
        price=exit_price,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
        result="win",
        percentage_change=profit / entry_price,
    )
    return trade, (entry_detail, exit_detail)


def _build_artifacts(
    trades_with_details: list[tuple[strategy.Trade, tuple[strategy.TradeDetail, strategy.TradeDetail]]],
) -> strategy.StrategyEvaluationArtifacts:
    """Create evaluation artifacts from prepared trades."""

    trades = [trade for trade, _ in trades_with_details]
    trade_symbol_lookup = {trade: detail_pair[0].symbol for trade, detail_pair in trades_with_details}
    closing_price_series_by_symbol = {
        detail_pair[0].symbol: pandas.Series(
            [detail_pair[0].price, detail_pair[1].price],
            index=[detail_pair[0].date, detail_pair[1].date],
        )
        for _, detail_pair in trades_with_details
    }
    trade_detail_pairs = {trade: detail_pair for trade, detail_pair in trades_with_details}
    simulation_results = [
        strategy.SimulationResult(
            trades=trades,
            total_profit=sum(current_trade.profit for current_trade in trades),
        )
    ]
    earliest_entry = min((trade.entry_date for trade in trades), default=None)
    return strategy.StrategyEvaluationArtifacts(
        trades=trades,
        simulation_results=simulation_results,
        trade_symbol_lookup=trade_symbol_lookup,
        closing_price_series_by_symbol=closing_price_series_by_symbol,
        trade_detail_pairs=trade_detail_pairs,
        simulation_start_date=earliest_entry,
    )


def _stub_metrics_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[int]]:
    """Replace expensive metric helpers with deterministic stubs."""

    call_records: dict[str, list[int]] = {
        "simulate_portfolio_balance": [],
        "calculate_max_drawdown": [],
    }

    monkeypatch.setattr(
        strategy, "calculate_annual_returns", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        strategy, "calculate_annual_trade_counts", lambda trades: {}
    )

    def fake_simulate_portfolio_balance(
        trades: list[strategy.Trade],
        starting_cash: float,
        maximum_position_count: int,
        *args: object,
        **kwargs: object,
    ) -> float:
        call_records["simulate_portfolio_balance"].append(maximum_position_count)
        return float(starting_cash)

    def fake_calculate_max_drawdown(
        trades: list[strategy.Trade],
        starting_cash: float,
        maximum_position_count: int,
        *args: object,
        **kwargs: object,
    ) -> float:
        call_records["calculate_max_drawdown"].append(maximum_position_count)
        return 0.0

    monkeypatch.setattr(
        strategy, "simulate_portfolio_balance", fake_simulate_portfolio_balance
    )
    monkeypatch.setattr(strategy, "calculate_max_drawdown", fake_calculate_max_drawdown)

    return call_records


def test_run_complex_simulation_enforces_shared_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared position cap should reject excess entries across strategy sets."""

    trade_a1 = _build_trade("2024-01-01", "2024-01-03", symbol="AAA")
    trade_a2 = _build_trade("2024-01-02", "2024-01-04", symbol="AAB")
    trade_b1 = _build_trade("2024-01-02", "2024-01-05", symbol="BAA")

    artifacts_a = _build_artifacts([trade_a1, trade_a2])
    artifacts_b = _build_artifacts([trade_b1])

    artifact_map = {
        "set_a": artifacts_a,
        "set_b": artifacts_b,
    }

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifact_map[str(buy_name)]

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=2,
    )

    assert metrics.metrics_by_set["A"].total_trades == 2
    assert metrics.metrics_by_set["B"].total_trades == 0
    assert metrics.overall_metrics.total_trades == 2
    assert metrics.overall_metrics.maximum_concurrent_positions == 2
    assert metrics.metrics_by_set["A"].annual_profit_totals == {
        2024: pytest.approx(2.0)
    }
    assert metrics.metrics_by_set["B"].annual_profit_totals == {}
    assert metrics.overall_metrics.annual_profit_totals == {
        2024: pytest.approx(2.0)
    }


def test_run_complex_simulation_allows_two_b_positions_when_limit_rounds_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set B should receive the rounded-up half of the shared position cap."""

    trade_a1 = _build_trade("2024-01-03", "2024-01-05", symbol="AAA")
    trade_b1 = _build_trade("2024-01-01", "2024-01-03", symbol="BAA")
    trade_b2 = _build_trade("2024-01-02", "2024-01-04", symbol="BAB")

    artifacts_a = _build_artifacts([trade_a1])
    artifacts_b = _build_artifacts([trade_b1, trade_b2])

    artifact_map = {
        "set_a": artifacts_a,
        "set_b": artifacts_b,
    }

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifact_map[str(buy_name)]

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=3,
    )

    assert metrics.metrics_by_set["B"].total_trades == 2
    assert metrics.overall_metrics.total_trades == 3
    assert metrics.metrics_by_set["A"].annual_profit_totals == {
        2024: pytest.approx(1.0)
    }
    assert metrics.metrics_by_set["B"].annual_profit_totals == {
        2024: pytest.approx(2.0)
    }
    assert metrics.overall_metrics.annual_profit_totals == {
        2024: pytest.approx(3.0)
    }


def test_run_complex_simulation_skips_b_when_global_open_count_reaches_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set B entries should be rejected once the shared open count hits its quota."""

    trade_a1 = _build_trade("2024-01-01", "2024-01-10", symbol="AAA")
    trade_a2 = _build_trade("2024-01-02", "2024-01-11", symbol="AAB")
    trade_b1 = _build_trade("2024-01-03", "2024-01-05", symbol="BAA")
    trade_b2 = _build_trade("2024-01-03", "2024-01-06", symbol="BAB")

    artifacts_a = _build_artifacts([trade_a1, trade_a2])
    artifacts_b = _build_artifacts([trade_b1, trade_b2])

    artifact_map = {
        "set_a": artifacts_a,
        "set_b": artifacts_b,
    }

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifact_map[str(buy_name)]

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=4,
    )

    assert metrics.metrics_by_set["A"].total_trades == 2
    assert metrics.metrics_by_set["B"].total_trades == 0
    assert metrics.overall_metrics.total_trades == 2
    assert metrics.metrics_by_set["A"].annual_profit_totals == {
        2024: pytest.approx(2.0)
    }
    assert metrics.metrics_by_set["B"].annual_profit_totals == {}
    assert metrics.overall_metrics.annual_profit_totals == {
        2024: pytest.approx(2.0)
    }


def test_run_complex_simulation_overall_metrics_use_global_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overall metric helpers should receive the shared position cap."""

    trade_a = _build_trade("2024-01-01", "2024-01-05", symbol="AAA")
    trade_b = _build_trade("2024-01-02", "2024-01-06", symbol="BBB")

    artifacts_a = _build_artifacts([trade_a])
    artifacts_b = _build_artifacts([trade_b])

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        buy_name = kwargs.get("buy_strategy_name") or args[1]
        return artifacts_a if buy_name == "set_a" else artifacts_b

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    call_records = _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
        "B": strategy.ComplexStrategySetDefinition(
            label="B",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
        ),
    }

    strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=3,
    )

    assert call_records["simulate_portfolio_balance"][-1] == 3
    assert call_records["calculate_max_drawdown"][-1] == 3
