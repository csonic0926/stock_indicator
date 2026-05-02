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
    near_price_volume_ratio: float | None = None,
    above_price_volume_ratio: float | None = None,
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
        near_price_volume_ratio=near_price_volume_ratio,
        above_price_volume_ratio=above_price_volume_ratio,
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

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
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


def test_evict_oldest_keeps_evicted_exit_from_far_future_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Far-future close events must not overwrite an already evicted trade."""

    first_trade, first_details = _build_trade(
        "2024-01-01",
        "2024-01-12",
        symbol="AAA",
    )
    first_trade = strategy.replace(
        first_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.01, 0.00, 0.02),
        ],
    )
    second_trade, second_details = _build_trade(
        "2024-01-08",
        "2024-01-12",
        symbol="BBB",
    )
    second_trade = strategy.replace(
        second_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-09"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.01, 0.00, 0.00),
        ],
    )

    artifacts = _build_artifacts(
        [
            (first_trade, first_details),
            (second_trade, second_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(strategy, "_generate_strategy_evaluation_artifacts", fake_generate)
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            fixed_sl=0.03,
            evict_oldest=True,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]
    first_close = next(
        detail for detail in close_details if detail.symbol == "AAA"
    )

    assert first_close.date == pandas.Timestamp("2024-01-08")
    assert first_close.exit_reason == "evicted"
    assert metrics.overall_metrics.maximum_concurrent_positions == 1


def test_evict_oldest_refreshes_same_symbol_signal_age_without_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-symbol signals refresh eviction age without adding a new position."""

    first_signal_trade, first_signal_details = _build_trade(
        "2024-01-01",
        "2024-01-08",
        entry_price=10.0,
        exit_price=12.0,
        profit=2.0,
        symbol="AAA",
    )
    first_signal_trade = strategy.replace(
        first_signal_trade,
        exit_reason="reentry",
        bar_excursions=[
            (pandas.Timestamp("2024-01-02"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-03"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-04"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-05"), 0.10, 0.00, 0.00),
            (pandas.Timestamp("2024-01-08"), 0.20, 0.10, 0.20),
        ],
    )
    refreshed_signal_trade, refreshed_signal_details = _build_trade(
        "2024-01-08",
        "2024-01-12",
        entry_price=12.0,
        exit_price=14.0,
        profit=2.0,
        symbol="AAA",
    )
    refreshed_signal_trade = strategy.replace(
        refreshed_signal_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-09"), 0.25, 0.00, 0.00),
            (pandas.Timestamp("2024-01-10"), 0.25, 0.00, 0.00),
            (pandas.Timestamp("2024-01-11"), 0.25, 0.00, 0.00),
            (pandas.Timestamp("2024-01-12"), 0.25, 0.00, 0.00),
        ],
    )
    competing_trade, competing_details = _build_trade(
        "2024-01-10",
        "2024-01-12",
        symbol="BBB",
    )
    competing_trade = strategy.replace(
        competing_trade,
        bar_excursions=[
            (pandas.Timestamp("2024-01-11"), 0.01, 0.00, 0.00),
            (pandas.Timestamp("2024-01-12"), 0.01, 0.00, 0.00),
        ],
    )

    artifacts = _build_artifacts(
        [
            (first_signal_trade, first_signal_details),
            (refreshed_signal_trade, refreshed_signal_details),
            (competing_trade, competing_details),
        ],
    )

    def fake_generate(
        *args: object,
        **kwargs: object,
    ) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate,
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
        ),
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        minimum_holding_bars=5,
        adaptive_tp_sl=strategy.AdaptiveTPSLConfig(
            fixed_sl=0.03,
            evict_oldest=True,
        ),
    )

    close_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
        if detail.action == "close"
    ]

    assert [detail.symbol for detail in close_details] == ["AAA"]
    assert close_details[0].date == pandas.Timestamp("2024-01-12")
    assert close_details[0].exit_reason == "end_of_data"
    assert close_details[0].max_favorable_excursion_pct == pytest.approx(0.5)
    assert metrics.overall_metrics.total_trades == 1
    assert metrics.overall_metrics.maximum_concurrent_positions == 1


def test_run_complex_simulation_assigns_global_position_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global position counts should reflect open positions across all sets."""

    trade_a = _build_trade("2024-01-01", "2024-01-04", symbol="AAA")
    trade_b = _build_trade("2024-01-02", "2024-01-05", symbol="BBB")

    artifacts_a = _build_artifacts([trade_a])
    artifacts_b = _build_artifacts([trade_b])

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

    overall_details = [
        detail
        for detail_list in metrics.overall_metrics.trade_details_by_year.values()
        for detail in detail_list
    ]
    assert overall_details, "expected trade details for global count verification"

    def find_count(symbol: str, action: str, date: str) -> int | None:
        for detail in overall_details:
            if (
                detail.symbol == symbol
                and detail.action == action
                and detail.date == pandas.Timestamp(date)
            ):
                return detail.global_concurrent_position_count
        return None

    assert find_count("AAA", "open", "2024-01-01") == 1
    assert find_count("BBB", "open", "2024-01-02") == 2
    assert find_count("AAA", "close", "2024-01-04") == 1
    assert find_count("BBB", "close", "2024-01-05") == 0


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


def test_run_complex_simulation_prioritizes_low_above_ratio_for_s4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set definitions linked to s4 favor lower above-ratio entries."""

    lower_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-05",
        symbol="LOW",
        above_price_volume_ratio=0.5,
    )
    higher_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-06",
        symbol="HIGH",
        above_price_volume_ratio=1.2,
    )

    artifacts = _build_artifacts([lower_ratio_trade, higher_ratio_trade])

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy, "_generate_strategy_evaluation_artifacts", fake_generate
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
            strategy_identifier="s4",
        )
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
    )

    assert metrics.metrics_by_set["A"].total_trades == 1
    entry_details = [
        detail
        for detail in metrics.metrics_by_set["A"].trade_details_by_year.get(2024, [])
        if detail.action == "open"
    ]
    assert len(entry_details) == 1
    assert entry_details[0].symbol == "LOW"


def test_run_complex_simulation_prioritizes_low_near_ratio_for_s6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set definitions linked to s6 favor lower near-ratio entries."""

    higher_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-05",
        symbol="HIGH",
        near_price_volume_ratio=0.8,
    )
    lower_ratio_trade = _build_trade(
        "2024-01-01",
        "2024-01-06",
        symbol="LOW",
        near_price_volume_ratio=0.2,
    )

    artifacts = _build_artifacts([higher_ratio_trade, lower_ratio_trade])

    def fake_generate(*args: object, **kwargs: object) -> strategy.StrategyEvaluationArtifacts:
        return artifacts

    monkeypatch.setattr(
        strategy, "_generate_strategy_evaluation_artifacts", fake_generate
    )
    _stub_metrics_functions(monkeypatch)

    definitions = {
        "A": strategy.ComplexStrategySetDefinition(
            label="A",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
            strategy_identifier="s6",
        )
    }

    metrics = strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
    )

    assert metrics.metrics_by_set["A"].total_trades == 1
    entry_details = [
        detail
        for detail in metrics.metrics_by_set["A"].trade_details_by_year.get(2024, [])
        if detail.action == "open"
    ]
    assert len(entry_details) == 1
    assert entry_details[0].symbol == "LOW"
