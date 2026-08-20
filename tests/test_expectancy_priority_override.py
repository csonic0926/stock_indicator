"""Tests for the soft-tier expectancy-driven bucket priority override."""

# TODO: review

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas
import pytest

from stock_indicator import manage, strategy
from test_expectancy_gate import (
    _build_artifacts,
    _build_trade,
    _stub_money_simulations,
)


def _gate_config_with_override(
    override_sigma_multiplier: float = 1.0,
    priorities: dict[str, int] | None = None,
) -> strategy.ExpectancyGateConfig:
    """Small-window gate: soft threshold -0.05, stop threshold -0.15."""

    return strategy.ExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=3.0,
        cold_start="open",
        priority_override=strategy.ExpectancyPriorityOverrideConfig(
            sigma_multiplier=override_sigma_multiplier,
            priorities=(
                priorities
                if priorities is not None
                else {"set_a": 2, "set_b": 1}
            ),
        ),
    )


def _parse_gate(raw_block: dict) -> strategy.ExpectancyGateConfig | None:
    return strategy.parse_expectancy_gate_config(raw_block)


def _valid_raw_block() -> dict:
    return {
        "enabled": True,
        "window": 2,
        "baseline_mean": 0.0,
        "baseline_sigma": 0.05,
        "sigma_multiplier": 3.0,
        "priority_override": {
            "enabled": True,
            "sigma_multiplier": 1.0,
            "priorities": {"set_a": 2, "set_b": 1},
        },
    }


def test_priority_override_parse_and_threshold() -> None:
    gate_config = _parse_gate(_valid_raw_block())
    assert gate_config is not None
    assert gate_config.priority_override is not None
    assert gate_config.priority_override.sigma_multiplier == 1.0
    assert gate_config.priority_override.priorities == {
        "set_a": 2,
        "set_b": 1,
    }
    assert gate_config.priority_override_threshold == pytest.approx(-0.05)


def test_priority_override_disabled_subblock_is_inert() -> None:
    raw_block = _valid_raw_block()
    raw_block["priority_override"]["enabled"] = False
    gate_config = _parse_gate(raw_block)
    assert gate_config is not None
    assert gate_config.priority_override is None
    assert gate_config.priority_override_threshold is None


@pytest.mark.parametrize(
    ("mutation", "expected_message_fragment"),
    [
        (
            lambda block: block["priority_override"].__setitem__(
                "sigma_multiplier", 3.0
            ),
            "smaller than expectancy_gate.sigma_multiplier",
        ),
        (
            lambda block: block["priority_override"].__setitem__(
                "sigma_multiplier", 0.0
            ),
            "must be > 0",
        ),
        (
            lambda block: block["priority_override"].__setitem__(
                "unexpected", 1
            ),
            "unknown key",
        ),
        (
            lambda block: block["priority_override"].__setitem__(
                "priorities", {"set_a": 1.5}
            ),
            "must be integers",
        ),
        (
            lambda block: block["priority_override"].__setitem__(
                "priorities", {}
            ),
            "non-empty",
        ),
    ],
)
def test_priority_override_validation_fails_closed(
    mutation,
    expected_message_fragment: str,
) -> None:
    raw_block = _valid_raw_block()
    mutation(raw_block)
    with pytest.raises(ValueError, match=expected_message_fragment):
        _parse_gate(raw_block)


def test_priority_override_requires_enabled_gate() -> None:
    raw_block = _valid_raw_block()
    raw_block["enabled"] = False
    with pytest.raises(ValueError, match="requires"):
        _parse_gate(raw_block)


def test_sensor_soft_tier_activates_between_soft_and_stop() -> None:
    gate_config = _gate_config_with_override()
    sensor = strategy.ExpectancyGateSensor(gate_config)
    decision_date = pandas.Timestamp("2024-02-01")

    seed_trades = [
        _build_trade("2024-01-02", "2024-01-05", -0.10, "AAA")[0],
        _build_trade("2024-01-08", "2024-01-11", -0.10, "BBB")[0],
    ]
    for order_index, seed_trade in enumerate(seed_trades):
        sensor.schedule_accepted_trade(
            id(seed_trade), seed_trade, order_index
        )
    # mean -0.10: below soft (-0.05) but above stop (-0.15).
    assert sensor.is_priority_override_active_before_entry(decision_date)
    assert not sensor.is_alarmed_before_entry(decision_date)


def test_sensor_soft_tier_stays_open_when_deque_not_full() -> None:
    gate_config = _gate_config_with_override()
    sensor = strategy.ExpectancyGateSensor(gate_config)
    lone_trade = _build_trade("2024-01-02", "2024-01-05", -0.40, "AAA")[0]
    sensor.schedule_accepted_trade(id(lone_trade), lone_trade, 0)
    assert not sensor.is_priority_override_active_before_entry(
        pandas.Timestamp("2024-02-01")
    )


def _run_two_bucket_fixture(
    monkeypatch: pytest.MonkeyPatch,
    seed_percentage_change: float,
    expectancy_gate: strategy.ExpectancyGateConfig | None,
    bucket_priority_overrides_by_month: dict[str, dict[str, int]] | None = None,
) -> strategy.ComplexSimulationMetrics:
    """Two buckets contest one slot on 2024-02-05 after two seed trades.

    Seed trades live in bucket set_a and exit before the contested day, so
    the sensor reading for 2024-02-05 is exactly ``seed_percentage_change``.
    """

    seed_trades = [
        _build_trade("2024-01-02", "2024-01-05", seed_percentage_change, "S1"),
        _build_trade("2024-01-08", "2024-01-11", seed_percentage_change, "S2"),
    ]
    contested_a = _build_trade("2024-02-05", "2024-02-09", 0.05, "AAA")
    contested_b = _build_trade("2024-02-05", "2024-02-09", 0.05, "BBB")
    artifacts_by_strategy_name = {
        "set_a": _build_artifacts(seed_trades + [contested_a]),
        "set_b": _build_artifacts([contested_b]),
    }

    def fake_generate_artifacts(*positional_arguments, **keyword_arguments):
        argument_values = list(positional_arguments) + list(
            keyword_arguments.values()
        )
        for strategy_name, artifacts in artifacts_by_strategy_name.items():
            if strategy_name in argument_values:
                return artifacts
        raise AssertionError("unexpected artifacts request")

    monkeypatch.setattr(
        strategy,
        "_generate_strategy_evaluation_artifacts",
        fake_generate_artifacts,
    )
    _stub_money_simulations(monkeypatch)
    definitions = {
        "set_a": strategy.ComplexStrategySetDefinition(
            label="set_a",
            buy_strategy_name="set_a",
            sell_strategy_name="set_a",
            entry_priority=1,
        ),
        "set_b": strategy.ComplexStrategySetDefinition(
            label="set_b",
            buy_strategy_name="set_b",
            sell_strategy_name="set_b",
            entry_priority=2,
        ),
    }
    return strategy.run_complex_simulation(
        Path("/tmp"),
        definitions,
        maximum_position_count=1,
        multi_bucket_mode=True,
        expectancy_gate=expectancy_gate,
        bucket_priority_overrides_by_month=bucket_priority_overrides_by_month,
    )


def _accepted_symbols(
    simulation_metrics: strategy.ComplexSimulationMetrics,
    label: str,
) -> set[str]:
    label_metrics = simulation_metrics.metrics_by_set.get(label)
    if label_metrics is None:
        return set()
    return {
        trade_detail.symbol
        for yearly_details in (
            label_metrics.trade_details_by_year or {}
        ).values()
        for trade_detail in yearly_details
        if trade_detail.action == "open"
    }


def test_healthy_sensor_keeps_default_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation_metrics = _run_two_bucket_fixture(
        monkeypatch,
        seed_percentage_change=0.10,
        expectancy_gate=_gate_config_with_override(),
    )
    assert "AAA" in _accepted_symbols(simulation_metrics, "set_a")
    assert "BBB" not in _accepted_symbols(simulation_metrics, "set_b")
    assert (
        simulation_metrics.expectancy_priority_override_trade_count == 0
    )
    assert (
        simulation_metrics.expectancy_priority_override_days_by_month == {}
    )


def test_degraded_sensor_flips_contested_slot_to_override_priorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation_metrics = _run_two_bucket_fixture(
        monkeypatch,
        seed_percentage_change=-0.10,
        expectancy_gate=_gate_config_with_override(),
    )
    assert "AAA" not in _accepted_symbols(simulation_metrics, "set_a")
    assert "BBB" in _accepted_symbols(simulation_metrics, "set_b")
    override_entry_details = [
        trade_detail
        for label in ("set_a", "set_b")
        for yearly_details in (
            simulation_metrics.metrics_by_set[label].trade_details_by_year
            or {}
        ).values()
        for trade_detail in yearly_details
        if trade_detail.action == "open"
        and trade_detail.expectancy_priority_override
    ]
    assert {
        trade_detail.symbol for trade_detail in override_entry_details
    } == {"BBB"}
    assert simulation_metrics.expectancy_priority_override_trade_count == 1
    assert simulation_metrics.expectancy_priority_override_days_by_month == {
        "2024-02": 1
    }


def test_stopped_sensor_ranks_phantom_entries_with_override_priorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the stop threshold: entries phantom, ranking still overridden."""

    simulation_metrics = _run_two_bucket_fixture(
        monkeypatch,
        seed_percentage_change=-0.40,
        expectancy_gate=_gate_config_with_override(),
    )
    assert "BBB" in _accepted_symbols(simulation_metrics, "set_b")
    assert "AAA" not in _accepted_symbols(simulation_metrics, "set_a")
    contested_entry_details = [
        trade_detail
        for yearly_details in (
            simulation_metrics.metrics_by_set["set_b"].trade_details_by_year
            or {}
        ).values()
        for trade_detail in yearly_details
        if trade_detail.action == "open" and trade_detail.symbol == "BBB"
    ]
    assert len(contested_entry_details) == 1
    assert contested_entry_details[0].expectancy_gated
    assert contested_entry_details[0].expectancy_priority_override


def test_degraded_sensor_without_override_block_keeps_default_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain_gate_config = strategy.ExpectancyGateConfig(
        window=2,
        baseline_mean=0.0,
        baseline_sigma=0.05,
        sigma_multiplier=3.0,
        cold_start="open",
    )
    simulation_metrics = _run_two_bucket_fixture(
        monkeypatch,
        seed_percentage_change=-0.10,
        expectancy_gate=plain_gate_config,
    )
    assert "AAA" in _accepted_symbols(simulation_metrics, "set_a")
    assert "BBB" not in _accepted_symbols(simulation_metrics, "set_b")


def test_union_with_month_keyed_risk_score_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM month trigger flips ranking even while the sensor is healthy."""

    simulation_metrics = _run_two_bucket_fixture(
        monkeypatch,
        seed_percentage_change=0.10,
        expectancy_gate=_gate_config_with_override(),
        bucket_priority_overrides_by_month={
            "2024-02": {"set_a": 2, "set_b": 1}
        },
    )
    assert "BBB" in _accepted_symbols(simulation_metrics, "set_b")
    assert "AAA" not in _accepted_symbols(simulation_metrics, "set_a")


def test_manage_aborts_when_override_priorities_do_not_cover_buckets(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "override_coverage.json"
    config_path.write_text(
        json.dumps(
            {
                "max_position_count": 1,
                "expectancy_gate": {
                    "enabled": True,
                    "window": 20,
                    "baseline_mean": 0.0087,
                    "baseline_sigma": 0.013,
                    "sigma_multiplier": 3.0,
                    "priority_override": {
                        "enabled": True,
                        "sigma_multiplier": 1.5,
                        "priorities": {
                            "bucket_one": 1,
                            "bucket_missing": 2,
                        },
                    },
                },
                "buckets": [
                    {
                        "label": "bucket_one",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                    },
                    {
                        "label": "bucket_two",
                        "strategy_id": "fish_head_vacuum_turn",
                        "dollar_volume_filter": (
                            "dollar_volume>0.02%,Top500,Pick5"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_buffer = io.StringIO()
    shell = manage.StockShell(stdout=output_buffer)
    shell.onecmd(f"multi_bucket_simulation {config_path}")
    output_text = output_buffer.getvalue()
    assert "must cover every configured bucket exactly" in output_text
    assert "bucket_missing" in output_text
    assert "bucket_two" in output_text
