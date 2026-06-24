"""Tests for data-revision signal audit and cancellation ledger commands."""

from __future__ import annotations

import csv
import io
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas
import pytest

from stock_indicator import data_revision_audit, daily_job, manage, strategy


class FakeFutuTradeContext:
    """Minimal Futu trade context for read-only position query tests."""

    def __init__(self, positions: list[dict[str, Any]]) -> None:
        self.positions = positions
        self.is_closed = False

    def position_list_query(self, trd_env: Any = None) -> tuple[int, pandas.DataFrame]:
        """Return deterministic position data."""

        return 0, pandas.DataFrame(
            self.positions,
            columns=["code", "qty", "cost_price"],
        )

    def close(self) -> None:
        """Mark the fake context closed."""

        self.is_closed = True


def _install_fake_futu_module(
    monkeypatch: pytest.MonkeyPatch,
    fake_context: FakeFutuTradeContext,
) -> None:
    """Install a deterministic futu module for ledger command tests."""

    fake_futu_module = types.SimpleNamespace(
        OpenSecTradeContext=lambda **_keyword_arguments: fake_context,
        SecurityFirm=types.SimpleNamespace(FUTUSECURITIES="FUTUSECURITIES"),
        TrdEnv=types.SimpleNamespace(REAL="REAL"),
        TrdMarket=types.SimpleNamespace(US="US"),
    )
    monkeypatch.setitem(sys.modules, "futu", fake_futu_module)


def _build_audit_config() -> Any:
    """Return a small config object with one bucket definition."""

    bucket_definition = strategy.ComplexStrategySetDefinition(
        label="fish_tail_production",
        buy_strategy_name="buy_strategy",
        sell_strategy_name="sell_strategy",
        strategy_identifier="fish_tail_blow_off_top",
        minimum_average_dollar_volume=10.0,
        minimum_average_dollar_volume_ratio=0.01,
        top_dollar_volume_rank=5,
        maximum_symbols_per_group=2,
        additional_above_ranges=[(0.3, 0.35)],
        exit_alpha_factor=1.5,
        skipped_fama_french_groups={12},
    )
    return types.SimpleNamespace(
        bucket_definitions={bucket_definition.label: bucket_definition}
    )


def _run_reevaluation_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    signals: dict[str, list[Any]],
) -> data_revision_audit.ReevalResult:
    """Run one reevaluation case with a mocked strategy result."""

    config = _build_audit_config()
    ff12_data_path = tmp_path / "ff12.parquet"
    recorded_override_paths: list[Path | None] = []
    recorded_signal_call: dict[str, Any] = {}

    @contextmanager
    def fake_override_ff12_group_source_path(
        sector_data_path: Path | None,
    ):
        recorded_override_paths.append(sector_data_path)
        yield

    def fake_compute_signals_for_date(**keyword_arguments: Any) -> dict[str, Any]:
        recorded_signal_call.update(keyword_arguments)
        return signals

    monkeypatch.setattr(
        data_revision_audit.strategy,
        "override_ff12_group_source_path",
        fake_override_ff12_group_source_path,
    )
    monkeypatch.setattr(
        data_revision_audit.strategy,
        "compute_signals_for_date",
        fake_compute_signals_for_date,
    )

    result = data_revision_audit.reevaluate_entry_signal(
        config,
        "CL",
        "fish_tail_production",
        "2026-06-10",
        tmp_path,
        {"CL", "AAA"},
        ff12_data_path,
    )

    bucket_definition = config.bucket_definitions["fish_tail_production"]
    assert recorded_override_paths == [ff12_data_path]
    assert recorded_signal_call["data_directory"] == tmp_path
    assert recorded_signal_call["evaluation_date"] == pandas.Timestamp(
        "2026-06-10"
    )
    assert recorded_signal_call["buy_strategy_name"] == "buy_strategy"
    assert recorded_signal_call["sell_strategy_name"] == "sell_strategy"
    assert recorded_signal_call["minimum_average_dollar_volume"] == 10.0
    assert recorded_signal_call["top_dollar_volume_rank"] == 5
    assert recorded_signal_call["maximum_symbols_per_group"] == 2
    assert recorded_signal_call["minimum_average_dollar_volume_ratio"] == 0.01
    assert recorded_signal_call["allowed_symbols"] == {"CL", "AAA"}
    assert recorded_signal_call["skipped_fama_french_groups"] == {12}
    assert recorded_signal_call["use_unshifted_signals"] is True
    assert recorded_signal_call["additional_above_ranges"] == (
        bucket_definition.additional_above_ranges
    )
    assert recorded_signal_call["exit_alpha_factor"] == 1.5
    return result


def test_reevaluate_entry_signal_reports_rank_flip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing refreshed universe membership should report rank flip."""

    result = _run_reevaluation_case(
        monkeypatch,
        tmp_path,
        {"filtered_symbols": [("AAA", 1)], "entry_signals": []},
    )

    assert result.in_universe is False
    assert result.pattern_fire is False
    assert result.new_rank is None
    assert result.reason == data_revision_audit.REASON_DOLLAR_VOLUME_RANK_FLIP


def test_reevaluate_entry_signal_reports_feature_flip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Universe membership without refreshed entry fire should report feature flip."""

    result = _run_reevaluation_case(
        monkeypatch,
        tmp_path,
        {"filtered_symbols": [("CL", 10), ("AAA", 1)], "entry_signals": []},
    )

    assert result.in_universe is True
    assert result.pattern_fire is False
    assert result.new_rank == 0
    assert result.reason == data_revision_audit.REASON_ENTRY_FEATURE_FLIP


def test_reevaluate_entry_signal_reports_still_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Universe membership plus refreshed entry fire should stay valid."""

    result = _run_reevaluation_case(
        monkeypatch,
        tmp_path,
        {
            "filtered_symbols": [("AAA", 1), ("CL", 10)],
            "entry_signals": ["CL"],
        },
    )

    assert result.in_universe is True
    assert result.pattern_fire is True
    assert result.new_rank == 1
    assert result.reason == data_revision_audit.REASON_STILL_VALID


def _build_runtime_context(tmp_path: Path) -> manage.MultiBucketDailyContext:
    """Return a minimal runtime context with one accepted CL entry."""

    return manage.MultiBucketDailyContext(
        config=_build_audit_config(),
        data_directory=tmp_path / "prices",
        allowed_symbols={"CL"},
        ff12_data_path=None,
        state_path=tmp_path / "adaptive_state.json",
        state={
            "accepted_entries": [
                {
                    "symbol": "CL",
                    "bucket": "fish_tail_production",
                    "strategy_id": "fish_tail_blow_off_top",
                    "entry_date": "2026-06-10",
                    "dollar_volume_rank": 3,
                }
            ]
        },
        symbol_first_eligible_trade_dates=None,
        setup_messages=[],
    )


def _install_common_cancel_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reevaluation_result: data_revision_audit.ReevalResult,
) -> Path:
    """Install shared filesystem and reevaluation mocks for cancel tests."""

    cron_log_directory = tmp_path / "cron_logs"
    stock_data_directory = tmp_path / "stock_data"
    stock_data_directory.mkdir()
    (stock_data_directory / f"{daily_job.SP500_SYMBOL}.csv").write_text(
        "Date,close\n2026-06-22,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manage, "CRON_LOG_DIRECTORY", cron_log_directory)
    monkeypatch.setattr(manage, "STOCK_DATA_DIRECTORY", stock_data_directory)
    monkeypatch.setattr(
        manage,
        "load_multi_bucket_daily_context",
        lambda _config_path, **_keyword_arguments: _build_runtime_context(tmp_path),
    )
    monkeypatch.setattr(
        manage.data_revision_audit,
        "reevaluate_entry_signal",
        lambda *unused_arguments: reevaluation_result,
    )
    return cron_log_directory / "data_revision_cancellations.csv"


def test_data_revision_cancel_writes_ledger_row_from_futu_position(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation command should record Futu cost, qty, and realized pct."""

    ledger_path = _install_common_cancel_mocks(
        monkeypatch,
        tmp_path,
        data_revision_audit.ReevalResult(
            in_universe=True,
            pattern_fire=False,
            new_rank=2,
            reason=data_revision_audit.REASON_ENTRY_FEATURE_FLIP,
        ),
    )
    fake_context = FakeFutuTradeContext(
        positions=[{"code": "US.CL", "qty": 53, "cost_price": 95.0}]
    )
    _install_fake_futu_module(monkeypatch, fake_context)

    output_buffer = io.StringIO()
    shell = manage.StockShell(stdout=output_buffer)
    shell.onecmd(
        "data_revision_cancel config.json CL "
        "--close-price 90 --close-date 2026-06-24"
    )

    assert fake_context.is_closed is True
    assert "[DATA_REVISION_CANCEL_LEDGERED]" in output_buffer.getvalue()
    with ledger_path.open("r", newline="", encoding="utf-8") as ledger_file:
        ledger_rows = list(csv.DictReader(ledger_file))

    assert len(ledger_rows) == 1
    ledger_row = ledger_rows[0]
    assert ledger_row["symbol"] == "CL"
    assert ledger_row["bucket"] == "fish_tail_production"
    assert ledger_row["strategy_id"] == "fish_tail_blow_off_top"
    assert ledger_row["entry_date"] == "2026-06-10"
    assert ledger_row["entry_price"] == "95.0"
    assert ledger_row["close_date"] == "2026-06-24"
    assert ledger_row["close_price"] == "90.0"
    assert ledger_row["qty"] == "53"
    assert float(ledger_row["realized_pct"]) == pytest.approx(-5.0 / 95.0)
    assert ledger_row["orig_rank"] == "3"
    assert ledger_row["new_rank"] == "2"
    assert ledger_row["reason"] == data_revision_audit.REASON_ENTRY_FEATURE_FLIP
    assert ledger_row["cache_latest_date"] == "2026-06-22"


def test_data_revision_cancel_refuses_still_valid_without_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A still-valid entry should not be ledgered unless forced."""

    ledger_path = _install_common_cancel_mocks(
        monkeypatch,
        tmp_path,
        data_revision_audit.ReevalResult(
            in_universe=True,
            pattern_fire=True,
            new_rank=3,
            reason=data_revision_audit.REASON_STILL_VALID,
        ),
    )
    monkeypatch.setattr(
        manage.data_revision_audit,
        "load_futu_position_for_symbol",
        lambda _symbol: pytest.fail("Futu query should not run"),
    )

    output_buffer = io.StringIO()
    shell = manage.StockShell(stdout=output_buffer)
    shell.onecmd("data_revision_cancel config.json CL --close-price 90")

    assert "still_valid" in output_buffer.getvalue()
    assert not ledger_path.exists()


def test_data_revision_cancel_refuses_duplicate_symbol_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Duplicate symbol and entry date should be idempotently rejected."""

    ledger_path = _install_common_cancel_mocks(
        monkeypatch,
        tmp_path,
        data_revision_audit.ReevalResult(
            in_universe=False,
            pattern_fire=False,
            new_rank=None,
            reason=data_revision_audit.REASON_DOLLAR_VOLUME_RANK_FLIP,
        ),
    )
    monkeypatch.setattr(
        manage.data_revision_audit,
        "load_futu_position_for_symbol",
        lambda _symbol: data_revision_audit.FutuPosition(
            quantity=53,
            cost_price=95.0,
        ),
    )

    first_output_buffer = io.StringIO()
    first_shell = manage.StockShell(stdout=first_output_buffer)
    first_shell.onecmd("data_revision_cancel config.json CL --close-price 90")
    second_output_buffer = io.StringIO()
    second_shell = manage.StockShell(stdout=second_output_buffer)
    second_shell.onecmd("data_revision_cancel config.json US.CL --close-price 89")

    with ledger_path.open("r", newline="", encoding="utf-8") as ledger_file:
        ledger_rows = list(csv.DictReader(ledger_file))

    assert len(ledger_rows) == 1
    assert "refusing duplicate" in second_output_buffer.getvalue()
