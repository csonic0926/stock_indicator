"""Helpers for auditing live entries against refreshed market data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas

from . import place_tp_sl, strategy


REASON_DOLLAR_VOLUME_RANK_FLIP = "dollar_volume_rank_flip"
REASON_ENTRY_FEATURE_FLIP = "entry_feature_flip"
REASON_STILL_VALID = "still_valid"

LEDGER_FIELD_NAMES = [
    "detect_date",
    "symbol",
    "bucket",
    "strategy_id",
    "entry_date",
    "entry_price",
    "close_date",
    "close_price",
    "qty",
    "realized_pct",
    "orig_rank",
    "new_rank",
    "reason",
    "cache_latest_date",
]


@dataclass(frozen=True)
class ReevalResult:
    """Result of recomputing one accepted entry on refreshed data."""

    in_universe: bool
    pattern_fire: bool
    new_rank: int | None
    reason: str


@dataclass(frozen=True)
class FutuPosition:
    """Read-only Futu position fields needed by the cancellation ledger."""

    quantity: int | None
    cost_price: float | None


def reevaluate_entry_signal(
    config: Any,
    symbol: str,
    bucket_label: str,
    entry_date: str,
    data_directory: Path,
    allowed_symbols: set[str] | None,
    ff12_data_path: Path | None,
) -> ReevalResult:
    """Recompute the original entry signal using the current data cache.

    The call into :func:`strategy.compute_signals_for_date` mirrors the live
    ``multi_bucket_today.compute_today_signals`` Step B call so this audit
    checks the same production entry universe and same-day signal convention.
    """

    if bucket_label not in config.bucket_definitions:
        raise ValueError(f"unknown bucket label: {bucket_label}")

    bucket_def = config.bucket_definitions[bucket_label]
    evaluation_date = pandas.Timestamp(entry_date)
    with strategy.override_ff12_group_source_path(ff12_data_path):
        signals = strategy.compute_signals_for_date(
            data_directory=data_directory,
            evaluation_date=evaluation_date,
            buy_strategy_name=bucket_def.buy_strategy_name,
            sell_strategy_name=bucket_def.sell_strategy_name,
            minimum_average_dollar_volume=bucket_def.minimum_average_dollar_volume,
            top_dollar_volume_rank=bucket_def.top_dollar_volume_rank,
            maximum_symbols_per_group=bucket_def.maximum_symbols_per_group,
            minimum_average_dollar_volume_ratio=bucket_def.minimum_average_dollar_volume_ratio,
            allowed_symbols=allowed_symbols,
            skipped_fama_french_groups=bucket_def.skipped_fama_french_groups,
            use_unshifted_signals=True,
            additional_above_ranges=bucket_def.additional_above_ranges,
            exit_alpha_factor=bucket_def.exit_alpha_factor,
        )

    normalized_symbol = normalize_symbol_for_state(symbol)
    new_rank: int | None = None
    for filtered_symbol_rank, filtered_entry in enumerate(
        signals.get("filtered_symbols", [])
    ):
        filtered_symbol = (
            filtered_entry[0]
            if isinstance(filtered_entry, tuple)
            else filtered_entry
        )
        if str(filtered_symbol).upper() == normalized_symbol:
            new_rank = filtered_symbol_rank
            break

    in_universe = new_rank is not None
    pattern_fire = normalized_symbol in {
        str(entry_symbol).upper()
        for entry_symbol in signals.get("entry_signals", [])
    }
    if not in_universe:
        reason = REASON_DOLLAR_VOLUME_RANK_FLIP
    elif not pattern_fire:
        reason = REASON_ENTRY_FEATURE_FLIP
    else:
        reason = REASON_STILL_VALID

    return ReevalResult(
        in_universe=in_universe,
        pattern_fire=pattern_fire,
        new_rank=new_rank,
        reason=reason,
    )


def normalize_symbol_for_state(symbol: str) -> str:
    """Normalize command-line and Futu-form symbols to state ledger symbols."""

    normalized_symbol = symbol.strip().upper()
    if normalized_symbol.startswith("US."):
        normalized_symbol = normalized_symbol[3:]
    return normalized_symbol


def load_futu_position_for_symbol(symbol: str) -> FutuPosition | None:
    """Return the live Futu position for ``symbol`` without placing orders."""

    from futu import (  # type: ignore[import-not-found]
        OpenSecTradeContext,
        SecurityFirm,
        TrdEnv,
        TrdMarket,
    )

    normalized_symbol = normalize_symbol_for_state(symbol)
    target_code = f"US.{normalized_symbol}"
    trade_context = OpenSecTradeContext(
        host="127.0.0.1",
        port=11111,
        filter_trdmarket=TrdMarket.US,
        security_firm=SecurityFirm.FUTUSECURITIES,
    )
    try:
        return_code, position_data = trade_context.position_list_query(
            trd_env=TrdEnv.REAL
        )
        if return_code != 0:
            raise RuntimeError(f"failed to query Futu positions: {position_data}")
        positions = place_tp_sl._load_futu_positions(position_data)
    finally:
        close_method = getattr(trade_context, "close", None)
        if close_method is not None:
            close_method()

    position = positions.get(target_code)
    if position is None:
        return None
    return FutuPosition(
        quantity=int(position["qty"]) if position.get("qty") is not None else None,
        cost_price=(
            float(position["cost_price"])
            if position.get("cost_price") is not None
            else None
        ),
    )


def ledger_contains_symbol_entry(
    ledger_path: Path,
    *,
    symbol: str,
    entry_date: str,
) -> bool:
    """Return whether the cancellation ledger already has this entry."""

    if not ledger_path.exists():
        return False
    with ledger_path.open("r", newline="", encoding="utf-8") as ledger_file:
        reader = csv.DictReader(ledger_file)
        for row in reader:
            if (
                normalize_symbol_for_state(row.get("symbol", ""))
                == normalize_symbol_for_state(symbol)
                and row.get("entry_date") == entry_date
            ):
                return True
    return False


def append_cancellation_ledger_row(
    ledger_path: Path,
    row: dict[str, Any],
) -> None:
    """Append one cancellation row to the data-revision ledger."""

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_exists = ledger_path.exists()
    with ledger_path.open("a", newline="", encoding="utf-8") as ledger_file:
        writer = csv.DictWriter(ledger_file, fieldnames=LEDGER_FIELD_NAMES)
        if not ledger_exists:
            writer.writeheader()
        writer.writerow(
            {
                field_name: row.get(field_name, "")
                for field_name in LEDGER_FIELD_NAMES
            }
        )
