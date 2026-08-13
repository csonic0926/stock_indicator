"""Persist and advance the production rolling-expectancy gate.

The simulator remains the semantic source of truth.  This module owns only
the live order-allocation sensor ledger: dashboard-confirmed entries are
registered here, and the daily cron advances their final adaptive outcomes
strictly before publishing the next allocation regime.
"""

from __future__ import annotations

# TODO: review

import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas

from stock_indicator import adaptive_tp_sl_virtual_trade_history, strategy


EXPECTANCY_GATE_STATE_SCHEMA_VERSION = 1


def parse_live_expectancy_gate_config(
    config_document: dict[str, Any],
) -> strategy.ExpectancyGateConfig | None:
    """Parse the live gate and validate exact soft-priority coverage."""

    gate_config = strategy.parse_expectancy_gate_config(
        config_document.get("expectancy_gate")
    )
    if gate_config is None or gate_config.priority_override is None:
        return gate_config

    raw_buckets = config_document.get("buckets")
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError(
            "expectancy_gate.priority_override requires a non-empty buckets array"
        )
    bucket_labels = {
        str(raw_bucket.get("label") or "")
        for raw_bucket in raw_buckets
        if isinstance(raw_bucket, dict)
    }
    priority_labels = set(gate_config.priority_override.priorities)
    missing_labels = sorted(bucket_labels - priority_labels)
    unknown_labels = sorted(priority_labels - bucket_labels)
    if missing_labels or unknown_labels:
        problem_parts: list[str] = []
        if missing_labels:
            problem_parts.append("missing=" + ",".join(missing_labels))
        if unknown_labels:
            problem_parts.append("unknown=" + ",".join(unknown_labels))
        raise ValueError(
            "expectancy_gate.priority_override.priorities must cover every "
            "configured bucket exactly (" + "; ".join(problem_parts) + ")"
        )
    return gate_config


def _config_snapshot(
    gate_config: strategy.ExpectancyGateConfig,
) -> dict[str, Any]:
    """Return the state-compatible semantic configuration snapshot."""

    priority_override = gate_config.priority_override
    return {
        "window": gate_config.window,
        "baseline_mean": gate_config.baseline_mean,
        "baseline_sigma": gate_config.baseline_sigma,
        "sigma_multiplier": gate_config.sigma_multiplier,
        "cold_start": gate_config.cold_start,
        "priority_override": (
            {
                "sigma_multiplier": priority_override.sigma_multiplier,
                "priorities": dict(priority_override.priorities),
            }
            if priority_override is not None
            else None
        ),
    }


def empty_expectancy_gate_state_document(
    gate_config: strategy.ExpectancyGateConfig,
) -> dict[str, Any]:
    """Return an empty, cold-start production sensor ledger."""

    return {
        "schema_version": EXPECTANCY_GATE_STATE_SCHEMA_VERSION,
        "config": _config_snapshot(gate_config),
        "last_evaluation_date": None,
        "next_acceptance_order": 0,
        "accepted_count_by_bucket": {},
        "accepted_trade_keys": [],
        "pending_trades": [],
        "rolling_outcomes": [],
        "previous_accepted_entry_was_gated": False,
        "closed_episodes": [],
        "expectancy_gated_trade_count": 0,
        "priority_override_entry_count": 0,
    }


def _validate_state_document(
    state_document: dict[str, Any],
    gate_config: strategy.ExpectancyGateConfig,
) -> None:
    """Reject corrupt or semantically incompatible live sensor state."""

    if state_document.get("schema_version") != EXPECTANCY_GATE_STATE_SCHEMA_VERSION:
        raise ValueError(
            "expectancy gate state has an unsupported schema_version"
        )
    if state_document.get("config") != _config_snapshot(gate_config):
        raise ValueError(
            "expectancy gate state config does not match production config"
        )
    list_fields = (
        "accepted_trade_keys",
        "pending_trades",
        "rolling_outcomes",
        "closed_episodes",
    )
    for field_name in list_fields:
        if not isinstance(state_document.get(field_name), list):
            raise ValueError(
                f"expectancy gate state field {field_name} must be a list"
            )
    if not isinstance(state_document.get("accepted_count_by_bucket"), dict):
        raise ValueError(
            "expectancy gate state field accepted_count_by_bucket must be an object"
        )
    try:
        next_acceptance_order = int(state_document["next_acceptance_order"])
    except (KeyError, TypeError, ValueError) as state_error:
        raise ValueError(
            "expectancy gate state next_acceptance_order must be an integer"
        ) from state_error
    if next_acceptance_order < 0:
        raise ValueError(
            "expectancy gate state next_acceptance_order must be non-negative"
        )
    last_evaluation_date = state_document.get("last_evaluation_date")
    if last_evaluation_date is not None:
        try:
            pandas.Timestamp(last_evaluation_date)
        except (TypeError, ValueError) as date_error:
            raise ValueError(
                "expectancy gate state last_evaluation_date is invalid"
            ) from date_error


def load_expectancy_gate_state(
    state_path: Path,
    gate_config: strategy.ExpectancyGateConfig,
    *,
    allow_missing: bool = True,
) -> dict[str, Any]:
    """Load live state, optionally allowing cron cold-start creation."""

    if not state_path.exists():
        if not allow_missing:
            raise ValueError("expectancy gate state is missing")
        return empty_expectancy_gate_state_document(gate_config)
    try:
        state_document = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as state_error:
        raise ValueError(
            f"failed to read expectancy gate state: {state_error}"
        ) from state_error
    if not isinstance(state_document, dict):
        raise ValueError("expectancy gate state root must be a JSON object")
    _validate_state_document(state_document, gate_config)
    return state_document


def save_expectancy_gate_state_atomically(
    state_path: Path,
    state_document: dict[str, Any],
    gate_config: strategy.ExpectancyGateConfig,
) -> None:
    """Validate and atomically persist the live sensor ledger."""

    _validate_state_document(state_document, gate_config)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as state_file:
        json.dump(state_document, state_file, indent=2, sort_keys=True)
    os.replace(temporary_path, state_path)


@contextlib.contextmanager
def _exclusive_state_lock(state_path: Path) -> Iterator[None]:
    """Serialize dashboard admissions with cron sensor advancement."""

    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _accepted_trade_key(accepted_trade: dict[str, Any]) -> str:
    """Return the stable identity of one live accepted entry."""

    return "|".join(
        str(accepted_trade.get(field_name) or "")
        for field_name in (
            "signal_date",
            "bucket",
            "strategy_id",
            "symbol",
        )
    )


def _bucket_order_by_label(
    config_document: dict[str, Any],
) -> dict[str, int]:
    """Return stable configured bucket order for heap tie-breaking."""

    raw_buckets = config_document.get("buckets")
    if not isinstance(raw_buckets, list):
        return {}
    return {
        str(raw_bucket.get("label")): bucket_order
        for bucket_order, raw_bucket in enumerate(raw_buckets)
        if isinstance(raw_bucket, dict) and raw_bucket.get("label")
    }


def record_accepted_trades_at_path(
    *,
    state_path: Path,
    config_document: dict[str, Any],
    accepted_trades: Sequence[dict[str, Any]],
) -> int:
    """Register dashboard-confirmed funded or phantom accepted entries.

    The write happens before broker submission.  It records allocation
    acceptance, not broker execution, matching the simulator object's role.
    Duplicate confirmations are ignored by the stable entry identity.
    """

    gate_config = parse_live_expectancy_gate_config(config_document)
    if gate_config is None or not accepted_trades:
        return 0
    if not state_path.exists():
        raise ValueError(
            "expectancy gate state is missing; run the daily cron before "
            "confirming entries"
        )

    bucket_order_by_label = _bucket_order_by_label(config_document)
    with _exclusive_state_lock(state_path):
        state_document = load_expectancy_gate_state(state_path, gate_config)
        last_evaluation_date = state_document.get("last_evaluation_date")
        accepted_trade_keys = set(state_document["accepted_trade_keys"])
        accepted_count_by_bucket = state_document["accepted_count_by_bucket"]
        added_trade_count = 0

        for raw_accepted_trade in accepted_trades:
            accepted_trade = dict(raw_accepted_trade)
            accepted_trade_key = _accepted_trade_key(accepted_trade)
            if not accepted_trade_key.strip("|"):
                raise ValueError("accepted expectancy trade identity is empty")
            required_fields = (
                "signal_date",
                "bucket",
                "strategy_id",
                "symbol",
                "tp_pct",
                "min_hold_tp",
            )
            missing_fields = [
                field_name
                for field_name in required_fields
                if accepted_trade.get(field_name) is None
            ]
            if missing_fields:
                raise ValueError(
                    "accepted expectancy trade missing field(s): "
                    + ", ".join(missing_fields)
                )
            if (
                last_evaluation_date is None
                or str(accepted_trade["signal_date"])
                != str(last_evaluation_date)
            ):
                raise ValueError(
                    "accepted expectancy trade signal_date does not match "
                    "the cron sensor evaluation date"
                )
            if accepted_trade_key in accepted_trade_keys:
                continue

            bucket_label = str(accepted_trade["bucket"])
            bucket_acceptance_order = int(
                accepted_count_by_bucket.get(bucket_label, 0)
            )
            acceptance_order = int(state_document["next_acceptance_order"])
            pending_trade = {
                "signal_date": str(accepted_trade["signal_date"]),
                "bucket": bucket_label,
                "strategy_id": str(accepted_trade["strategy_id"]),
                "symbol": str(accepted_trade["symbol"]),
                "tp_pct": float(accepted_trade["tp_pct"]),
                "sl_pct": float(accepted_trade.get("sl_pct") or 0.0),
                "min_hold_tp": int(accepted_trade["min_hold_tp"]),
                "min_hold_sl": int(accepted_trade.get("min_hold_sl") or 0),
                "disable_sl_trigger": bool(
                    accepted_trade.get("disable_sl_trigger", False)
                ),
                "max_hold": (
                    int(accepted_trade["max_hold"])
                    if accepted_trade.get("max_hold") is not None
                    else None
                ),
                "rolling_mp": (
                    float(accepted_trade["rolling_mp"])
                    if accepted_trade.get("rolling_mp") is not None
                    else 0.0
                ),
                "reset_hold_on_reentry_signal": bool(
                    accepted_trade.get(
                        "reset_hold_on_reentry_signal", False
                    )
                ),
                "acceptance_order": acceptance_order,
                "bucket_order": int(
                    bucket_order_by_label.get(bucket_label, 2**31 - 1)
                ),
                "bucket_acceptance_order": bucket_acceptance_order,
                "wr_gate_phantom": bool(
                    accepted_trade.get("wr_gate_phantom", False)
                ),
                "expectancy_gated": bool(
                    accepted_trade.get("expectancy_gated", False)
                ),
                "expectancy_priority_override": bool(
                    accepted_trade.get("expectancy_priority_override", False)
                ),
            }
            state_document["pending_trades"].append(pending_trade)
            state_document["next_acceptance_order"] = acceptance_order + 1
            accepted_count_by_bucket[bucket_label] = bucket_acceptance_order + 1
            state_document["accepted_trade_keys"].append(accepted_trade_key)
            accepted_trade_keys.add(accepted_trade_key)
            added_trade_count += 1

            is_expectancy_gated = pending_trade["expectancy_gated"]
            if is_expectancy_gated:
                state_document["expectancy_gated_trade_count"] = int(
                    state_document.get("expectancy_gated_trade_count", 0)
                ) + 1
                closed_episodes = state_document["closed_episodes"]
                if not state_document.get(
                    "previous_accepted_entry_was_gated", False
                ):
                    closed_episodes.append({
                        "first_entry_date": pending_trade["signal_date"],
                        "last_entry_date": pending_trade["signal_date"],
                        "gated_trade_count": 1,
                    })
                else:
                    current_episode = closed_episodes[-1]
                    current_episode["last_entry_date"] = pending_trade[
                        "signal_date"
                    ]
                    current_episode["gated_trade_count"] = int(
                        current_episode["gated_trade_count"]
                    ) + 1
            state_document[
                "previous_accepted_entry_was_gated"
            ] = is_expectancy_gated
            if pending_trade["expectancy_priority_override"]:
                state_document["priority_override_entry_count"] = int(
                    state_document.get("priority_override_entry_count", 0)
                ) + 1

        if added_trade_count:
            save_expectancy_gate_state_atomically(
                state_path, state_document, gate_config
            )
        return added_trade_count


def build_closed_signal_trade_index(
    adaptive_history_state: dict[str, Any],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Index counterfactual signal exits by full accepted-entry identity."""

    closed_trade_by_identity: dict[
        tuple[str, str, str, str], dict[str, Any]
    ] = {}
    closed_trades = adaptive_history_state.get(
        adaptive_tp_sl_virtual_trade_history.
        ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY,
        [],
    )
    for closed_trade in closed_trades:
        if not isinstance(closed_trade, dict):
            continue
        identity = (
            str(closed_trade.get("symbol") or ""),
            str(closed_trade.get("bucket") or ""),
            str(closed_trade.get("strategy_id") or ""),
            str(closed_trade.get("entry_date") or ""),
        )
        closed_trade_by_identity[identity] = closed_trade
    return closed_trade_by_identity


def build_reentry_signal_dates_by_symbol(
    adaptive_history_state: dict[str, Any],
) -> dict[str, set[pandas.Timestamp]]:
    """Index every raw bucket entry date for cross-bucket hold resets."""

    reentry_dates_by_symbol: dict[str, set[pandas.Timestamp]] = {}
    for history_key in (
        adaptive_tp_sl_virtual_trade_history.
        ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY,
        adaptive_tp_sl_virtual_trade_history.
        ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY,
    ):
        for raw_trade in adaptive_history_state.get(history_key, []):
            if not isinstance(raw_trade, dict):
                continue
            symbol = str(raw_trade.get("symbol") or "")
            entry_date = raw_trade.get("entry_date")
            if not symbol or not entry_date:
                continue
            try:
                entry_timestamp = pandas.Timestamp(entry_date)
            except (TypeError, ValueError):
                continue
            reentry_dates_by_symbol.setdefault(symbol, set()).add(
                entry_timestamp
            )
    return reentry_dates_by_symbol


def resolve_trade_outcome_before_date(
    *,
    pending_trade: dict[str, Any],
    evaluation_date: pandas.Timestamp,
    data_directory: Path,
    closed_trade_by_identity: dict[
        tuple[str, str, str, str], dict[str, Any]
    ],
    reentry_signal_dates_by_symbol: dict[
        str, set[pandas.Timestamp]
    ] | None = None,
) -> dict[str, Any] | None:
    """Resolve one final adaptive outcome observable before a date."""

    from stock_indicator import multi_bucket_today

    signal_date_string = str(pending_trade.get("signal_date") or "")
    symbol = str(pending_trade.get("symbol") or "")
    if not signal_date_string or not symbol:
        return None
    fill_date_string = str(
        pending_trade.get("fill_date")
        or multi_bucket_today._execution_date_string(signal_date_string)
    )
    entry_price_value = pending_trade.get("entry_price")
    if entry_price_value is None:
        resolved_entry_price = multi_bucket_today._read_open_price(
            data_directory, symbol, fill_date_string
        )
        if resolved_entry_price is None:
            return None
        pending_trade["fill_date"] = fill_date_string
        pending_trade["entry_price"] = round(float(resolved_entry_price), 4)
        entry_price_value = pending_trade["entry_price"]

    identity = (
        symbol,
        str(pending_trade.get("bucket") or ""),
        str(pending_trade.get("strategy_id") or ""),
        signal_date_string,
    )
    signal_close = closed_trade_by_identity.get(identity)
    signal_exit_date_string = (
        str(signal_close.get("exit_date"))
        if signal_close is not None and signal_close.get("exit_date")
        else None
    )
    horizon_date_string = signal_exit_date_string or evaluation_date.date().isoformat()
    adaptive_result = (
        multi_bucket_today.
        compute_adaptive_tp_sl_virtual_trade_close(
            data_directory,
            symbol,
            fill_date_string,
            float(entry_price_value),
            horizon_date_string,
            float(pending_trade["tp_pct"]),
            float(pending_trade.get("sl_pct") or 0.0),
            min_hold_tp=int(pending_trade["min_hold_tp"]),
            min_hold_sl=int(pending_trade.get("min_hold_sl") or 0),
            disable_sl_trigger=bool(
                pending_trade.get("disable_sl_trigger", True)
            ),
            breakeven_trigger_pct=float(
                pending_trade.get("rolling_mp") or 0.0
            ),
            max_hold=pending_trade.get("max_hold"),
            reset_hold_on_reentry_signal=bool(
                pending_trade.get(
                    "reset_hold_on_reentry_signal", False
                )
            ),
            re_fire_dates=(
                (reentry_signal_dates_by_symbol or {}).get(symbol)
            ),
        )
    )
    if adaptive_result is not None:
        _win, percentage_change, exit_reason, adaptive_exit_date = adaptive_result
        if (
            exit_reason
            in {"adaptive_take_profit", "adaptive_stop_loss", "max_hold"}
            and adaptive_exit_date < evaluation_date
        ):
            return {
                "exit_date": adaptive_exit_date.date().isoformat(),
                "percentage_change": float(percentage_change),
                "exit_reason": exit_reason,
            }

    if signal_close is None or signal_exit_date_string is None:
        return None
    signal_exit_date = pandas.Timestamp(signal_exit_date_string)
    signal_percentage_change = signal_close.get("raw_pct")
    if (
        signal_percentage_change is None
        or signal_exit_date >= evaluation_date
    ):
        return None
    return {
        "exit_date": signal_exit_date.date().isoformat(),
        "percentage_change": float(signal_percentage_change),
        "exit_reason": "signal",
    }


def advance_expectancy_gate_state(
    *,
    state_document: dict[str, Any],
    gate_config: strategy.ExpectancyGateConfig,
    adaptive_history_state: dict[str, Any],
    evaluation_date: pandas.Timestamp,
    data_directory: Path,
) -> int:
    """Consume final accepted outcomes that exited strictly before today."""

    _validate_state_document(state_document, gate_config)
    previous_evaluation_date = state_document.get("last_evaluation_date")
    if (
        previous_evaluation_date is not None
        and evaluation_date < pandas.Timestamp(previous_evaluation_date)
    ):
        raise ValueError(
            "expectancy gate evaluation date cannot move backwards"
        )
    closed_trade_by_identity = build_closed_signal_trade_index(
        adaptive_history_state
    )
    reentry_signal_dates_by_symbol = build_reentry_signal_dates_by_symbol(
        adaptive_history_state
    )
    resolved_outcomes: list[dict[str, Any]] = []
    still_pending: list[dict[str, Any]] = []
    for pending_trade in state_document["pending_trades"]:
        if not isinstance(pending_trade, dict):
            raise ValueError("expectancy gate pending trade must be an object")
        resolved_outcome = resolve_trade_outcome_before_date(
            pending_trade=pending_trade,
            evaluation_date=evaluation_date,
            data_directory=data_directory,
            closed_trade_by_identity=closed_trade_by_identity,
            reentry_signal_dates_by_symbol=(
                reentry_signal_dates_by_symbol
            ),
        )
        if resolved_outcome is None:
            still_pending.append(pending_trade)
            continue
        resolved_outcomes.append({**pending_trade, **resolved_outcome})

    resolved_outcomes.sort(
        key=lambda outcome: (
            pandas.Timestamp(outcome["exit_date"]),
            pandas.Timestamp(outcome["signal_date"]),
            int(outcome.get("bucket_order", 2**31 - 1)),
            int(outcome.get("bucket_acceptance_order", 2**31 - 1)),
            int(outcome.get("acceptance_order", 2**31 - 1)),
        )
    )
    rolling_outcomes = list(state_document["rolling_outcomes"])
    for resolved_outcome in resolved_outcomes:
        rolling_outcomes.append({
            "signal_date": resolved_outcome["signal_date"],
            "exit_date": resolved_outcome["exit_date"],
            "bucket": resolved_outcome["bucket"],
            "strategy_id": resolved_outcome["strategy_id"],
            "symbol": resolved_outcome["symbol"],
            "percentage_change": resolved_outcome["percentage_change"],
            "exit_reason": resolved_outcome["exit_reason"],
            "acceptance_order": resolved_outcome["acceptance_order"],
        })
    state_document["rolling_outcomes"] = rolling_outcomes[-gate_config.window :]
    state_document["pending_trades"] = still_pending
    state_document["last_evaluation_date"] = (
        evaluation_date.date().isoformat()
    )
    return len(resolved_outcomes)


def build_expectancy_gate_sensor_state(
    state_document: dict[str, Any],
    gate_config: strategy.ExpectancyGateConfig,
) -> dict[str, Any]:
    """Return today's once-per-day stop and soft-tier decisions."""

    _validate_state_document(state_document, gate_config)
    rolling_outcomes = state_document["rolling_outcomes"]
    rolling_return_values = [
        float(outcome["percentage_change"])
        for outcome in rolling_outcomes
        if isinstance(outcome, dict)
    ]
    rolling_mean = (
        sum(rolling_return_values) / len(rolling_return_values)
        if rolling_return_values
        else None
    )
    window_full = len(rolling_return_values) >= gate_config.window
    gate_closed = (
        gate_config.cold_start == "closed"
        if not window_full
        else bool(rolling_mean is not None and rolling_mean < gate_config.threshold)
    )
    soft_threshold = gate_config.priority_override_threshold
    priority_override_active = bool(
        window_full
        and soft_threshold is not None
        and rolling_mean is not None
        and rolling_mean < soft_threshold
    )
    return {
        "status": "ready",
        "count": len(rolling_return_values),
        "window": gate_config.window,
        "window_full": window_full,
        "rolling_mean": rolling_mean,
        "stop_threshold": gate_config.threshold,
        "soft_threshold": soft_threshold,
        "gate_closed": gate_closed,
        "priority_override_active": priority_override_active,
        "open_pending": len(state_document["pending_trades"]),
        "closed_episodes": len(state_document["closed_episodes"]),
        "expectancy_gated_trades": int(
            state_document.get("expectancy_gated_trade_count", 0)
        ),
        "priority_override_entries": int(
            state_document.get("priority_override_entry_count", 0)
        ),
    }


def format_expectancy_gate_sensor_log(
    sensor_state: dict[str, Any],
    *,
    fed_this_run: int,
) -> str:
    """Format the machine-readable daily sensor heartbeat."""

    rolling_mean = sensor_state.get("rolling_mean")
    soft_threshold = sensor_state.get("soft_threshold")
    rolling_mean_text = (
        f"{float(rolling_mean):.6f}" if rolling_mean is not None else "None"
    )
    soft_threshold_text = (
        f"{float(soft_threshold):.6f}"
        if soft_threshold is not None
        else "None"
    )
    return (
        "[EXPECTANCY_GATE_SENSOR] status=ready "
        f"mean={rolling_mean_text} "
        f"stop_threshold={float(sensor_state['stop_threshold']):.6f} "
        f"soft_threshold={soft_threshold_text} "
        f"gate_closed={sensor_state['gate_closed']} "
        "priority_override_active="
        f"{sensor_state['priority_override_active']} "
        f"window={sensor_state['count']}/{sensor_state['window']} "
        f"window_full={sensor_state['window_full']} "
        f"open_pending={sensor_state['open_pending']} "
        f"fed_this_run={fed_this_run} "
        f"closed_episodes={sensor_state['closed_episodes']} "
        "expectancy_gated_trades="
        f"{sensor_state['expectancy_gated_trades']} "
        "priority_override_entries="
        f"{sensor_state['priority_override_entries']}"
    )


def advance_expectancy_gate_state_at_path(
    *,
    state_path: Path,
    config_document: dict[str, Any],
    adaptive_history_state: dict[str, Any],
    evaluation_date: pandas.Timestamp,
    data_directory: Path,
) -> tuple[dict[str, Any], str] | None:
    """Advance the configured live state atomically and return its heartbeat."""

    gate_config = parse_live_expectancy_gate_config(config_document)
    if gate_config is None:
        return None
    with _exclusive_state_lock(state_path):
        state_document = load_expectancy_gate_state(state_path, gate_config)
        fed_this_run = advance_expectancy_gate_state(
            state_document=state_document,
            gate_config=gate_config,
            adaptive_history_state=adaptive_history_state,
            evaluation_date=evaluation_date,
            data_directory=data_directory,
        )
        save_expectancy_gate_state_atomically(
            state_path, state_document, gate_config
        )
    sensor_state = build_expectancy_gate_sensor_state(
        state_document, gate_config
    )
    return (
        sensor_state,
        format_expectancy_gate_sensor_log(
            sensor_state, fed_this_run=fed_this_run
        ),
    )


def live_expectancy_gate_state_path(
    live_state_directory: Path,
    *,
    shadow_mode: bool,
) -> Path:
    """Return the isolated live or shadow production sensor path."""

    suffix = "_shadow" if shadow_mode else ""
    return live_state_directory / f"expectancy_gate_state{suffix}.json"


def is_finite_sensor_state(sensor_state: dict[str, Any]) -> bool:
    """Return whether all present numeric decision values are finite."""

    for field_name in (
        "rolling_mean",
        "mean",
        "stop_threshold",
        "soft_threshold",
    ):
        field_value = sensor_state.get(field_name)
        if field_value is not None and not math.isfinite(float(field_value)):
            return False
    return True
