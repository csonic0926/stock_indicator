"""Production today-slice signal generator for multi-bucket configs.

Reads `data/multi_bucket_production.json` (or any compatible config) and
reproduces the simulator's single-day decision: per-bucket signals,
cross-bucket slot competition, frozen TP/SL via the shared helper.
Persists rolling state (winners/losers/pending_rolling) in
`adaptive_state.json` for the next day.

Design contract: the simulator (run_complex_simulation in strategy.py)
remains the source of truth. Anything in this module that touches
strategy logic must call into shared callables in strategy.py rather
than reimplementing them. Phase 2.2 parity gate compares this module's
output against the simulator's bar-by-bar replay; any divergence is a
blocker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas

from . import daily_job, strategy
from .strategy_sets import (
    load_strategy_entry_filters,
    load_strategy_set_mapping,
)


SCHEMA_VERSION = 2


@dataclass
class MultiBucketRunConfig:
    """Parsed multi-bucket simulator/daily-signal configuration.

    Mirrors the JSON shape consumed by `do_multi_bucket_simulation` and
    used by the new `do_multi_bucket_daily_signal`. I/O-heavy resolution
    (data source path lookup, symbol list expansion) is left to the
    caller so this loader stays pure and testable.
    """

    bucket_definitions: Dict[str, strategy.ComplexStrategySetDefinition]
    adaptive_tp_sl: strategy.AdaptiveTPSLConfig | None
    maximum_position_count: int
    starting_cash: float
    withdraw_amount: float
    margin_multiplier: float
    minimum_holding_bars: int
    show_trade_details: bool
    start_date_string: str | None
    confirmation_mode: str | None
    use_confirmation_angle: bool
    confirmation_entry_mode: str
    confirmation_sma_angle_range: Tuple[float, float] | None
    data_source_name: str | None
    symbol_list_name: str | None
    ff12_data_path_text: str | None
    max_same_symbol: int
    raw_document: Dict[str, Any]


def _parse_volume_filter_text(text: str) -> Tuple[float | None, float | None, int | None, int]:
    """Parse a dollar-volume filter expression into (min_abs, min_ratio,
    top_n, max_per_group). Mirrors manage._parse_volume_filter; the two
    implementations must stay in sync until a future refactor merges
    them. Defaults max_per_group=1 when no ,PickN suffix is given."""
    import re as _re

    max_per_group = 1
    pick_match = _re.fullmatch(
        r"(.*),Pick(\d+)", text, flags=_re.IGNORECASE
    )
    if pick_match is not None:
        text = pick_match.group(1)
        max_per_group = int(pick_match.group(2))

    pattern_pct_top = _re.compile(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,Top(\d+)", flags=_re.IGNORECASE
    )
    match = pattern_pct_top.fullmatch(text)
    if match:
        ratio = float(match.group(1)) / 100
        top_n = int(match.group(2))
        return None, ratio, top_n, max_per_group

    pattern_pct_nth = _re.compile(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,(\d+)th"
    )
    match = pattern_pct_nth.fullmatch(text)
    if match:
        ratio = float(match.group(1)) / 100
        rank_n = int(match.group(2))
        return None, ratio, rank_n, max_per_group

    pattern_abs_top = _re.compile(
        r"dollar_volume>(\d+(?:\.\d+)?),Top(\d+)", flags=_re.IGNORECASE
    )
    match = pattern_abs_top.fullmatch(text)
    if match:
        absolute = float(match.group(1))
        top_n = int(match.group(2))
        return absolute, None, top_n, max_per_group

    pattern_abs_nth = _re.compile(
        r"dollar_volume>(\d+(?:\.\d+)?),(\d+)th"
    )
    match = pattern_abs_nth.fullmatch(text)
    if match:
        absolute = float(match.group(1))
        rank_n = int(match.group(2))
        return absolute, None, rank_n, max_per_group

    pattern_pct_only = _re.compile(r"dollar_volume>(\d+(?:\.\d{1,2})?)%")
    match = pattern_pct_only.fullmatch(text)
    if match:
        ratio = float(match.group(1)) / 100
        return None, ratio, None, max_per_group

    pattern_abs_only = _re.compile(r"dollar_volume>(\d+(?:\.\d+)?)")
    match = pattern_abs_only.fullmatch(text)
    if match:
        absolute = float(match.group(1))
        return absolute, None, None, max_per_group

    pattern_top = _re.compile(r"dollar_volume=Top(\d+)", flags=_re.IGNORECASE)
    pattern_nth = _re.compile(r"dollar_volume=(\d+)th")
    match_top = pattern_top.fullmatch(text)
    match_nth = pattern_nth.fullmatch(text)
    if match_top or match_nth:
        rank_n = int((match_top or match_nth).group(1))
        return None, None, rank_n, max_per_group

    raise ValueError(
        "unsupported filter; expected dollar_volume>NUMBER, "
        "dollar_volume>NUMBER%, dollar_volume=TopN (or Nth), "
        "dollar_volume>NUMBER,TopN (or ,Nth), or "
        "dollar_volume>NUMBER%,TopN (or ,Nth)"
    )


def parse_skip_ff12_groups(
    raw_group_values: Any,
    *,
    bucket_label: str,
) -> set[int]:
    """Parse the optional per-bucket ``skip_ff12_groups`` config field."""

    if raw_group_values is None:
        return set()
    if isinstance(raw_group_values, str):
        group_value_parts: Sequence[Any] = [
            value_part.strip()
            for value_part in raw_group_values.split(",")
            if value_part.strip()
        ]
    elif isinstance(raw_group_values, Sequence):
        group_value_parts = raw_group_values
    else:
        raise ValueError(
            f"bucket {bucket_label}: skip_ff12_groups must be a list or comma-separated string"
        )

    skipped_group_identifiers: set[int] = set()
    for raw_group_value in group_value_parts:
        if isinstance(raw_group_value, bool):
            raise ValueError(
                f"bucket {bucket_label}: skip_ff12_groups values must be positive integers"
            )
        try:
            group_identifier = int(raw_group_value)
        except (TypeError, ValueError) as parse_error:
            raise ValueError(
                f"bucket {bucket_label}: skip_ff12_groups values must be positive integers"
            ) from parse_error
        if group_identifier < 1:
            raise ValueError(
                f"bucket {bucket_label}: skip_ff12_groups values must be positive integers"
            )
        skipped_group_identifiers.add(group_identifier)
    return skipped_group_identifiers


def load_multi_bucket_config(config_path: Path) -> MultiBucketRunConfig:
    """Parse a multi-bucket config JSON. Raises ValueError on any field
    that fails validation; caller is responsible for surfacing messages
    and returning early."""
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        document = json.load(config_file)
    if not isinstance(document, dict):
        raise ValueError("config root must be a JSON object")
    raw_buckets = document.get("buckets")
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError("config must contain a non-empty 'buckets' array")

    try:
        maximum_position_count = int(document.get("max_position_count", 0))
    except (TypeError, ValueError) as parse_error:
        raise ValueError("max_position_count must be an integer") from parse_error
    if maximum_position_count <= 0:
        raise ValueError("max_position_count must be positive")

    starting_cash = float(document.get("starting_cash", 3000.0))
    withdraw_amount = float(document.get("withdraw", 0.0))
    margin_multiplier = float(document.get("margin", 1.0))
    if margin_multiplier < 1.0:
        raise ValueError("margin must be >= 1.0")
    minimum_holding_bars = int(document.get("min_hold", 0))
    if minimum_holding_bars < 0:
        raise ValueError("min_hold must be >= 0")
    show_trade_details = bool(document.get("show_trade_details", False))

    start_date_string = document.get("start_date")
    if start_date_string is not None:
        import datetime as _datetime
        try:
            _datetime.date.fromisoformat(start_date_string)
        except ValueError as parse_error:
            raise ValueError(
                "invalid start_date; expected YYYY-MM-DD"
            ) from parse_error

    confirmation_mode = document.get("confirmation_mode")
    use_confirmation_angle = False
    confirmation_entry_mode = "limit"
    if confirmation_mode in (None, "", False):
        pass
    elif confirmation_mode == "market":
        use_confirmation_angle = True
        confirmation_entry_mode = "market"
    elif confirmation_mode == "limit":
        use_confirmation_angle = True
        confirmation_entry_mode = "limit"
    else:
        raise ValueError(
            f"invalid confirmation_mode: {confirmation_mode} "
            "(expected 'market', 'limit', or null)"
        )

    confirmation_sma_angle_range: Tuple[float, float] | None = None
    raw_confirmation_min = document.get("confirmation_sma_angle_min")
    raw_confirmation_max = document.get("confirmation_sma_angle_max")
    if raw_confirmation_min is not None or raw_confirmation_max is not None:
        default_min, default_max = strategy.CONFIRMATION_SMA_ANGLE_RANGE
        try:
            resolved_min = (
                float(raw_confirmation_min)
                if raw_confirmation_min is not None
                else default_min
            )
            resolved_max = (
                float(raw_confirmation_max)
                if raw_confirmation_max is not None
                else default_max
            )
        except (TypeError, ValueError) as parse_error:
            raise ValueError(
                "confirmation_sma_angle_min/max must be numbers"
            ) from parse_error
        if resolved_min > resolved_max:
            raise ValueError("confirmation_sma_angle_min must be <= max")
        confirmation_sma_angle_range = (resolved_min, resolved_max)

    strategy_mapping = load_strategy_set_mapping()
    entry_filters_mapping = load_strategy_entry_filters()

    bucket_definitions: Dict[str, strategy.ComplexStrategySetDefinition] = {}
    seen_labels: set[str] = set()
    for bucket_index, raw_bucket in enumerate(raw_buckets):
        if not isinstance(raw_bucket, dict):
            raise ValueError(f"bucket[{bucket_index}] must be a JSON object")
        label = str(raw_bucket.get("label") or f"bucket{bucket_index + 1}")
        if label in seen_labels:
            raise ValueError(f"duplicate bucket label: {label}")
        seen_labels.add(label)
        strategy_identifier = raw_bucket.get("strategy_id")
        if not strategy_identifier:
            raise ValueError(f"bucket {label} requires 'strategy_id'")
        if strategy_identifier not in strategy_mapping:
            raise ValueError(
                f"bucket {label}: unknown strategy_id '{strategy_identifier}'"
            )
        buy_strategy_name, sell_strategy_name = strategy_mapping[
            strategy_identifier
        ]
        volume_filter_text = raw_bucket.get("dollar_volume_filter")
        if not volume_filter_text:
            raise ValueError(
                f"bucket {label} requires 'dollar_volume_filter'"
            )
        try:
            (
                minimum_average_dollar_volume,
                minimum_average_dollar_volume_ratio,
                top_dollar_volume_rank,
                maximum_symbols_per_group,
            ) = _parse_volume_filter_text(volume_filter_text)
        except ValueError as parse_error:
            raise ValueError(
                f"bucket {label} volume filter: {parse_error}"
            ) from parse_error

        try:
            stop_loss_percentage = float(raw_bucket.get("stop_loss", 1.0))
            take_profit_percentage = float(raw_bucket.get("take_profit", 0.0))
        except (TypeError, ValueError) as parse_error:
            raise ValueError(
                f"bucket {label}: stop_loss/take_profit must be numbers"
            ) from parse_error
        try:
            entry_priority = int(raw_bucket.get("priority", 0))
        except (TypeError, ValueError) as parse_error:
            raise ValueError(
                f"bucket {label}: priority must be an integer"
            ) from parse_error
        raw_max_positions = raw_bucket.get("max_positions")
        if raw_max_positions is None:
            bucket_maximum_positions: int | None = None
        else:
            try:
                bucket_maximum_positions = int(raw_max_positions)
            except (TypeError, ValueError) as parse_error:
                raise ValueError(
                    f"bucket {label}: max_positions must be an integer or null"
                ) from parse_error
            if bucket_maximum_positions <= 0:
                raise ValueError(
                    f"bucket {label}: max_positions must be positive"
                )
        skipped_fama_french_groups = parse_skip_ff12_groups(
            raw_bucket.get("skip_ff12_groups"),
            bucket_label=label,
        )

        d_sma_range = None
        ema_range = None
        d_ema_range = None
        price_score_min_value = None
        price_score_max_value = None
        shape_slope_min_value = None
        shape_dev_50_max_value = None
        shape_bsv_lookback_value = None
        if strategy_identifier in entry_filters_mapping:
            entry_filters = entry_filters_mapping[strategy_identifier]
            if (
                entry_filters.d_sma_min is not None
                or entry_filters.d_sma_max is not None
            ):
                d_sma_range = (
                    entry_filters.d_sma_min
                    if entry_filters.d_sma_min is not None
                    else -99.0,
                    entry_filters.d_sma_max
                    if entry_filters.d_sma_max is not None
                    else 99.0,
                )
            if (
                entry_filters.ema_min is not None
                or entry_filters.ema_max is not None
            ):
                ema_range = (
                    entry_filters.ema_min
                    if entry_filters.ema_min is not None
                    else -99.0,
                    entry_filters.ema_max
                    if entry_filters.ema_max is not None
                    else 99.0,
                )
            if (
                entry_filters.d_ema_min is not None
                or entry_filters.d_ema_max is not None
            ):
                d_ema_range = (
                    entry_filters.d_ema_min
                    if entry_filters.d_ema_min is not None
                    else -99.0,
                    entry_filters.d_ema_max
                    if entry_filters.d_ema_max is not None
                    else 99.0,
                )
            price_score_min_value = entry_filters.price_score_min
            price_score_max_value = entry_filters.price_score_max
            shape_slope_min_value = entry_filters.shape_slope_min
            shape_dev_50_max_value = entry_filters.shape_dev_50_max
            shape_bsv_lookback_value = entry_filters.shape_bsv_lookback

        raw_exit_alpha_factor = raw_bucket.get("exit_alpha_factor")
        exit_alpha_factor_value: float | None = None
        if raw_exit_alpha_factor is not None:
            try:
                exit_alpha_factor_value = float(raw_exit_alpha_factor)
            except (TypeError, ValueError) as parse_error:
                raise ValueError(
                    f"bucket {label}: exit_alpha_factor must be a number"
                ) from parse_error

        near_delta_range_value: Tuple[float, float] | None = None
        raw_near_delta = raw_bucket.get("near_delta_range")
        if raw_near_delta is not None:
            try:
                near_delta_range_value = (
                    float(raw_near_delta[0]),
                    float(raw_near_delta[1]),
                )
            except (TypeError, ValueError, IndexError) as parse_error:
                raise ValueError(
                    f"bucket {label}: near_delta_range must be [min, max]"
                ) from parse_error

        price_tightness_range_value: Tuple[float, float] | None = None
        raw_price_tightness = raw_bucket.get("price_tightness_range")
        if raw_price_tightness is not None:
            try:
                price_tightness_range_value = (
                    float(raw_price_tightness[0]),
                    float(raw_price_tightness[1]),
                )
            except (TypeError, ValueError, IndexError) as parse_error:
                raise ValueError(
                    f"bucket {label}: price_tightness_range must be [min, max]"
                ) from parse_error

        sma_150_angle_min_value: float | None = None
        raw_sma_150 = raw_bucket.get("sma_150_angle_min")
        if raw_sma_150 is not None:
            try:
                sma_150_angle_min_value = float(raw_sma_150)
            except (TypeError, ValueError) as parse_error:
                raise ValueError(
                    f"bucket {label}: sma_150_angle_min must be a number"
                ) from parse_error

        bucket_definitions[label] = strategy.ComplexStrategySetDefinition(
            label=label,
            buy_strategy_name=buy_strategy_name,
            sell_strategy_name=sell_strategy_name,
            strategy_identifier=strategy_identifier,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
            top_dollar_volume_rank=top_dollar_volume_rank,
            maximum_symbols_per_group=maximum_symbols_per_group,
            d_sma_range=d_sma_range,
            ema_range=ema_range,
            d_ema_range=d_ema_range,
            near_delta_range=near_delta_range_value,
            price_tightness_range=price_tightness_range_value,
            sma_150_angle_min=sma_150_angle_min_value,
            use_ftd_confirmation=bool(raw_bucket.get("use_ftd", False)),
            trailing_stop_percentage=float(raw_bucket.get("trailing_stop", 0)),
            price_score_min=price_score_min_value,
            price_score_max=price_score_max_value,
            entry_priority=entry_priority,
            maximum_positions=bucket_maximum_positions,
            fill_remaining=bool(raw_bucket.get("fill_remaining", False)),
            skipped_fama_french_groups=skipped_fama_french_groups,
            exit_alpha_factor=exit_alpha_factor_value,
            shape_slope_min=shape_slope_min_value,
            shape_dev_50_max=shape_dev_50_max_value,
            shape_bsv_lookback=shape_bsv_lookback_value,
            tp_regime_adjust=(
                bool(raw_bucket["tp_regime_adjust"])
                if "tp_regime_adjust" in raw_bucket
                and raw_bucket["tp_regime_adjust"] is not None
                else None
            ),
            fixed_tp=(
                float(raw_bucket["fixed_tp"])
                if "fixed_tp" in raw_bucket
                and raw_bucket["fixed_tp"] is not None
                else None
            ),
            fixed_sl=(
                float(raw_bucket["fixed_sl"])
                if "fixed_sl" in raw_bucket
                and raw_bucket["fixed_sl"] is not None
                else None
            ),
            min_sl=(
                float(raw_bucket["min_sl"])
                if "min_sl" in raw_bucket
                and raw_bucket["min_sl"] is not None
                else None
            ),
            # TODO: review
            # Production cron must honor the same per-bucket sigma override
            # as the simulator; otherwise live frozen TP can silently inherit
            # the top-level adaptive sigma for every bucket.
            sigma=(
                float(raw_bucket["sigma"])
                if "sigma" in raw_bucket
                and raw_bucket["sigma"] is not None
                else None
            ),
            slope_max=(
                float(raw_bucket["slope_max"])
                if "slope_max" in raw_bucket
                and raw_bucket["slope_max"] is not None
                else None
            ),
            slope_min=(
                float(raw_bucket["slope_min"])
                if "slope_min" in raw_bucket
                and raw_bucket["slope_min"] is not None
                else None
            ),
            free_fall_slope=(
                float(raw_bucket["free_fall_slope"])
                if "free_fall_slope" in raw_bucket
                and raw_bucket["free_fall_slope"] is not None
                else None
            ),
            free_fall_near_delta=(
                float(raw_bucket["free_fall_near_delta"])
                if "free_fall_near_delta" in raw_bucket
                and raw_bucket["free_fall_near_delta"] is not None
                else None
            ),
            slope_dead_zone_min=(
                float(raw_bucket["slope_dead_zone_min"])
                if "slope_dead_zone_min" in raw_bucket
                and raw_bucket["slope_dead_zone_min"] is not None
                else None
            ),
            slope_dead_zone_max=(
                float(raw_bucket["slope_dead_zone_max"])
                if "slope_dead_zone_max" in raw_bucket
                and raw_bucket["slope_dead_zone_max"] is not None
                else None
            ),
            v_filter_threshold=(
                float(raw_bucket["v_filter_threshold"])
                if "v_filter_threshold" in raw_bucket
                and raw_bucket["v_filter_threshold"] is not None
                else None
            ),
            pre_cross_signal_lookback=bool(
                raw_bucket.get("pre_cross_signal_lookback", False)
            ),
            additional_above_ranges=(
                [
                    (float(low), float(high))
                    for low, high in raw_bucket["additional_above_ranges"]
                ]
                if "additional_above_ranges" in raw_bucket
                and raw_bucket["additional_above_ranges"]
                else None
            ),
            max_hold=(
                int(raw_bucket["max_hold"])
                if "max_hold" in raw_bucket
                and raw_bucket["max_hold"] is not None
                else None
            ),
            reset_hold_on_reentry_signal=bool(
                raw_bucket.get("reset_hold_on_reentry_signal", False)
            ),
            gate_enabled=bool(raw_bucket.get("gate_enabled", True)),
            tp_slope_amplify=bool(raw_bucket.get("tp_slope_amplify", False)),
            override_min_hold_tp_only=(
                bool(raw_bucket["override_min_hold_tp_only"])
                if "override_min_hold_tp_only" in raw_bucket
                and raw_bucket["override_min_hold_tp_only"] is not None
                else None
            ),
            min_hold_tp=(
                int(raw_bucket["min_hold_tp"])
                if "min_hold_tp" in raw_bucket
                and raw_bucket["min_hold_tp"] is not None
                else None
            ),
            override_min_hold_sl_only=(
                bool(raw_bucket["override_min_hold_sl_only"])
                if "override_min_hold_sl_only" in raw_bucket
                and raw_bucket["override_min_hold_sl_only"] is not None
                else None
            ),
            min_hold_sl=(
                int(raw_bucket["min_hold_sl"])
                if "min_hold_sl" in raw_bucket
                and raw_bucket["min_hold_sl"] is not None
                else None
            ),
        )

    adaptive_tp_sl_config: strategy.AdaptiveTPSLConfig | None = None
    raw_adaptive = document.get("adaptive_tp_sl")
    if raw_adaptive:
        if isinstance(raw_adaptive, dict):
            raw_fixed_sl = raw_adaptive.get("fixed_sl")
            adaptive_tp_sl_config = strategy.AdaptiveTPSLConfig(
                window=int(raw_adaptive.get("window", 20)),
                sigma_multiplier=float(raw_adaptive.get("sigma", 0.5)),
                target_r=float(raw_adaptive.get("target_r", 2.0)),
                sl_sigma_multiplier=(
                    float(raw_adaptive["sl_sigma_multiplier"])
                    if "sl_sigma_multiplier" in raw_adaptive
                    else (
                        float(raw_adaptive["sl_sigma"])
                        if "sl_sigma" in raw_adaptive
                        else None
                    )
                ),
                min_tp=float(raw_adaptive.get("min_tp", 0.02)),
                min_sl=float(raw_adaptive.get("min_sl", 0.01)),
                min_samples=int(raw_adaptive.get("min_samples", 5)),
                fixed_sl=float(raw_fixed_sl) if raw_fixed_sl is not None else None,
                override_min_hold=bool(
                    raw_adaptive.get("override_min_hold", False),
                ),
                override_min_hold_tp_only=bool(
                    raw_adaptive.get("override_min_hold_tp_only", False),
                ),
                min_hold_tp=int(raw_adaptive.get("min_hold_tp", 0)),
                override_min_hold_sl_only=bool(
                    raw_adaptive.get("override_min_hold_sl_only", False),
                ),
                min_hold_sl=int(raw_adaptive.get("min_hold_sl", 0)),
                fixed_tp=(
                    float(raw_adaptive["fixed_tp"])
                    if raw_adaptive.get("fixed_tp") is not None
                    else None
                ),
                disable_sl_trigger=bool(
                    raw_adaptive.get("disable_sl_trigger", False),
                ),
                tp_regime_adjust=bool(
                    raw_adaptive.get("tp_regime_adjust", False),
                ),
                tp_regime_ratio_min=float(
                    raw_adaptive.get("tp_regime_ratio_min", 0.5),
                ),
                tp_regime_ratio_max=float(
                    raw_adaptive.get("tp_regime_ratio_max", 1.5),
                ),
                delayed_rolling_update=bool(
                    raw_adaptive.get("delayed_rolling_update", False),
                ),
                breakeven_at_mp=bool(raw_adaptive.get("breakeven_at_mp", False)),
                evict_oldest=bool(raw_adaptive.get("evict_oldest", False)),
            )
        else:
            adaptive_tp_sl_config = strategy.AdaptiveTPSLConfig()

    raw_ff12_data_path_text = document.get("ff12_data_path") or document.get(
        "sector_data_path"
    )

    return MultiBucketRunConfig(
        bucket_definitions=bucket_definitions,
        adaptive_tp_sl=adaptive_tp_sl_config,
        maximum_position_count=maximum_position_count,
        starting_cash=starting_cash,
        withdraw_amount=withdraw_amount,
        margin_multiplier=margin_multiplier,
        minimum_holding_bars=minimum_holding_bars,
        show_trade_details=show_trade_details,
        start_date_string=start_date_string,
        confirmation_mode=confirmation_mode,
        use_confirmation_angle=use_confirmation_angle,
        confirmation_entry_mode=confirmation_entry_mode,
        confirmation_sma_angle_range=confirmation_sma_angle_range,
        data_source_name=document.get("data_source"),
        symbol_list_name=document.get("symbol_list"),
        ff12_data_path_text=(
            str(raw_ff12_data_path_text)
            if raw_ff12_data_path_text is not None
            else None
        ),
        max_same_symbol=int(document.get("max_same_symbol", 1)),
        raw_document=document,
    )


# ----------------------------------------------------------------------
# Rolling state load / save
# ----------------------------------------------------------------------


def _empty_state() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "winners": [],
        "losers": [],
        "pending_rolling": [],
        "closed_trades": [],
    }


def load_state(state_path: Path) -> Dict[str, Any]:
    """Load `adaptive_state.json` (new schema). Returns an empty state
    when the file is missing, malformed, or stamped with an older
    schema_version. Caller should treat schema-mismatch as a cold-start
    signal and re-bootstrap via the simulator's --export-state-on-date
    helper rather than silently overwrite."""
    if not state_path.exists():
        return _empty_state()
    try:
        with state_path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except (json.JSONDecodeError, OSError):
        return _empty_state()
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        return _empty_state()
    state.setdefault("winners", [])
    state.setdefault("losers", [])
    state.setdefault("pending_rolling", [])
    state.setdefault("closed_trades", [])
    return state


def save_state_atomically(state_path: Path, state: Dict[str, Any]) -> None:
    """Write the state via tmp + os.replace so a crash mid-write cannot
    corrupt the file. Crucial: pending_rolling holds (closed_date, pct)
    pairs that flush only on a future entry — losing them means the
    rolling window forgets a closed trade."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as temp_file:
        json.dump(state, temp_file, indent=2)
    import os as _os
    _os.replace(temp_path, state_path)


# ----------------------------------------------------------------------
# Rolling-window flush + per-bucket entry filters
# ----------------------------------------------------------------------


def flush_pending_rolling_into_deques(
    state: Dict[str, Any],
    eval_date: pandas.Timestamp,
    window: int,
) -> None:
    """Mirror simulator strategy.py:1647-1668.

    `pending_rolling` holds (closed_date, pct) entries deposited by trades
    that closed on/after the entry day they should NOT contribute to (the
    `delayed_rolling_update` invariant). When today's eval_date is strictly
    later than a pending entry's closed_date, the entry is flushed into
    `winners` (pct > 0) or `losers` (pct < 0), each capped at `window`.
    """
    if not state.get("pending_rolling"):
        return
    winners = list(state.get("winners", []))
    losers = list(state.get("losers", []))
    remaining: List[Dict[str, Any]] = []
    for pending_entry in state["pending_rolling"]:
        closed_date_string = pending_entry.get("closed_date")
        try:
            closed_date_ts = pandas.Timestamp(closed_date_string)
        except (ValueError, TypeError):
            continue
        try:
            pct_value = float(pending_entry.get("pct", 0.0))
        except (TypeError, ValueError):
            continue
        if closed_date_ts < eval_date:
            if pct_value > 0:
                winners.append(pct_value)
                if len(winners) > window:
                    winners = winners[-window:]
            elif pct_value < 0:
                losers.append(pct_value)
                if len(losers) > window:
                    losers = losers[-window:]
        else:
            remaining.append(pending_entry)
    state["winners"] = winners
    state["losers"] = losers
    state["pending_rolling"] = remaining


def append_rolling_update(
    state: Dict[str, Any],
    closed_date: pandas.Timestamp,
    pct: float,
    *,
    delayed: bool,
    window: int,
) -> None:
    """Append a freshly-closed trade's pct into rolling state. When
    `delayed` is true, parks it in `pending_rolling` (the simulator's
    delayed_rolling_update path); otherwise flushes directly into
    winners/losers. Mirrors strategy.py:1610-1636.
    """
    if delayed:
        state.setdefault("pending_rolling", []).append({
            "closed_date": closed_date.strftime("%Y-%m-%d"),
            "pct": float(pct),
        })
        return
    if pct > 0:
        winners = list(state.get("winners", []))
        winners.append(float(pct))
        if len(winners) > window:
            winners = winners[-window:]
        state["winners"] = winners
    elif pct < 0:
        losers = list(state.get("losers", []))
        losers.append(float(pct))
        if len(losers) > window:
            losers = losers[-window:]
        state["losers"] = losers


def passes_per_bucket_entry_filters(
    bucket_def: strategy.ComplexStrategySetDefinition,
    slope_60: float | None,
    near_delta: float | None,
    above_pv: float | None = None,
    above_pv_previous: float | None = None,
) -> bool:
    """Mirror simulator strategy.py:1684-1780 entry filters.

    - slope_max / slope_min: unconditional bounds on slope_60 at entry.
    - free_fall_slope + free_fall_near_delta: compound AND filter (skip
      when both deeply negative — toxic free-fall cell).
    - slope_dead_zone_min / slope_dead_zone_max: skip INSIDE band
      (mid-rally noise, not regime transition).
    - v_filter_threshold: keep ONLY when above_pv crosses DOWN through
      the threshold within one bar (T-1 > threshold AND T < threshold).
    Returns True when the candidate survives all filters."""
    if slope_60 is not None:
        if (
            bucket_def.slope_max is not None
            and slope_60 > bucket_def.slope_max
        ):
            return False
        if (
            bucket_def.slope_min is not None
            and slope_60 < bucket_def.slope_min
        ):
            return False
    if (
        bucket_def.free_fall_slope is not None
        and bucket_def.free_fall_near_delta is not None
        and slope_60 is not None
        and near_delta is not None
        and slope_60 < bucket_def.free_fall_slope
        and near_delta < bucket_def.free_fall_near_delta
    ):
        return False
    if (
        bucket_def.slope_dead_zone_min is not None
        and bucket_def.slope_dead_zone_max is not None
        and slope_60 is not None
        and bucket_def.slope_dead_zone_min
        <= slope_60
        <= bucket_def.slope_dead_zone_max
    ):
        return False
    if bucket_def.v_filter_threshold is not None:
        if (
            above_pv is None
            or above_pv_previous is None
            or above_pv_previous <= bucket_def.v_filter_threshold
            or above_pv >= bucket_def.v_filter_threshold
        ):
            return False
    return True


# ----------------------------------------------------------------------
# Today-slice orchestrator
# ----------------------------------------------------------------------


@dataclass
class AcceptedEntry:
    """Entry-time metadata accepted by live cron and consumed by dashboard."""

    bucket_label: str
    strategy_id: str
    symbol: str
    entry_date: str
    tp_pct: float
    sl_pct: float
    rolling_mp: float
    slope_60: float | None
    near_delta: float | None
    dollar_volume_rank: int
    max_hold: int | None
    reset_hold_on_reentry_signal: bool


@dataclass
class TodaySignalsResult:
    eval_date_string: str
    accepted_per_strategy: Dict[str, List[Dict[str, str]]]
    accepted_records: List[AcceptedEntry]
    rejected_records: List[Tuple[AcceptedEntry, str]]
    log_lines: List[str]


def _read_open_price(
    data_directory: Path, symbol: str, date_string: str
) -> float | None:
    """Return the open price on `date_string` from the per-symbol CSV.
    None when the file or row is missing — caller defers to the next
    daily run."""
    csv_path = data_directory / f"{symbol}.csv"
    if not csv_path.exists():
        return None
    try:
        price_frame = pandas.read_csv(
            csv_path, index_col=0, parse_dates=True
        )
    except Exception:  # noqa: BLE001
        return None
    timestamp = pandas.Timestamp(date_string)
    if timestamp not in price_frame.index:
        return None
    open_column_name = next(
        (column for column in price_frame.columns if column.lower() == "open"),
        None,
    )
    if open_column_name is None:
        return None
    try:
        value = float(price_frame.loc[timestamp, open_column_name])
    except (TypeError, ValueError):
        return None
    if pandas.isna(value):
        return None
    return value


def _execution_date_string(signal_date_string: str) -> str:
    """Map a signal date (T) to the execution date (T+1 business day).
    Mirrors simulator semantics: signals on T, fill at T+1 open."""
    return (
        pandas.Timestamp(signal_date_string) + pandas.offsets.BDay(1)
    ).date().isoformat()


def _bars_held(entry_date_string: str, eval_date_string: str) -> int:
    """Approximate trading bars between entry signal date and the
    evaluation date. 9999 when entry_date is missing — treat as
    "long enough"."""
    if not entry_date_string:
        return 9999
    try:
        entry_timestamp = pandas.Timestamp(entry_date_string)
        evaluation_timestamp = pandas.Timestamp(eval_date_string)
    except (ValueError, TypeError):
        return 9999
    trading_days = pandas.bdate_range(entry_timestamp, evaluation_timestamp)
    return max(0, len(trading_days) - 1)


def _fill_deferred_pcts(
    state: Dict[str, Any],
    data_directory: Path,
) -> List[str]:
    """Iterate `closed_trades` for entries where `raw_pct` is still None
    and try to compute it now. Returns a list of human-readable log
    messages describing fills.

    Why this is needed: when an exit signal fires at end-of-day T, the
    actual execution open price is on T+1's bar — not yet present in the
    CSV at the time of T's cron run. The pct is filled on the next cron
    run after T+1's bar materializes. This mirrors the existing
    `compute_adaptive_tp_sl` defer pattern (manage.py:3683-3713) but
    appends to the new schema's `pending_rolling` list instead of the
    legacy mixed `raw_trade_profits`."""
    log_messages: List[str] = []
    delayed_pending: List[Dict[str, Any]] = state.setdefault("pending_rolling", [])
    for closed_trade in state.get("closed_trades", []):
        if closed_trade.get("raw_pct") is not None:
            continue
        symbol = closed_trade.get("symbol", "")
        entry_date_string = closed_trade.get("entry_date")
        exit_date_string = closed_trade.get("exit_date")
        if not symbol or not entry_date_string or not exit_date_string:
            continue
        entry_open = closed_trade.get("entry_price")
        if entry_open is None:
            entry_open = _read_open_price(
                data_directory, symbol, _execution_date_string(entry_date_string)
            )
            if entry_open is not None:
                closed_trade["entry_price"] = round(entry_open, 4)
        exit_open = closed_trade.get("exit_price")
        if exit_open is None:
            exit_open = _read_open_price(
                data_directory, symbol, _execution_date_string(exit_date_string)
            )
            if exit_open is not None:
                closed_trade["exit_price"] = round(exit_open, 4)
        if (
            entry_open is None
            or exit_open is None
            or float(entry_open) <= 0
        ):
            continue
        pct_value = (float(exit_open) - float(entry_open)) / float(entry_open)
        closed_trade["raw_pct"] = round(pct_value, 6)
        # The simulator's delayed_rolling_update path queues pcts in
        # pending_rolling keyed by close_date so a same-day entry on
        # close_date does not see this pct. Mirror that here using the
        # exit signal date as the close_date.
        delayed_pending.append({
            "closed_date": exit_date_string,
            "pct": float(pct_value),
        })
        # Also feed legacy raw_trade_profits so compute_adaptive_tp_sl can
        # produce a global tp_pct/sl_pct top-level value for System B
        # (place_tp_sl.py).  compute_adaptive_tp_sl skips closed_trades
        # whose raw_pct is already filled, so without this dual-write the
        # legacy rolling pool stays empty under the live multi-bucket cron.
        state.setdefault("raw_trade_profits", []).append(round(pct_value, 6))
        log_messages.append(
            f"  Filled raw_pct for {symbol} (bucket {closed_trade.get('bucket', '?')}): {pct_value:+.2%}"
        )
    return log_messages


def compute_today_signals(
    *,
    config: MultiBucketRunConfig,
    eval_date: pandas.Timestamp,
    held_positions: Dict[str, List[Dict[str, str]]],
    state: Dict[str, Any],
    data_directory: Path,
    allowed_symbols: set[str] | None,
) -> TodaySignalsResult:
    """Reproduce the simulator's single-day decision in production.

    `held_positions` matches `signal_trades.json` shape:
        {strategy_id: [{symbol, entry_date}, ...]}
    `state` matches the new schema_version=2 adaptive_state.json:
        {schema_version, winners, losers, pending_rolling, closed_trades}
    Both are mutated in place; caller is responsible for atomic writes.

    Returns a `TodaySignalsResult` with the post-day virtual signal ledger,
    accepted/rejected entry records, and log lines that the caller writes
    to the cron log.  The log is the dashboard contract: it must contain all
    strategy entry/exit signals plus the frozen TP/SL data needed for order
    previews.  Broker/Futu position reconciliation happens outside cron.
    """
    if config.adaptive_tp_sl is None:
        raise ValueError("adaptive_tp_sl is required for today-slice signal generation")
    adaptive = config.adaptive_tp_sl
    delayed_rolling = adaptive.delayed_rolling_update
    rolling_window = adaptive.window
    eval_date_string = eval_date.date().isoformat()

    log_lines: List[str] = []
    log_lines.append(
        f"[multi_bucket_daily_signal] eval_date={eval_date_string} "
        f"max_position_count={config.maximum_position_count} "
        f"buckets={list(config.bucket_definitions.keys())}"
    )

    # ------------------------------------------------------------------
    # Step A. Try to fill pcts deferred from prior runs. This is the
    # equivalent of the existing `compute_adaptive_tp_sl` first-pass.
    # ------------------------------------------------------------------
    fill_messages = _fill_deferred_pcts(state, data_directory)
    log_lines.extend(fill_messages)

    # ------------------------------------------------------------------
    # Step B. Per-bucket signal generation via compute_signals_for_date.
    # ------------------------------------------------------------------
    per_bucket_signals: Dict[str, Dict[str, Any]] = {}
    dashboard_exit_symbols: List[str] = []
    dashboard_exit_metadata_by_symbol: Dict[str, List[Tuple[str, str]]] = {}
    dashboard_entry_signal_lines: List[str] = []
    for bucket_label, bucket_def in config.bucket_definitions.items():
        signals = strategy.compute_signals_for_date(
            data_directory=data_directory,
            evaluation_date=eval_date,
            buy_strategy_name=bucket_def.buy_strategy_name,
            sell_strategy_name=bucket_def.sell_strategy_name,
            minimum_average_dollar_volume=bucket_def.minimum_average_dollar_volume,
            top_dollar_volume_rank=bucket_def.top_dollar_volume_rank,
            maximum_symbols_per_group=bucket_def.maximum_symbols_per_group,
            minimum_average_dollar_volume_ratio=bucket_def.minimum_average_dollar_volume_ratio,
            allowed_symbols=allowed_symbols,
            skipped_fama_french_groups=bucket_def.skipped_fama_french_groups,
            # Live cron uses signal-day convention (entry_date == the
            # bar the strategy fired on). _fill_deferred_pcts later
            # adds BDay(1) to fetch the actual T+1 open as the fill
            # price, so the rolling pool gets the right raw_pct. Sim's
            # shifted (fill-day) convention applies inside
            # run_complex_simulation; compute_today_signals is the
            # live emitter and matches Cal's "today the signal fired"
            # mental model + the legacy find_history_signal output.
            use_unshifted_signals=True,
            additional_above_ranges=bucket_def.additional_above_ranges,
            exit_alpha_factor=bucket_def.exit_alpha_factor,
        )
        per_bucket_signals[bucket_label] = signals
        log_lines.append(f"--- {bucket_def.strategy_identifier} ---")
        log_lines.append(f"filtered symbols: {signals.get('filtered_symbols', [])}")
        log_lines.append(f"entry signals: {signals.get('entry_signals', [])}")
        log_lines.append(f"exit signals: {signals.get('exit_signals', [])}")
        for entry_symbol in signals.get("entry_signals", []):
            dashboard_entry_signal_lines.append(
                f"[ENTRY_SIGNAL] bucket={bucket_label} "
                f"strategy_id={bucket_def.strategy_identifier} "
                f"symbol={entry_symbol}"
            )
        for exit_symbol in signals.get("exit_signals", []):
            if exit_symbol not in dashboard_exit_metadata_by_symbol:
                dashboard_exit_symbols.append(exit_symbol)
                dashboard_exit_metadata_by_symbol[exit_symbol] = []
            exit_metadata = (
                bucket_label,
                str(bucket_def.strategy_identifier),
            )
            if exit_metadata not in dashboard_exit_metadata_by_symbol[exit_symbol]:
                dashboard_exit_metadata_by_symbol[exit_symbol].append(exit_metadata)

    # ------------------------------------------------------------------
    # Step C. Process held-position exits per bucket. Today's signal
    # exits become closed_trades records with raw_pct=None; the price
    # lookup is deferred to the next daily run via _fill_deferred_pcts.
    # ------------------------------------------------------------------
    new_held_per_strategy: Dict[str, List[Dict[str, str]]] = {}
    same_day_close_count_global = 0
    virtual_closed_symbols_global: List[str] = []
    for bucket_label, bucket_def in config.bucket_definitions.items():
        strategy_identifier = bucket_def.strategy_identifier
        held_for_strategy = held_positions.get(strategy_identifier, [])
        signals = per_bucket_signals[bucket_label]
        filter_exit_set = set(signals.get("exit_signals", []))
        retained: List[Dict[str, str]] = []
        bucket_exit_messages: List[str] = []
        for held_entry in held_for_strategy:
            held_symbol = held_entry.get("symbol", "")
            entry_date_string = held_entry.get("entry_date", "")
            has_exit = held_symbol in filter_exit_set
            if not has_exit:
                try:
                    debug_values = daily_job.filter_debug_values(
                        held_symbol,
                        eval_date_string,
                        bucket_def.buy_strategy_name,
                        bucket_def.sell_strategy_name,
                    )
                except Exception:  # noqa: BLE001
                    debug_values = {}
                has_exit = bool(debug_values.get("exit", False))
            bars_held = _bars_held(entry_date_string, eval_date_string)
            if has_exit and bars_held >= config.minimum_holding_bars:
                state.setdefault("closed_trades", []).append({
                    "symbol": held_symbol,
                    "bucket": bucket_label,
                    "strategy_id": strategy_identifier,
                    "entry_date": entry_date_string,
                    "exit_date": eval_date_string,
                    "entry_price": held_entry.get("entry_price"),
                    "exit_price": None,
                    "raw_pct": None,
                    "exit_reason": "signal",
                })
                bucket_exit_messages.append(held_symbol)
                same_day_close_count_global += 1
            else:
                retained.append(held_entry)
        new_held_per_strategy[strategy_identifier] = retained
        if bucket_exit_messages:
            log_lines.append(
                f"[exit] bucket={bucket_label} symbols={bucket_exit_messages}"
            )
            virtual_closed_symbols_global.extend(bucket_exit_messages)

    # ------------------------------------------------------------------
    # Step D. Flush pending_rolling for entries strictly older than today.
    # ------------------------------------------------------------------
    flush_pending_rolling_into_deques(state, eval_date, rolling_window)

    # ------------------------------------------------------------------
    # Step E. Per-bucket frozen TP/SL + entry candidate collection.
    # Within-bucket order: dollar_volume desc (matches simulator post
    # commit 1240118d). Across-bucket order: bucket_priority asc, then
    # dollar_volume rank asc (lower-priority value wins; lower rank wins).
    # ------------------------------------------------------------------
    candidates: List[Tuple[int, int, str, str, AcceptedEntry]] = []
    for bucket_label, bucket_def in config.bucket_definitions.items():
        strategy_identifier = bucket_def.strategy_identifier
        signals = per_bucket_signals[bucket_label]
        entry_signal_set = set(signals.get("entry_signals", []))
        filtered_symbols = signals.get("filtered_symbols", [])
        held_symbols_in_strategy = {
            entry["symbol"]
            for entry in new_held_per_strategy.get(strategy_identifier, [])
        }
        for dollar_volume_rank, filtered_entry in enumerate(filtered_symbols):
            symbol_name = (
                filtered_entry[0]
                if isinstance(filtered_entry, tuple)
                else filtered_entry
            )
            if symbol_name not in entry_signal_set:
                continue
            # Signal layer is intentionally pure: no held filter here.
            # signal_trades.json is a signal-emission log, not a fill
            # record. Filtering today's signals by yesterday's record
            # created a stale-state bug where a symbol fired once,
            # wrote itself to signal_trades, and never re-fired even
            # though the broker order may never have filled.
            #
            # Dedup against actual holdings is the order layer's job
            # (dashboard's api_preview_orders already filters by Futu
            # positions when building order preview).
            #
            # `held_symbols_in_strategy` above is retained for Step C
            # exit detection / bucket mapping only.
            # Per-bucket pre-cross lookback shifts the A-layer read back
            # one trading bar (mirrors strategy.py:_resolve_trade_decision_dates).
            # Required by fish_head_vacuum_turn so slope_60 / near_delta
            # capture the bar BEFORE the cross — the cross bar already
            # includes the first reaction-up tick. Other buckets read
            # at eval_date itself.
            if bucket_def.pre_cross_signal_lookback:
                signal_lookup_date_string = (
                    (eval_date - pandas.offsets.BDay(1)).date().isoformat()
                )
            else:
                signal_lookup_date_string = eval_date_string
            try:
                debug_values = daily_job.filter_debug_values(
                    symbol_name,
                    signal_lookup_date_string,
                    bucket_def.buy_strategy_name,
                    bucket_def.sell_strategy_name,
                )
            except Exception:  # noqa: BLE001
                debug_values = {}
            slope_60_value = debug_values.get("slope_60")
            near_delta_value = debug_values.get("near_delta")
            above_pv_value = debug_values.get("above_price_volume_ratio")
            above_pv_previous_value = debug_values.get(
                "above_price_volume_ratio_previous"
            )
            if not passes_per_bucket_entry_filters(
                bucket_def,
                slope_60_value,
                near_delta_value,
                above_pv=above_pv_value,
                above_pv_previous=above_pv_previous_value,
            ):
                continue
            (
                tp_pct,
                sl_pct,
                rolling_mp,
                _rolling_ml,
            ) = strategy.compute_frozen_tp_sl_for_bucket(
                bucket_def=bucket_def,
                adaptive_tp_sl=adaptive,
                closed_winners=state.get("winners", []),
                closed_losers=state.get("losers", []),
                entry_slope_60=slope_60_value,
            )
            candidate_record = AcceptedEntry(
                bucket_label=bucket_label,
                strategy_id=strategy_identifier,
                symbol=symbol_name,
                entry_date=eval_date_string,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                rolling_mp=rolling_mp,
                slope_60=slope_60_value if slope_60_value is not None else None,
                near_delta=near_delta_value if near_delta_value is not None else None,
                dollar_volume_rank=dollar_volume_rank,
                max_hold=bucket_def.max_hold,
                reset_hold_on_reentry_signal=(
                    bucket_def.reset_hold_on_reentry_signal
                ),
            )
            candidates.append((
                bucket_def.entry_priority,
                dollar_volume_rank,
                bucket_label,
                symbol_name,
                candidate_record,
            ))

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1], candidate[2], candidate[3]))

    # ------------------------------------------------------------------
    # Step F. Cross-bucket slot competition. Same-day closes count as
    # still-occupying slots (simulator strategy.py:1657-1660 lookahead
    # prevention). Same-symbol cap uses post-close counts because the
    # simulator decrements open_symbol_counts at close events.
    # ------------------------------------------------------------------
    held_total_before_today = sum(len(positions) for positions in held_positions.values())
    held_total_after_today = sum(len(positions) for positions in new_held_per_strategy.values())
    bucket_held_before: Dict[str, int] = {}
    for bucket_label, bucket_def in config.bucket_definitions.items():
        bucket_held_before[bucket_label] = len(
            held_positions.get(bucket_def.strategy_identifier, [])
        )
    held_symbol_counts_after: Dict[str, int] = {}
    for retained_entries in new_held_per_strategy.values():
        for retained_entry in retained_entries:
            symbol_name = retained_entry["symbol"]
            held_symbol_counts_after[symbol_name] = (
                held_symbol_counts_after.get(symbol_name, 0) + 1
            )

    # Slot cap intentionally does NOT subtract signal_trades-tracked
    # held positions. signal_trades is a signal-emission log, not a
    # fill ledger; subtracting it would let yesterday's emissions
    # silently steal today's slots even when the broker order never
    # filled. Cron emits up to max_position_count candidates per day;
    # the order layer (dashboard) reconciles against the real broker
    # portfolio when building actual orders.
    global_remaining = config.maximum_position_count
    bucket_remaining: Dict[str, int] = {}
    for bucket_label, bucket_def in config.bucket_definitions.items():
        cap = (
            bucket_def.maximum_positions
            if bucket_def.maximum_positions is not None
            else config.maximum_position_count
        )
        bucket_remaining[bucket_label] = cap

    accepted_records: List[AcceptedEntry] = []
    rejected_records: List[Tuple[AcceptedEntry, str]] = []
    for _, _, bucket_label, symbol_name, candidate_record in candidates:
        if global_remaining <= 0:
            rejected_records.append((candidate_record, "slot_full"))
            continue
        if bucket_remaining.get(bucket_label, 0) <= 0:
            rejected_records.append((candidate_record, "bucket_cap"))
            continue
        if (
            config.max_same_symbol < 999
            and held_symbol_counts_after.get(symbol_name, 0)
            >= config.max_same_symbol
        ):
            rejected_records.append((candidate_record, "same_symbol"))
            continue
        accepted_records.append(candidate_record)
        global_remaining -= 1
        bucket_remaining[bucket_label] -= 1
        held_symbol_counts_after[symbol_name] = (
            held_symbol_counts_after.get(symbol_name, 0) + 1
        )

    # ------------------------------------------------------------------
    # Step G. Build new signal_trades dict (retained held + new accepted).
    # ------------------------------------------------------------------
    accepted_per_strategy: Dict[str, List[Dict[str, str]]] = {
        strategy_identifier: [
            {"symbol": entry["symbol"], "entry_date": entry.get("entry_date", "")}
            for entry in retained
        ]
        for strategy_identifier, retained in new_held_per_strategy.items()
    }
    for record in accepted_records:
        accepted_per_strategy.setdefault(record.strategy_id, []).append({
            "symbol": record.symbol,
            "entry_date": record.entry_date,
        })

    # ------------------------------------------------------------------
    # Step H. Slot allocation summary + per-position FROZEN_TP_SL log
    # lines for System B parsing.
    # ------------------------------------------------------------------
    log_lines.append("--- multi-bucket slot allocation ---")
    log_lines.append(
        f"max_position_count={config.maximum_position_count} "
        f"held_before_today={held_total_before_today} "
        f"same_day_closes={same_day_close_count_global}"
    )
    log_lines.append(
        f"accepted: {[(record.symbol, record.bucket_label) for record in accepted_records]}"
    )
    log_lines.append(
        f"rejected: {[(record.symbol, record.bucket_label, reason) for record, reason in rejected_records]}"
    )

    # Machine-readable signal lines for dashboard.  These are pure strategy
    # signals over the filtered universe, not real-position decisions.  The
    # dashboard/order layer cross-references these symbols with Futu positions
    # before presenting or sending any order.
    log_lines.extend(dashboard_entry_signal_lines)
    for exit_symbol in dashboard_exit_symbols:
        metadata_values = dashboard_exit_metadata_by_symbol.get(exit_symbol, [])
        bucket_text = ",".join(bucket_label for bucket_label, _ in metadata_values)
        strategy_text = ",".join(
            strategy_identifier for _, strategy_identifier in metadata_values
        )
        log_lines.append(
            f"[EXIT_SIGNAL] symbol={exit_symbol} "
            f"buckets={bucket_text} strategies={strategy_text}"
        )

    if virtual_closed_symbols_global:
        log_lines.append(
            f"[VIRTUAL_CLOSED_FOR_ROLLING] symbols={virtual_closed_symbols_global}"
        )

    log_lines.append(
        f"[ROLLING_TP_SL_STATE] winners={len(state.get('winners', []))} "
        f"losers={len(state.get('losers', []))} "
        f"pending_rolling={len(state.get('pending_rolling', []))} "
        f"closed_trades={len(state.get('closed_trades', []))}"
    )

    for bucket_label, bucket_def in config.bucket_definitions.items():
        (
            bucket_tp_pct,
            bucket_sl_pct,
            bucket_rolling_mp,
            bucket_rolling_ml,
        ) = strategy.compute_frozen_tp_sl_for_bucket(
            bucket_def=bucket_def,
            adaptive_tp_sl=adaptive,
            closed_winners=state.get("winners", []),
            closed_losers=state.get("losers", []),
            entry_slope_60=None,
        )
        max_hold_text = (
            str(bucket_def.max_hold) if bucket_def.max_hold is not None else "None"
        )
        log_lines.append(
            f"[BUCKET_TP_SL] date={eval_date_string} "
            f"bucket={bucket_label} "
            f"strategy_id={bucket_def.strategy_identifier} "
            f"tp_pct={bucket_tp_pct:.6f} sl_pct={bucket_sl_pct:.6f} "
            f"rolling_mp={bucket_rolling_mp:.6f} "
            f"rolling_ml={bucket_rolling_ml:.6f} "
            f"min_hold_tp={adaptive.min_hold_tp} "
            f"min_hold_sl={adaptive.min_hold_sl} "
            f"disable_sl_trigger={adaptive.disable_sl_trigger} "
            f"max_hold={max_hold_text} "
            f"reset_hold_on_reentry_signal={bucket_def.reset_hold_on_reentry_signal}"
        )

    for record in accepted_records:
        slope_text = (
            f"{record.slope_60:.4f}" if record.slope_60 is not None else "None"
        )
        near_delta_text = (
            f"{record.near_delta:.4f}" if record.near_delta is not None else "None"
        )
        log_lines.append(
            f"[FROZEN_TP_SL] entry_date={record.entry_date} "
            f"bucket={record.bucket_label} strategy_id={record.strategy_id} "
            f"symbol={record.symbol} "
            f"dollar_volume_rank={record.dollar_volume_rank} "
            f"tp_pct={record.tp_pct:.6f} sl_pct={record.sl_pct:.6f} "
            f"rolling_mp={record.rolling_mp:.6f} "
            f"slope_60={slope_text} near_delta={near_delta_text} "
            f"min_hold_tp={adaptive.min_hold_tp} "
            f"disable_sl_trigger={adaptive.disable_sl_trigger} "
            f"max_hold={record.max_hold} "
            f"reset_hold_on_reentry_signal={record.reset_hold_on_reentry_signal}"
        )

    # ------------------------------------------------------------------
    # Step I. Persist accepted_entries to state so place_tp_sl.py can
    # look up per-position frozen TP/SL. Carryover preserves prior frozen
    # values for held positions (never re-frozen mid-trade).
    # ------------------------------------------------------------------
    def _resolve_min_hold_sl_for_bucket(bucket_label: str) -> int:
        """Resolve effective min_hold_sl mirroring strategy.py:1999-2014.

        Captured at entry-time so place_tp_sl uses the value that was in
        force when the trade was opened, not the current (possibly later)
        config. This keeps live SL gating in lockstep with the simulator.
        """
        bucket_def_local = config.bucket_definitions.get(bucket_label)
        if bucket_def_local is None:
            return int(config.minimum_holding_bars)
        effective_override_sl = (
            bucket_def_local.override_min_hold_sl_only
            if bucket_def_local.override_min_hold_sl_only is not None
            else adaptive.override_min_hold_sl_only
        )
        if effective_override_sl:
            return int(
                bucket_def_local.min_hold_sl
                if bucket_def_local.min_hold_sl is not None
                else adaptive.min_hold_sl
            )
        return int(config.minimum_holding_bars)

    prior_entries_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (e.get("strategy_id", ""), e.get("symbol", "")): e
        for e in state.get("accepted_entries", [])
    }
    new_records_by_key: Dict[Tuple[str, str], AcceptedEntry] = {
        (rec.strategy_id, rec.symbol): rec for rec in accepted_records
    }
    persisted_entries: List[Dict[str, Any]] = []
    for strategy_identifier, position_list in accepted_per_strategy.items():
        for position_record in position_list:
            symbol_value = position_record.get("symbol", "")
            key = (strategy_identifier, symbol_value)
            new_record = new_records_by_key.get(key)
            if new_record is not None:
                persisted_entries.append({
                    "entry_date": new_record.entry_date,
                    "bucket": new_record.bucket_label,
                    "strategy_id": new_record.strategy_id,
                    "symbol": new_record.symbol,
                    "dollar_volume_rank": int(new_record.dollar_volume_rank),
                    "tp_pct": round(new_record.tp_pct, 6),
                    "sl_pct": round(new_record.sl_pct, 6),
                    "rolling_mp": round(new_record.rolling_mp, 6),
                    "min_hold_sl": _resolve_min_hold_sl_for_bucket(
                        new_record.bucket_label
                    ),
                    "max_hold": new_record.max_hold,
                    "reset_hold_on_reentry_signal": (
                        new_record.reset_hold_on_reentry_signal
                    ),
                    "disable_sl_trigger": bool(adaptive.disable_sl_trigger),
                    "slope_60": (
                        round(new_record.slope_60, 6)
                        if new_record.slope_60 is not None else None
                    ),
                    "near_delta": (
                        round(new_record.near_delta, 6)
                        if new_record.near_delta is not None else None
                    ),
                })
            else:
                # Carryover: held position retains its original frozen
                # tp_pct/sl_pct from when it was first accepted. If no
                # prior record exists, the position is an orphan from
                # the live side (will surface as [ORPHAN_POSITION] in
                # place_tp_sl); we do not synthesize a record here.
                prior = prior_entries_by_key.get(key)
                if prior is not None:
                    # Backfill min_hold_sl on records persisted before
                    # this field existed. Backfill uses CURRENT config —
                    # not strictly "frozen-at-entry" semantics for those
                    # legacy records, but the alternative is None which
                    # would force place_tp_sl to use its hard-coded
                    # default. Acceptable transition cost.
                    if "min_hold_sl" not in prior:
                        bucket_label_for_prior = prior.get(
                            "bucket", ""
                        )
                        prior = {
                            **prior,
                            "min_hold_sl": (
                                _resolve_min_hold_sl_for_bucket(
                                    bucket_label_for_prior
                                )
                            ),
                        }
                    if "disable_sl_trigger" not in prior:
                        prior = {
                            **prior,
                            "disable_sl_trigger": bool(
                                adaptive.disable_sl_trigger
                            ),
                        }
                    persisted_entries.append(prior)
    state["accepted_entries"] = persisted_entries

    return TodaySignalsResult(
        eval_date_string=eval_date_string,
        accepted_per_strategy=accepted_per_strategy,
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        log_lines=log_lines,
    )
