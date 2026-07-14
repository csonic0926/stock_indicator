"""Production today-slice signal generator for multi-bucket configs.

Reads `data/multi_bucket_production.json` (or any compatible config) and
reproduces the simulator's single-day signal mathematics: per-bucket signals,
entry filters, and frozen TP/SL via the shared helper.  It deliberately does
not allocate portfolio slots.  The dashboard owns the one live allocation
pass because only that layer has the current settings plus Futu holdings and
pending orders.
Persists the ADAPTIVE TP/SL virtual trade history in `adaptive_state.json`.
That history is a statistical reference sample, never a portfolio or broker
position ledger.

Design contract: the simulator (run_complex_simulation in strategy.py)
remains the source of truth. Anything in this module that touches
strategy logic must call into shared callables in strategy.py rather
than reimplementing them. Phase 2.2 parity gate compares this module's
output against the simulator's bar-by-bar replay; any divergence is a
blocker.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas

from . import (
    adaptive_tp_sl_virtual_trade_history,
    daily_job,
    futu_trade_metadata,
    simulator,
    strategy,
    symbol_seasoning,
)
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
    # Universe denylist name/path (see manage.SYMBOL_EXCLUDE_LIST_PATHS).
    # Root criterion: symbols with no operating enterprise behind the
    # equity (pure-liquidity crypto proxies) violate the iceberg-position
    # premise and stay excluded even when historically profitable.
    symbol_exclude_list_name: str | None = None
    symbol_seasoning: symbol_seasoning.SymbolSeasoningConfig | None = None
    # WR-gate (phantom) sensor config. The cron only maintains the sensor
    # and emits the per-entry degrading flag; the RS-combine + phantom
    # execution (slot occupancy, exit) live entirely in the order layer
    # (dashboard). None means the gate is unconfigured (cron stays inert).
    wr_gate: "strategy.WRGateConfig | None" = None


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

        cohort_co_movement_gate_config = (
            strategy.parse_cohort_co_movement_gate_config(
                raw_bucket.get("cohort_co_movement_gate"),
                bucket_label=label,
            )
        )

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
            fuel_drawdown_max=(
                float(raw_bucket["fuel_drawdown_max"])
                if "fuel_drawdown_max" in raw_bucket
                and raw_bucket["fuel_drawdown_max"] is not None
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
            cohort_co_movement_gate=cohort_co_movement_gate_config,
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
    seasoning_config = symbol_seasoning.parse_symbol_seasoning_config(
        document.get("symbol_seasoning")
    )

    # WR-gate sensor config. The cron consumes only the sensor-facing
    # fields (sensor_bucket / gated_buckets / window / curve) to maintain
    # the win-rate cross and stamp the per-entry degrading flag. The
    # risk_score_activation_threshold is intentionally NOT applied here —
    # the RS combine belongs to the dashboard's order layer. Mirrors the
    # simulator parse in manage.py so both read the same JSON key.
    wr_gate_config: strategy.WRGateConfig | None = None
    raw_wr_gate = document.get("ft_family_wr_gate")
    if raw_wr_gate is not None:
        wr_gate_config = strategy.WRGateConfig(
            sensor_bucket=str(
                raw_wr_gate.get("sensor_bucket", "fish_tail_production")
            ),
            gated_buckets=tuple(
                raw_wr_gate.get(
                    "gated_buckets",
                    ["fish_tail_production", "fish_tail_squeeze"],
                )
            ),
            window=int(raw_wr_gate.get("window", 12)),
            score_threshold=float(raw_wr_gate.get("score_threshold", 0.5)),
            weight_wr=float(raw_wr_gate.get("weight_wr", 0.5)),
            weight_no_tp=float(raw_wr_gate.get("weight_no_tp", 0.5)),
            weight_max_hold=float(raw_wr_gate.get("weight_max_hold", 0.0)),
            curve=str(raw_wr_gate.get("curve", "score")),
            risk_score_activation_threshold=(
                int(raw_wr_gate["risk_score_activation_threshold"])
                if raw_wr_gate.get("risk_score_activation_threshold")
                is not None
                else None
            ),
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
        symbol_exclude_list_name=document.get("symbol_exclude_list"),
        ff12_data_path_text=(
            str(raw_ff12_data_path_text)
            if raw_ff12_data_path_text is not None
            else None
        ),
        max_same_symbol=int(document.get("max_same_symbol", 1)),
        raw_document=document,
        symbol_seasoning=seasoning_config,
        wr_gate=wr_gate_config,
    )


# ----------------------------------------------------------------------
# ADAPTIVE TP/SL virtual trade history state load / save
# ----------------------------------------------------------------------


def empty_adaptive_tp_sl_virtual_trade_history_state_document() -> Dict[str, Any]:
    """Return an empty persisted document for the statistical history."""

    return {
        "schema_version": SCHEMA_VERSION,
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_KEY: (
            adaptive_tp_sl_virtual_trade_history.empty_adaptive_tp_sl_virtual_trade_history()
        ),
    }


def load_adaptive_tp_sl_virtual_trade_history_state(
    state_path: Path,
) -> Dict[str, Any]:
    """Load the ADAPTIVE TP/SL virtual trade history state document.

    Returns an empty state
    when the file is missing, malformed, or stamped with an older
    schema_version. Caller should treat schema-mismatch as a cold-start
    signal and re-bootstrap via the simulator's --export-state-on-date
    helper rather than silently overwrite."""
    if not state_path.exists():
        return empty_adaptive_tp_sl_virtual_trade_history_state_document()
    try:
        with state_path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except (json.JSONDecodeError, OSError):
        return empty_adaptive_tp_sl_virtual_trade_history_state_document()
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        return empty_adaptive_tp_sl_virtual_trade_history_state_document()
    adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
        state
    )
    return state


def save_adaptive_tp_sl_virtual_trade_history_state_atomically(
    state_path: Path,
    state_document: Dict[str, Any],
) -> None:
    """Atomically persist the ADAPTIVE TP/SL virtual trade history.

    Pending returns flush only on a later signal date.  Losing them would make
    the statistical sensor forget a completed reference trade.
    """

    adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
        state_document
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as temp_file:
        json.dump(state_document, temp_file, indent=2)
    import os as _os
    _os.replace(temp_path, state_path)


# ----------------------------------------------------------------------
# ADAPTIVE TP/SL virtual return-window updates + per-bucket entry filters
# ----------------------------------------------------------------------


def flush_pending_adaptive_tp_sl_virtual_trade_returns(
    adaptive_tp_sl_virtual_trade_history_state: Dict[str, Any],
    eval_date: pandas.Timestamp,
    window: int,
) -> None:
    """Mirror simulator strategy.py:1647-1668.

    `pending_returns` holds (closed_date, pct) entries deposited by virtual
    reference trades
    that closed on/after the entry day they should NOT contribute to (the
    `delayed_rolling_update` invariant). When today's eval_date is strictly
    later than a pending entry's closed_date, the entry is flushed into
    `winner_returns` (pct > 0) or `loser_returns` (pct < 0), each capped at
    `window`.
    """
    pending_returns_key = (
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY
    )
    winner_returns_key = (
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY
    )
    loser_returns_key = (
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY
    )
    if not adaptive_tp_sl_virtual_trade_history_state.get(pending_returns_key):
        return
    winner_returns = list(
        adaptive_tp_sl_virtual_trade_history_state.get(winner_returns_key, [])
    )
    loser_returns = list(
        adaptive_tp_sl_virtual_trade_history_state.get(loser_returns_key, [])
    )
    remaining: List[Dict[str, Any]] = []
    for pending_entry in adaptive_tp_sl_virtual_trade_history_state[
        pending_returns_key
    ]:
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
                winner_returns.append(pct_value)
                if len(winner_returns) > window:
                    winner_returns = winner_returns[-window:]
            elif pct_value < 0:
                loser_returns.append(pct_value)
                if len(loser_returns) > window:
                    loser_returns = loser_returns[-window:]
        else:
            remaining.append(pending_entry)
    adaptive_tp_sl_virtual_trade_history_state[
        winner_returns_key
    ] = winner_returns
    adaptive_tp_sl_virtual_trade_history_state[
        loser_returns_key
    ] = loser_returns
    adaptive_tp_sl_virtual_trade_history_state[pending_returns_key] = remaining


def append_adaptive_tp_sl_virtual_trade_return(
    adaptive_tp_sl_virtual_trade_history_state: Dict[str, Any],
    closed_date: pandas.Timestamp,
    pct: float,
    *,
    delayed: bool,
    window: int,
) -> None:
    """Append one completed virtual reference trade return. When
    `delayed` is true, parks it in `pending_returns` (the simulator's
    delayed_rolling_update path); otherwise flushes directly into
    winner/loser return samples. Mirrors strategy.py:1610-1636.
    """
    pending_returns_key = (
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY
    )
    winner_returns_key = (
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY
    )
    loser_returns_key = (
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY
    )
    if delayed:
        adaptive_tp_sl_virtual_trade_history_state.setdefault(
            pending_returns_key, []
        ).append({
            "closed_date": closed_date.strftime("%Y-%m-%d"),
            "pct": float(pct),
        })
        return
    if pct > 0:
        winner_returns = list(
            adaptive_tp_sl_virtual_trade_history_state.get(
                winner_returns_key, []
            )
        )
        winner_returns.append(float(pct))
        if len(winner_returns) > window:
            winner_returns = winner_returns[-window:]
        adaptive_tp_sl_virtual_trade_history_state[
            winner_returns_key
        ] = winner_returns
    elif pct < 0:
        loser_returns = list(
            adaptive_tp_sl_virtual_trade_history_state.get(
                loser_returns_key, []
            )
        )
        loser_returns.append(float(pct))
        if len(loser_returns) > window:
            loser_returns = loser_returns[-window:]
        adaptive_tp_sl_virtual_trade_history_state[
            loser_returns_key
        ] = loser_returns


def compute_adaptive_tp_sl_virtual_trade_close_for_wr_gate(
    data_directory: Path,
    symbol: str,
    entry_date_string: str,
    entry_price: float,
    horizon_date_string: str,
    tp_pct: float,
    *,
    min_hold_tp: int,
    max_hold: int | None,
) -> tuple[bool, float, str, pandas.Timestamp] | None:
    """Compute a closed ft trade's ADAPTIVE (TP/SL) exit for the WR-gate
    sensor, by reconstructing the entry-relative bar path from the daily
    price cache and replaying it through the simulator's own
    ``_replay_trade_with_adaptive_tp_sl``.

    Production runs SL disabled, so the adaptive exit is TP, max_hold, or
    the signal exit (horizon). Bars start strictly AFTER the entry bar
    (validated byte-for-byte against the simulator: 734/734 ft TP/max_hold
    trades match on exit date, reason and pct). Returns
    ``(win, pct, exit_reason, adaptive_exit_date)`` or None when price
    history is missing.
    """

    price_path = data_directory / f"{symbol}.csv"
    if not price_path.exists() or entry_price <= 0:
        return None
    try:
        frame = pandas.read_csv(price_path, parse_dates=["Date"])
    except (OSError, ValueError):
        return None
    if not {"high", "low", "open"} <= set(frame.columns):
        return None
    frame = frame.set_index("Date").sort_index()
    entry_timestamp = pandas.Timestamp(entry_date_string)
    horizon_timestamp = pandas.Timestamp(horizon_date_string)
    if entry_timestamp not in frame.index:
        return None
    entry_position = frame.index.get_loc(entry_timestamp)
    bar_excursions: list[tuple[pandas.Timestamp, float, float, float]] = []
    for row_position in range(entry_position + 1, len(frame.index)):
        bar_date = frame.index[row_position]
        # Excursions end at the horizon (the signal-exit bar), exactly
        # like the simulator's bar_excursions. A TP/max_hold beyond the
        # horizon must NOT fire — that is the signal exit's territory.
        # max_hold's next-bar-open exit only fires when that next bar is
        # within the horizon (mirrors the sim's next_idx < len check).
        if bar_date > horizon_timestamp:
            break
        bar = frame.iloc[row_position]
        bar_excursions.append((
            bar_date,
            (float(bar["high"]) - entry_price) / entry_price,
            (float(bar["low"]) - entry_price) / entry_price,
            (float(bar["open"]) - entry_price) / entry_price,
        ))
    if not bar_excursions:
        return None
    raw_trade = simulator.Trade(
        entry_date=entry_timestamp,
        exit_date=bar_excursions[-1][0],
        entry_price=entry_price,
        exit_price=entry_price * (1 + bar_excursions[-1][1]),
        profit=0.0,
        holding_period=len(bar_excursions),
        bar_excursions=bar_excursions,
    )
    adaptive_trade = strategy._replay_trade_with_adaptive_tp_sl(
        raw_trade,
        tp_pct=tp_pct,
        sl_pct=0.0,
        minimum_holding_bars=0,
        minimum_holding_bars_tp=min_hold_tp,
        disable_sl_trigger=True,
        max_hold_bars=max_hold,
    )
    adaptive_pct = (
        (adaptive_trade.exit_price - entry_price) / entry_price
    )
    return (
        adaptive_pct > 0,
        adaptive_pct,
        adaptive_trade.exit_reason,
        adaptive_trade.exit_date,
    )


def advance_wr_gate_sensor_from_adaptive_tp_sl_virtual_trade_history(
    state_document: Dict[str, Any],
    adaptive_tp_sl_virtual_trade_history_state: Dict[str, Any],
    gate_config: "strategy.WRGateConfig | None",
    eval_date_string: str,
    data_directory: Path,
) -> List[str]:
    """Feed ft adaptive closes to the WR-gate sensor for any close that
    completed STRICTLY before eval_date, in adaptive-exit-date order, and
    maintain ``wr_gate_pending_ft`` in place.

    Mirrors the simulator: the sensor is fed each ft trade's adaptive
    (TP/max_hold/signal) outcome at its adaptive exit date — NOT at the
    signal-exit date the virtual ledger discovers it. So each daily run
    replays open ft positions to detect TP/max_hold closes, and resolves
    signal closes via the recorded exit date as the replay horizon. The
    strictly-before-eval-date rule preserves no-lookahead (a close on day
    D never feeds D's own entries). No-op without a bootstrapped sensor
    (gate stays off — the safe default until --export-state-on-date seeds
    ``wr_gate_sensor``).
    """

    log_messages: List[str] = []
    if gate_config is None:
        return log_messages
    sensor = state_document.get("wr_gate_sensor")
    if sensor is None:
        return log_messages
    pending: List[Dict[str, Any]] = state_document.setdefault(
        "wr_gate_pending_ft", []
    )
    eval_timestamp = pandas.Timestamp(eval_date_string)
    signal_exit_by_key: Dict[tuple, str] = {}
    for closed_trade in adaptive_tp_sl_virtual_trade_history_state.get(
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY,
        [],
    ):
        if (
            closed_trade.get("bucket") == gate_config.sensor_bucket
            and closed_trade.get("exit_date")
        ):
            signal_exit_by_key[
                (closed_trade.get("symbol"), closed_trade.get("entry_date"))
            ] = (
                closed_trade.get("exit_date"),
                closed_trade.get("raw_pct"),
            )

    feeds: List[tuple] = []
    still_pending: List[Dict[str, Any]] = []
    for pending_adaptive_tp_sl_virtual_trade in pending:
        signal_date_string = pending_adaptive_tp_sl_virtual_trade[
            "signal_date"
        ]
        # Lazy fill resolution: entry fills at signal+1 open.
        fill_date_string = (
            pending_adaptive_tp_sl_virtual_trade.get("fill_date")
            or _execution_date_string(signal_date_string)
        )
        entry_price = pending_adaptive_tp_sl_virtual_trade.get("entry_price")
        if entry_price is None:
            resolved = _read_open_price(
                data_directory,
                pending_adaptive_tp_sl_virtual_trade["symbol"],
                fill_date_string,
            )
            if resolved is not None:
                pending_adaptive_tp_sl_virtual_trade[
                    "fill_date"
                ] = fill_date_string
                pending_adaptive_tp_sl_virtual_trade[
                    "entry_price"
                ] = round(float(resolved), 4)
                entry_price = pending_adaptive_tp_sl_virtual_trade[
                    "entry_price"
                ]
        if entry_price is None:
            still_pending.append(pending_adaptive_tp_sl_virtual_trade)
            continue
        signal_close = signal_exit_by_key.get(
            (
                pending_adaptive_tp_sl_virtual_trade["symbol"],
                signal_date_string,
            )
        )
        signal_exit_string = signal_close[0] if signal_close else None
        # Replay only detects TP / max_hold (which fire before any signal
        # exit). Horizon = the signal exit when known, else today.
        horizon_string = signal_exit_string or eval_date_string
        adaptive = compute_adaptive_tp_sl_virtual_trade_close_for_wr_gate(
            data_directory,
            pending_adaptive_tp_sl_virtual_trade["symbol"],
            fill_date_string,
            float(entry_price),
            horizon_string,
            float(pending_adaptive_tp_sl_virtual_trade["tp_pct"]),
            min_hold_tp=int(
                pending_adaptive_tp_sl_virtual_trade["min_hold_tp"]
            ),
            max_hold=pending_adaptive_tp_sl_virtual_trade.get("max_hold"),
        )
        if adaptive is None:
            still_pending.append(pending_adaptive_tp_sl_virtual_trade)
            continue
        win, pct, reason, adaptive_exit_timestamp = adaptive
        if reason in ("adaptive_take_profit", "max_hold"):
            # TP / max_hold is the true adaptive exit.
            if adaptive_exit_timestamp < eval_timestamp:
                feeds.append((
                    adaptive_exit_timestamp, fill_date_string,
                    pending_adaptive_tp_sl_virtual_trade["symbol"],
                    win,
                    pct,
                    reason,
                ))
            else:
                still_pending.append(pending_adaptive_tp_sl_virtual_trade)
        elif signal_close is not None:
            # No TP/max_hold before the signal exit -> the signal exit IS
            # the adaptive exit; use the recorded raw_pct (the replay's
            # fall-through price is not a real exit).
            signal_raw_pct = signal_close[1]
            signal_exit_timestamp = pandas.Timestamp(signal_exit_string)
            if signal_raw_pct is None:
                still_pending.append(pending_adaptive_tp_sl_virtual_trade)
            elif signal_exit_timestamp < eval_timestamp:
                feeds.append((
                    signal_exit_timestamp, fill_date_string,
                    pending_adaptive_tp_sl_virtual_trade["symbol"],
                    float(signal_raw_pct) > 0,
                    float(signal_raw_pct), "signal",
                ))
            else:
                still_pending.append(pending_adaptive_tp_sl_virtual_trade)
        else:
            # Still open: no TP/max_hold and not signal-closed yet.
            still_pending.append(pending_adaptive_tp_sl_virtual_trade)

    # Feed in adaptive-exit-date order; tie-break by (fill_date, symbol)
    # to approximate the simulator's insertion (entry) order for closes
    # that land on the same date, keeping the EMA path deterministic.
    feeds.sort(key=lambda feed: (feed[0], feed[1]))
    for adaptive_date, _fill, symbol, win, pct, reason in feeds:
        strategy.update_wr_gate_sensor_state(
            sensor, win, pct, gate_config.window
        )
        log_messages.append(
            f"  WR-gate sensor fed {symbol} "
            f"({reason} {adaptive_date.date()}): {pct:+.2%}"
        )
    state_document["wr_gate_pending_ft"] = still_pending
    return log_messages


def build_wr_gate_sensor_summary(
    state_document: Dict[str, Any],
    gate_config: "strategy.WRGateConfig | None",
    fed_this_run: int,
) -> str | None:
    """Return a one-line heartbeat of the WR-gate sensor's current reading,
    or None when the gate is unconfigured or not yet bootstrapped. Printed
    every run so a quiet day (no entries, no closes) still shows the sensor
    is alive and what it reads — distinguishing 'normally silent' from
    'silently broken'. degrading mirrors the flag stamped on entries."""
    from collections import deque

    if gate_config is None:
        return None
    sensor = state_document.get("wr_gate_sensor")
    if sensor is None:
        return "[WR_GATE_SENSOR] status=not_bootstrapped (gate inert until --export-state-on-date)"
    cross_window = sensor.get("cross_window", [])
    window_full = len(cross_window) >= gate_config.window
    ema_value = sensor.get("cross_ema")
    sma_value = (
        sum(cross_window) / len(cross_window) if cross_window else None
    )
    breakeven_value = strategy.compute_dynamic_breakeven_win_rate(
        deque(sensor.get("winner_pcts", [])),
        deque(sensor.get("loser_pcts", [])),
        gate_config.window,
    )
    degrading = False
    if window_full and gate_config.curve == "wr_cross":
        degrading = strategy.evaluate_wr_gate_phantom(sensor, gate_config)
    ema_text = f"{ema_value:.4f}" if ema_value is not None else "None"
    sma_text = f"{sma_value:.4f}" if sma_value is not None else "None"
    breakeven_text = (
        f"{breakeven_value:.4f}" if breakeven_value is not None else "None"
    )
    return (
        f"[WR_GATE_SENSOR] ema={ema_text} sma={sma_text} "
        f"breakeven={breakeven_text} degrading={degrading} "
        f"window={len(cross_window)}/{gate_config.window} "
        f"window_full={window_full} "
        "open_pending="
        f"{len(state_document.get('wr_gate_pending_ft', []))} "
        f"fed_this_run={fed_this_run}"
    )


def register_wr_gate_pending_adaptive_tp_sl_virtual_trade(
    state_document: Dict[str, Any],
    gate_config: "strategy.WRGateConfig | None",
    bucket_label: str,
    symbol: str,
    signal_date_string: str,
    tp_pct: float,
    min_hold_tp: int,
    max_hold: int | None,
) -> None:
    """Register a gated-bucket ADAPTIVE TP/SL reference observation.

    The WR sensor waits for this virtual trade's adaptive close, so
    ``advance_wr_gate_sensor_from_adaptive_tp_sl_virtual_trade_history``
    feeds the sensor when it closes.
    Fill price is resolved lazily on a later run."""

    # Only the sensor bucket feeds the WR cross stream (gated_buckets are
    # phantom-gated by it, but the sensor reads sensor_bucket alone).
    if gate_config is None or bucket_label != gate_config.sensor_bucket:
        return

    # TODO: review
    pending_virtual_trades = state_document.setdefault(
        "wr_gate_pending_ft", []
    )
    already_registered = any(
        pending_virtual_trade.get("symbol") == symbol
        and pending_virtual_trade.get("signal_date") == signal_date_string
        for pending_virtual_trade in pending_virtual_trades
    )
    if already_registered:
        return

    pending_virtual_trades.append({
        "symbol": symbol,
        "signal_date": signal_date_string,
        "tp_pct": float(tp_pct),
        "min_hold_tp": int(min_hold_tp),
        "max_hold": max_hold,
    })


def compute_fuel_drawdown_for_today(
    data_directory: Path,
    symbol_name: str,
    signal_date_string: str,
) -> float | None:
    """Mirror strategy.py:_compute_fuel_drawdown_for_signal for live cron.

    Maximum drawdown (running-max basis) over the pre-surge window
    [T-60, T-11] bars before the signal date, computed from the daily
    price cache. Returns None when the cache lacks 60 bars of history
    before the signal date or the symbol file is missing — the fuel
    gate then skips the candidate (it demands positive evidence).
    """
    csv_path = data_directory / f"{symbol_name}.csv"
    if not csv_path.exists():
        return None
    try:
        frame = pandas.read_csv(csv_path, parse_dates=["Date"])
    except (OSError, ValueError):
        return None
    if "close" not in frame.columns or "Date" not in frame.columns:
        return None
    frame = frame.set_index("Date").sort_index()
    signal_timestamp = pandas.Timestamp(signal_date_string)
    if signal_timestamp not in frame.index:
        return None
    signal_position = frame.index.get_loc(signal_timestamp)
    window_start = signal_position - 60
    if window_start < 0:
        return None
    window_closes = frame["close"].iloc[window_start : signal_position - 10]
    if window_closes.empty or window_closes.isna().any():
        return None
    running_maximum = window_closes.cummax()
    return float((window_closes / running_maximum - 1.0).min())


# TODO: review
def compute_cohort_co_movement_for_today(
    data_directory: Path,
    symbol_name: str,
    signal_date_string: str,
    gate_config: strategy.CohortCoMovementGateConfig,
) -> strategy.TradeDetail:
    """Compute the live-day cohort token with simulator-compatible timing."""

    signal_timestamp = pandas.Timestamp(signal_date_string)
    symbol_to_group_map = strategy.load_ff12_groups_by_symbol()
    target_group_identifier = symbol_to_group_map.get(symbol_name.upper())

    def _read_lookback_return(candidate_symbol: str) -> float | None:
        price_path = data_directory / f"{candidate_symbol}.csv"
        if not price_path.exists():
            return None
        try:
            price_frame = pandas.read_csv(price_path, parse_dates=["Date"])
        except (OSError, ValueError):
            return None
        if "close" not in price_frame.columns or "Date" not in price_frame.columns:
            return None
        close_series = price_frame.set_index("Date").sort_index()["close"]
        known_close_series = close_series[
            close_series.index <= signal_timestamp
        ].dropna()
        if len(known_close_series) < gate_config.lookback_bars + 1:
            return None
        return float(
            known_close_series.iloc[-1]
            / known_close_series.iloc[-(gate_config.lookback_bars + 1)]
            - 1.0
        )

    symbol_return = _read_lookback_return(symbol_name)
    peer_returns: list[float] = []
    if target_group_identifier is not None:
        for peer_symbol, peer_group_identifier in symbol_to_group_map.items():
            if (
                peer_symbol == symbol_name.upper()
                or peer_group_identifier != target_group_identifier
            ):
                continue
            peer_return = _read_lookback_return(peer_symbol)
            if peer_return is not None:
                peer_returns.append(peer_return)
    cohort_median_return: float | None = None
    idiosyncratic_gap: float | None = None
    negative_peer_share: float | None = None
    if peer_returns:
        peer_return_series = pandas.Series(peer_returns)
        cohort_median_return = float(peer_return_series.median())
        negative_peer_share = float((peer_return_series < 0.0).mean())
        if symbol_return is not None:
            idiosyncratic_gap = symbol_return - cohort_median_return
    market_return = _read_lookback_return(strategy.SP500_SYMBOL)
    return strategy.TradeDetail(
        date=signal_timestamp,
        symbol=symbol_name,
        action="open",
        price=0.0,
        simple_moving_average_dollar_volume=0.0,
        total_simple_moving_average_dollar_volume=0.0,
        simple_moving_average_dollar_volume_ratio=0.0,
        cohort_symbol_lookback_return=symbol_return,
        cohort_median_lookback_return=cohort_median_return,
        cohort_market_lookback_return=market_return,
        cohort_idiosyncratic_gap=idiosyncratic_gap,
        cohort_negative_peer_share=negative_peer_share,
        cohort_peer_count=len(peer_returns),
    )


def passes_per_bucket_entry_filters(
    bucket_def: strategy.ComplexStrategySetDefinition,
    slope_60: float | None,
    near_delta: float | None,
    above_pv: float | None = None,
    above_pv_previous: float | None = None,
    fuel_drawdown: float | None = None,
    cohort_entry_detail: strategy.TradeDetail | None = None,
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
    # Squeeze-fuel gate: keep ONLY when the pre-surge window drew down
    # at least to the threshold (mirrors strategy.py event-loop gate).
    if bucket_def.fuel_drawdown_max is not None:
        if (
            fuel_drawdown is None
            or fuel_drawdown > bucket_def.fuel_drawdown_max
        ):
            return False
    if (
        bucket_def.cohort_co_movement_gate is not None
        and cohort_entry_detail is not None
        and strategy.should_skip_for_cohort_co_movement_gate(
            cohort_entry_detail,
            bucket_def.cohort_co_movement_gate,
        )
    ):
        return False
    return True


# ----------------------------------------------------------------------
# Today-slice orchestrator
# ----------------------------------------------------------------------


@dataclass
class TradableEntrySignal:
    """Complete entry metadata for one tradable bucket/symbol signal."""

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
    """Signal-layer output before any live portfolio-slot allocation."""

    eval_date_string: str
    retained_adaptive_tp_sl_virtual_trades_per_strategy: Dict[
        str, List[Dict[str, str]]
    ]
    tradable_records: List[TradableEntrySignal]
    filtered_out_records: List[Tuple[TradableEntrySignal, str]]
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


def resolve_deferred_adaptive_tp_sl_virtual_trade_returns(
    adaptive_tp_sl_virtual_trade_history_state: Dict[str, Any],
    data_directory: Path,
) -> List[str]:
    """Resolve reference-trade returns awaiting the next available open.

    Iterates completed ADAPTIVE TP/SL virtual trades where ``raw_pct`` is
    still ``None``. Returns human-readable log messages describing resolved
    returns.

    Why this is needed: when an exit signal fires at end-of-day T, the
    actual execution open price is on T+1's bar — not yet present in the
    CSV at the time of T's cron run. The pct is filled on the next cron
    run after T+1's bar materializes. This mirrors the existing
    `compute_adaptive_tp_sl` defer pattern (manage.py:3683-3713) but
    appends to the namespaced ``pending_returns`` list instead of the former
    mixed raw-return field."""
    log_messages: List[str] = []
    delayed_pending: List[Dict[str, Any]] = (
        adaptive_tp_sl_virtual_trade_history_state.setdefault(
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY,
            [],
        )
    )
    for closed_trade in adaptive_tp_sl_virtual_trade_history_state.get(
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY,
        [],
    ):
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
        # pending ADAPTIVE TP/SL virtual returns keyed by close_date so a
        # same-day entry on
        # close_date does not see this pct. Mirror that here using the
        # exit signal date as the close_date.
        delayed_pending.append({
            "closed_date": exit_date_string,
            "pct": float(pct_value),
        })
        # Feed the same resolved observation to the raw-return view used by
        # the global ADAPTIVE TP/SL calculation for System B (place_tp_sl.py).
        # This is a second view inside the same statistical history, not an
        # execution or portfolio record.
        adaptive_tp_sl_virtual_trade_history_state.setdefault(
            adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_RAW_RETURNS_KEY,
            [],
        ).append(round(pct_value, 6))
        log_messages.append(
            f"  Filled raw_pct for {symbol} (bucket {closed_trade.get('bucket', '?')}): {pct_value:+.2%}"
        )
    return log_messages


def compute_today_signals_and_advance_adaptive_tp_sl_virtual_trade_history(
    *,
    config: MultiBucketRunConfig,
    eval_date: pandas.Timestamp,
    adaptive_tp_sl_virtual_open_trades_by_strategy: Dict[
        str, List[Dict[str, str]]
    ],
    state_document: Dict[str, Any],
    data_directory: Path,
    allowed_symbols: set[str] | None,
    symbol_first_eligible_trade_dates: Dict[str, datetime.date] | None = None,
) -> TodaySignalsResult:
    """Advance raw-signal history and publish one day's tradable signals.

    ``adaptive_tp_sl_virtual_open_trades_by_strategy`` contains only open
    counterfactual reference trades grouped by strategy::

        {strategy_id: [{symbol, entry_date}, ...]}

    ``state_document`` contains the namespaced ADAPTIVE TP/SL virtual trade
    history plus separate sensor state. Both are mutated in place; the caller
    is responsible for atomic writes.

    Returns a `TodaySignalsResult` with retained open statistical reference
    trades, every tradable entry record, signal-filter rejections, and log
    lines that the caller writes to the cron log.  The log is the dashboard
    contract: it contains every tradable candidate plus the frozen TP/SL data
    needed for one broker-aware allocation pass.  Broker/Futu reconciliation,
    bucket caps, the global cap, and same-symbol competition happen outside
    cron.
    """
    if config.adaptive_tp_sl is None:
        raise ValueError("adaptive_tp_sl is required for today-slice signal generation")
    adaptive_tp_sl_virtual_trade_history_state = (
        adaptive_tp_sl_virtual_trade_history.get_adaptive_tp_sl_virtual_trade_history(
            state_document
        )
    )
    # TODO: review
    # The persisted statistical history is authoritative. The grouped argument
    # remains in the API for compatibility with existing callers, but it must
    # never be allowed to omit and thereby delete a persisted virtual trade.
    persisted_virtual_open_trades = (
        adaptive_tp_sl_virtual_trade_history_state.get(
            adaptive_tp_sl_virtual_trade_history.
            ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY,
            [],
        )
    )
    if persisted_virtual_open_trades:
        adaptive_tp_sl_virtual_open_trades_by_strategy = {}
        for persisted_virtual_open_trade in persisted_virtual_open_trades:
            strategy_identifier = str(
                persisted_virtual_open_trade.get("strategy_id", "")
            )
            if not strategy_identifier:
                continue
            adaptive_tp_sl_virtual_open_trades_by_strategy.setdefault(
                strategy_identifier,
                [],
            ).append(dict(persisted_virtual_open_trade))
    adaptive = config.adaptive_tp_sl
    rolling_window = adaptive.window
    eval_date_string = eval_date.date().isoformat()
    seasoning_enabled = (
        config.symbol_seasoning is not None
        and config.symbol_seasoning.enabled
    )
    if seasoning_enabled and symbol_first_eligible_trade_dates is None:
        raise ValueError(
            "symbol_seasoning is enabled but eligibility dates were not loaded"
        )

    log_lines: List[str] = []
    log_lines.append(
        f"[multi_bucket_daily_signal] eval_date={eval_date_string} "
        f"buckets={list(config.bucket_definitions.keys())}"
    )

    # ------------------------------------------------------------------
    # Step A. Try to fill pcts deferred from prior runs. This is the
    # equivalent of the existing `compute_adaptive_tp_sl` first-pass.
    # ------------------------------------------------------------------
    fill_messages = resolve_deferred_adaptive_tp_sl_virtual_trade_returns(
        adaptive_tp_sl_virtual_trade_history_state,
        data_directory,
    )
    log_lines.extend(fill_messages)

    # ------------------------------------------------------------------
    # Step A2. WR-gate sensor maintenance. Feed any sensor-bucket adaptive
    # close that completed STRICTLY before today into the win-rate cross,
    # so the degrading flag stamped on today's entries reflects the same
    # sensor state the simulator's entry gate reads (no-lookahead). No-op
    # until the gate is configured AND the sensor has been bootstrapped via
    # --export-state-on-date. Runs after Step A so signal closes carry
    # their filled raw_pct.
    # ------------------------------------------------------------------
    sensor_messages = (
        advance_wr_gate_sensor_from_adaptive_tp_sl_virtual_trade_history(
            state_document,
            adaptive_tp_sl_virtual_trade_history_state,
            config.wr_gate,
            eval_date_string,
            data_directory,
        )
    )
    log_lines.extend(sensor_messages)
    fed_this_run = sum(1 for message in sensor_messages if "fed" in message)
    sensor_summary = build_wr_gate_sensor_summary(
        state_document, config.wr_gate, fed_this_run
    )
    if sensor_summary is not None:
        log_lines.append(sensor_summary)

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
            # bar the strategy fired on).
            # resolve_deferred_adaptive_tp_sl_virtual_trade_returns later
            # adds BDay(1) to fetch the actual T+1 open as the fill
            # price, so the rolling pool gets the right raw_pct. Sim's
            # shifted (fill-day) convention applies inside
            # run_complex_simulation; this live signal emitter is the
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
    # exits become ADAPTIVE TP/SL virtual closed-trade records with
    # raw_pct=None; the price
    # lookup is deferred to the next daily run via
    # resolve_deferred_adaptive_tp_sl_virtual_trade_returns.
    # ------------------------------------------------------------------
    retained_adaptive_tp_sl_virtual_trades_by_strategy: Dict[
        str, List[Dict[str, str]]
    ] = {}
    adaptive_tp_sl_virtual_trades_closed_today_count = 0
    virtual_closed_symbols_global: List[str] = []
    for bucket_label, bucket_def in config.bucket_definitions.items():
        strategy_identifier = bucket_def.strategy_identifier
        adaptive_tp_sl_virtual_trades_for_strategy = (
            adaptive_tp_sl_virtual_open_trades_by_strategy.get(
                strategy_identifier,
                [],
            )
        )
        # Buckets may share a strategy_id (fish_tail_squeeze reuses
        # fish_tail_blow_off_top's detection). Each held entry belongs
        # to exactly ONE bucket: its recorded "bucket" field, or the
        # strategy's default bucket for legacy entries written before
        # the field existed. Without this filter a shared strategy_id
        # would process the same held list once per bucket — double
        # exit records and double rolling updates.
        default_bucket_for_strategy = (
            futu_trade_metadata.STRATEGY_ID_TO_DEFAULT_BUCKET.get(
                strategy_identifier, bucket_label
            )
        )
        held_for_bucket = [
            held_entry
            for held_entry in adaptive_tp_sl_virtual_trades_for_strategy
            if held_entry.get("bucket", default_bucket_for_strategy)
            == bucket_label
        ]
        signals = per_bucket_signals[bucket_label]
        filter_exit_set = set(signals.get("exit_signals", []))
        retained: List[Dict[str, str]] = []
        bucket_exit_messages: List[str] = []
        for held_entry in held_for_bucket:
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
                        exit_alpha_factor=bucket_def.exit_alpha_factor,
                    )
                except Exception:  # noqa: BLE001
                    debug_values = {}
                has_exit = bool(debug_values.get("exit", False))
            bars_held = _bars_held(entry_date_string, eval_date_string)
            if has_exit and bars_held >= config.minimum_holding_bars:
                adaptive_tp_sl_virtual_trade_history_state.setdefault(
                    adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY,
                    [],
                ).append({
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
                adaptive_tp_sl_virtual_trades_closed_today_count += 1
            else:
                retained.append(held_entry)
        # Merge (not overwrite): buckets sharing a strategy_id each
        # contribute their own retained slice.
        retained_adaptive_tp_sl_virtual_trades_by_strategy.setdefault(
            strategy_identifier,
            [],
        ).extend(retained)
        if bucket_exit_messages:
            log_lines.append(
                f"[exit] bucket={bucket_label} symbols={bucket_exit_messages}"
            )
            virtual_closed_symbols_global.extend(bucket_exit_messages)

    # ------------------------------------------------------------------
    # Step D. Flush pending ADAPTIVE TP/SL virtual-trade returns for entries
    # strictly older than today.
    # ------------------------------------------------------------------
    flush_pending_adaptive_tp_sl_virtual_trade_returns(
        adaptive_tp_sl_virtual_trade_history_state,
        eval_date,
        rolling_window,
    )

    # ------------------------------------------------------------------
    # Step E. Per-bucket frozen TP/SL + entry candidate collection.
    # Within-bucket order: dollar_volume desc (matches simulator post
    # commit 1240118d). Across-bucket order: bucket_priority asc, then
    # dollar_volume rank asc (lower-priority value wins; lower rank wins).
    # ------------------------------------------------------------------
    filtered_out_records: List[Tuple[TradableEntrySignal, str]] = []
    candidates: List[Tuple[int, int, str, str, TradableEntrySignal]] = []
    for bucket_label, bucket_def in config.bucket_definitions.items():
        strategy_identifier = bucket_def.strategy_identifier
        signals = per_bucket_signals[bucket_label]
        entry_signal_set = set(signals.get("entry_signals", []))
        filtered_symbols = signals.get("filtered_symbols", [])
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
            # Fuel drawdown only loads price history when the bucket
            # actually gates on it (fish_tail_squeeze).
            fuel_drawdown_value = (
                compute_fuel_drawdown_for_today(
                    data_directory,
                    symbol_name,
                    signal_lookup_date_string,
                )
                if bucket_def.fuel_drawdown_max is not None
                else None
            )
            cohort_entry_detail = (
                compute_cohort_co_movement_for_today(
                    data_directory,
                    symbol_name,
                    signal_lookup_date_string,
                    bucket_def.cohort_co_movement_gate,
                )
                if bucket_def.cohort_co_movement_gate is not None
                else None
            )
            if not passes_per_bucket_entry_filters(
                bucket_def,
                slope_60_value,
                near_delta_value,
                above_pv=above_pv_value,
                above_pv_previous=above_pv_previous_value,
                fuel_drawdown=fuel_drawdown_value,
                cohort_entry_detail=cohort_entry_detail,
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
                closed_winners=(
                    adaptive_tp_sl_virtual_trade_history_state.get(
                        adaptive_tp_sl_virtual_trade_history.
                        ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY,
                        [],
                    )
                ),
                closed_losers=(
                    adaptive_tp_sl_virtual_trade_history_state.get(
                        adaptive_tp_sl_virtual_trade_history.
                        ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY,
                        [],
                    )
                ),
                entry_slope_60=slope_60_value,
            )
            candidate_record = TradableEntrySignal(
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
            if seasoning_enabled and not symbol_seasoning.is_symbol_eligible_on(
                symbol_name,
                eval_date,
                symbol_first_eligible_trade_dates or {},
            ):
                filtered_out_records.append(
                    (candidate_record, "symbol_seasoning")
                )
                continue
            candidates.append((
                bucket_def.entry_priority,
                dollar_volume_rank,
                bucket_label,
                symbol_name,
                candidate_record,
            ))

    candidates.sort(
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[2],
            candidate[3],
        )
    )

    # ------------------------------------------------------------------
    # Step F. Publish every tradable candidate.  This layer must not consume
    # global slots, bucket slots, or same-symbol capacity: doing that without
    # Futu state creates a first allocation pass that can discard the lower
    # candidate needed when the dashboard rejects a higher one.  Candidate
    # ordering is still deterministic so the dashboard can apply the exact
    # configured greedy competition after loading live state.
    # ------------------------------------------------------------------
    adaptive_tp_sl_virtual_open_trades_before_today_count = sum(
        len(virtual_trades)
        for virtual_trades in (
            adaptive_tp_sl_virtual_open_trades_by_strategy.values()
        )
    )
    tradable_records = [candidate[-1] for candidate in candidates]

    # ------------------------------------------------------------------
    # Step G. Carry forward the ADAPTIVE TP/SL virtual reference trades that
    # were already open and did not close in Step C. Today's tradable signals
    # are added in Step I as new statistical observations. No portfolio slot,
    # dashboard decision, Futu position, or order outcome participates here.
    # ------------------------------------------------------------------
    retained_adaptive_tp_sl_virtual_trades_per_strategy: Dict[
        str, List[Dict[str, str]]
    ] = {
        strategy_identifier: [
            {
                "symbol": entry["symbol"],
                "entry_date": entry.get("entry_date", ""),
                # Preserve bucket attribution so buckets sharing a
                # strategy_id keep their reference trades separable.
                **(
                    {"bucket": entry["bucket"]}
                    if entry.get("bucket")
                    else {}
                ),
            }
            for entry in retained
        ]
        for strategy_identifier, retained in (
            retained_adaptive_tp_sl_virtual_trades_by_strategy.items()
        )
    }

    # ------------------------------------------------------------------
    # Step H. Tradable-candidate summary + per-candidate FROZEN_TP_SL log
    # lines for the dashboard/order layer.
    # ------------------------------------------------------------------
    log_lines.append("--- multi-bucket tradable candidates ---")
    log_lines.append(
        "adaptive_tp_sl_virtual_open_trades_before_today="
        f"{adaptive_tp_sl_virtual_open_trades_before_today_count} "
        "adaptive_tp_sl_virtual_trades_closed_today="
        f"{adaptive_tp_sl_virtual_trades_closed_today_count}"
    )
    log_lines.append(
        "tradable_candidates: "
        f"{[(record.symbol, record.bucket_label) for record in tradable_records]}"
    )
    filtered_out_summary = [
        (record.symbol, record.bucket_label, reason)
        for record, reason in filtered_out_records
    ]
    log_lines.append(
        "filtered_out: "
        f"{filtered_out_summary}"
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
            "[ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_CLOSED] "
            f"symbols={virtual_closed_symbols_global}"
        )

    winner_returns = adaptive_tp_sl_virtual_trade_history_state.get(
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_WINNER_RETURNS_KEY,
        [],
    )
    loser_returns = adaptive_tp_sl_virtual_trade_history_state.get(
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_LOSER_RETURNS_KEY,
        [],
    )
    pending_returns = adaptive_tp_sl_virtual_trade_history_state.get(
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_PENDING_RETURNS_KEY,
        [],
    )
    closed_virtual_trades = adaptive_tp_sl_virtual_trade_history_state.get(
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_CLOSED_TRADES_KEY,
        [],
    )
    log_lines.append(
        "[ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_STATE] "
        f"winner_returns={len(winner_returns)} "
        f"loser_returns={len(loser_returns)} "
        f"pending_returns={len(pending_returns)} "
        f"closed_trades={len(closed_virtual_trades)}"
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
            closed_winners=winner_returns,
            closed_losers=loser_returns,
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

    # WR-gate degrading flag. The sensor is identical for every entry on
    # this run (it advanced once, before any entry decision), so evaluate
    # it once. The flag is the cron's endogenous half of the phantom
    # decision; the dashboard ANDs it with the month's risk score and owns
    # the phantom execution (slot occupancy + exit). Stamped only on gated
    # buckets — non-gated entries always read wr_degrading=False.
    wr_gate_degrading = False
    if config.wr_gate is not None:
        sensor_state = state_document.get("wr_gate_sensor")
        if sensor_state is not None:
            wr_gate_degrading = strategy.evaluate_wr_gate_phantom(
                sensor_state, config.wr_gate
            )
    for record in tradable_records:
        slope_text = (
            f"{record.slope_60:.4f}" if record.slope_60 is not None else "None"
        )
        near_delta_text = (
            f"{record.near_delta:.4f}" if record.near_delta is not None else "None"
        )
        record_degrading = (
            wr_gate_degrading
            and config.wr_gate is not None
            and record.bucket_label in config.wr_gate.gated_buckets
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
            f"wr_degrading={record_degrading} "
            f"reset_hold_on_reentry_signal={record.reset_hold_on_reentry_signal}"
        )

    # ------------------------------------------------------------------
    # Step I. Persist retained reference trades and start one new ADAPTIVE
    # TP/SL virtual trade for every tradable signal. These observations follow
    # the raw strategy through its eventual signal exit even when dashboard
    # allocation rejects the signal or a real order never fills.
    # ------------------------------------------------------------------
    def _resolve_min_hold_sl_for_bucket(bucket_label: str) -> int:
        """Resolve effective min_hold_sl mirroring strategy.py:1999-2014.

        Captured at signal-time so the virtual reference trade retains the
        value that was in force when its statistical observation began.
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

    prior_virtual_trades_by_key: Dict[
        Tuple[str, str, str, str], Dict[str, Any]
    ] = {
        (
            adaptive_tp_sl_virtual_open_trade.get("strategy_id", ""),
            adaptive_tp_sl_virtual_open_trade.get("bucket", ""),
            adaptive_tp_sl_virtual_open_trade.get("symbol", ""),
            adaptive_tp_sl_virtual_open_trade.get("entry_date", ""),
        ): adaptive_tp_sl_virtual_open_trade
        for adaptive_tp_sl_virtual_open_trade in (
            adaptive_tp_sl_virtual_trade_history_state.get(
                adaptive_tp_sl_virtual_trade_history.
                ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY,
                [],
            )
        )
    }
    persisted_adaptive_tp_sl_virtual_open_trades: List[Dict[str, Any]] = []
    persisted_virtual_trade_keys: set[Tuple[str, str, str, str]] = set()
    for strategy_identifier, adaptive_tp_sl_virtual_trade_list in (
        retained_adaptive_tp_sl_virtual_trades_per_strategy.items()
    ):
        for adaptive_tp_sl_virtual_trade_record in (
            adaptive_tp_sl_virtual_trade_list
        ):
            symbol_value = adaptive_tp_sl_virtual_trade_record.get(
                "symbol", ""
            )
            bucket_label = adaptive_tp_sl_virtual_trade_record.get(
                "bucket", ""
            )
            entry_date_string = adaptive_tp_sl_virtual_trade_record.get(
                "entry_date", ""
            )
            key = (
                strategy_identifier,
                bucket_label,
                symbol_value,
                entry_date_string,
            )
            if key in persisted_virtual_trade_keys:
                continue
            prior = prior_virtual_trades_by_key.get(key)
            if prior is None:
                continue
            # Backfill fields on legacy virtual-history records only.
            if "min_hold_sl" not in prior:
                prior = {
                    **prior,
                    "min_hold_sl": _resolve_min_hold_sl_for_bucket(
                        bucket_label
                    ),
                }
            if "disable_sl_trigger" not in prior:
                prior = {
                    **prior,
                    "disable_sl_trigger": bool(
                        adaptive.disable_sl_trigger
                    ),
                }
            persisted_adaptive_tp_sl_virtual_open_trades.append(prior)
            persisted_virtual_trade_keys.add(key)

    # TODO: review
    newly_started_virtual_trade_count = 0
    for tradable_record in tradable_records:
        virtual_trade_key = (
            tradable_record.strategy_id,
            tradable_record.bucket_label,
            tradable_record.symbol,
            tradable_record.entry_date,
        )
        if virtual_trade_key in persisted_virtual_trade_keys:
            continue

        persisted_adaptive_tp_sl_virtual_open_trades.append({
            "entry_date": tradable_record.entry_date,
            "bucket": tradable_record.bucket_label,
            "strategy_id": tradable_record.strategy_id,
            "symbol": tradable_record.symbol,
            "dollar_volume_rank": tradable_record.dollar_volume_rank,
            "tp_pct": tradable_record.tp_pct,
            "sl_pct": tradable_record.sl_pct,
            "rolling_mp": tradable_record.rolling_mp,
            "min_hold_sl": _resolve_min_hold_sl_for_bucket(
                tradable_record.bucket_label
            ),
            "max_hold": tradable_record.max_hold,
            "reset_hold_on_reentry_signal": (
                tradable_record.reset_hold_on_reentry_signal
            ),
            "disable_sl_trigger": bool(adaptive.disable_sl_trigger),
            "slope_60": tradable_record.slope_60,
            "near_delta": tradable_record.near_delta,
        })
        persisted_virtual_trade_keys.add(virtual_trade_key)
        newly_started_virtual_trade_count += 1
        register_wr_gate_pending_adaptive_tp_sl_virtual_trade(
            state_document,
            config.wr_gate,
            tradable_record.bucket_label,
            tradable_record.symbol,
            tradable_record.entry_date,
            tradable_record.tp_pct,
            adaptive.min_hold_tp,
            tradable_record.max_hold,
        )

    adaptive_tp_sl_virtual_trade_history_state[
        adaptive_tp_sl_virtual_trade_history.ADAPTIVE_TP_SL_VIRTUAL_OPEN_TRADES_KEY
    ] = persisted_adaptive_tp_sl_virtual_open_trades

    log_lines.append(
        "[ADAPTIVE_TP_SL_VIRTUAL_TRADE_HISTORY_ADMISSION] "
        f"tradable_signals={len(tradable_records)} "
        f"new_open_trades={newly_started_virtual_trade_count} "
        "source=cron_tradable_signals"
    )

    adaptive_tp_sl_virtual_open_trades_after_today: Dict[
        str, List[Dict[str, str]]
    ] = {}
    for persisted_virtual_trade in (
        persisted_adaptive_tp_sl_virtual_open_trades
    ):
        strategy_identifier = str(
            persisted_virtual_trade.get("strategy_id", "")
        )
        if not strategy_identifier:
            continue
        adaptive_tp_sl_virtual_open_trades_after_today.setdefault(
            strategy_identifier,
            [],
        ).append({
            "symbol": str(persisted_virtual_trade.get("symbol", "")),
            "entry_date": str(
                persisted_virtual_trade.get("entry_date", "")
            ),
            **(
                {"bucket": str(persisted_virtual_trade["bucket"])}
                if persisted_virtual_trade.get("bucket")
                else {}
            ),
        })

    return TodaySignalsResult(
        eval_date_string=eval_date_string,
        retained_adaptive_tp_sl_virtual_trades_per_strategy=(
            adaptive_tp_sl_virtual_open_trades_after_today
        ),
        tradable_records=tradable_records,
        filtered_out_records=filtered_out_records,
        log_lines=log_lines,
    )
