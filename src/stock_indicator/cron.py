"""Helpers for parsing daily-job arguments and executing tasks.

This module contains only lightweight utilities used by ``daily_job`` and
related modules. Scheduling responsibilities are handled elsewhere.
"""
# TODO: review

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas

from .symbols import update_symbol_cache, load_symbols
from .data_loader import download_history
from .strategy import (
    SUPPORTED_STRATEGIES,
    load_ff12_groups_by_symbol,
    compute_signals_for_date,
)

LOGGER = logging.getLogger(__name__)


def parse_daily_task_arguments(argument_line: str) -> Tuple[
    float | None,
    int | None,
    int,
    str,
    str,
    float,
    set[int] | None,
]:
    """Parse a cron job argument string.

    The expected format is ``dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``,
    ``dollar_volume>NUMBER% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``,
    ``dollar_volume=TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]`` (or legacy ``Nth``),
    ``dollar_volume>NUMBER,TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]`` (or
    legacy ``,Nth``), ``dollar_volume>NUMBER%,TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]``
    (or legacy ``,Nth``), and the optional trailing ``,PickM`` token that limits
    selections to at most ``M`` symbols per sector. Matching is case-insensitive
    for ``TopN`` and ``PickM``.

    Parameters
    ----------
    argument_line: str
        Argument string describing filters and strategy names.

    Returns
    -------
    Tuple[float | None, int | None, int, str, str, float, set[int] | None]
        Tuple containing either a minimum dollar volume threshold in millions
        or a ratio of the total market, followed by the ranking position, the
        maximum number of symbols per group, the buy strategy name, the sell
        strategy name, the stop loss percentage, and any allowed FF12 group
        identifiers.
    """
    argument_parts = argument_line.split()
    if len(argument_parts) not in (3, 4, 5):
        raise ValueError(
            "argument_line must be 'dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume>NUMBER% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume=TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]' (or 'Nth'), "
            "'dollar_volume>NUMBER,TopN' (or ',Nth'), or 'dollar_volume>NUMBER%,TopN' (or ',Nth')",
        )
    # Allow an optional group=1,2,... token to appear anywhere in the first
    # few arguments. Extract it if present, then parse the remaining tokens as
    # volume filter, buy strategy, sell strategy, [stop loss].
    allowed_groups: set[int] | None = None
    tokens: list[str] = []
    for token in argument_parts:
        if token.startswith("group="):
            try:
                raw = token.split("=", 1)[1]
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                parsed = {int(p) for p in parts}
            except ValueError as parse_error:  # noqa: BLE001
                raise ValueError(
                    "Invalid group list; expected positive integers"
                ) from parse_error
            if any(identifier < 1 for identifier in parsed):
                raise ValueError("Group identifiers must be positive integers")
            allowed_groups = parsed
        else:
            tokens.append(token)
    if len(tokens) not in (3, 4):
        raise ValueError(
            "Unsupported argument format; expected DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] with optional group=...",
        )
    volume_filter, buy_strategy_name, sell_strategy_name = tokens[:3]
    stop_loss_percentage = float(tokens[3]) if len(tokens) == 4 else 1.0
    minimum_average_dollar_volume: float | None = None
    minimum_average_dollar_volume_ratio: float | None = None
    top_dollar_volume_rank: int | None = None
    maximum_symbols_per_group: int = 1

    pick_match = re.fullmatch(r"(.*),Pick(\d+)", volume_filter, flags=re.IGNORECASE)
    if pick_match is not None:
        volume_filter = pick_match.group(1)
        maximum_symbols_per_group = int(pick_match.group(2))
    # Accept both TopN and legacy Nth; case-insensitive for Top
    combined_percentage_top_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,Top(\d+)",
        volume_filter,
        flags=re.IGNORECASE,
    )
    combined_percentage_nth_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,(\d+)th",
        volume_filter,
    )
    if combined_percentage_top_match is not None or combined_percentage_nth_match is not None:
        match_obj = combined_percentage_top_match or combined_percentage_nth_match
        minimum_average_dollar_volume_ratio = float(match_obj.group(1)) / 100
        top_dollar_volume_rank = int(match_obj.group(2))
    else:
        combined_top_match = re.fullmatch(
            r"dollar_volume>(\d+(?:\.\d+)?),Top(\d+)",
            volume_filter,
            flags=re.IGNORECASE,
        )
        combined_nth_match = re.fullmatch(
            r"dollar_volume>(\d+(?:\.\d+)?),(\d+)th",
            volume_filter,
        )
        if combined_top_match is not None or combined_nth_match is not None:
            match_obj = combined_top_match or combined_nth_match
            minimum_average_dollar_volume = float(match_obj.group(1))
            top_dollar_volume_rank = int(match_obj.group(2))
        else:
            percentage_match = re.fullmatch(
                r"dollar_volume>(\d+(?:\.\d{1,2})?)%",
                volume_filter,
            )
            if percentage_match is not None:
                minimum_average_dollar_volume_ratio = float(percentage_match.group(1)) / 100
            else:
                volume_match = re.fullmatch(
                    r"dollar_volume>(\d+(?:\.\d+)?)",
                    volume_filter,
                )
                if volume_match is not None:
                    minimum_average_dollar_volume = float(volume_match.group(1))
                else:
                    rank_top_match = re.fullmatch(
                        r"dollar_volume=Top(\d+)",
                        volume_filter,
                        flags=re.IGNORECASE,
                    )
                    rank_nth_match = re.fullmatch(
                        r"dollar_volume=(\d+)th",
                        volume_filter,
                    )
                    if rank_top_match is not None or rank_nth_match is not None:
                        top_dollar_volume_rank = int((rank_top_match or rank_nth_match).group(1))
                    else:
                        raise ValueError(
                            "Unsupported filter format. Expected 'dollar_volume>NUMBER', "
                            "'dollar_volume>NUMBER%', 'dollar_volume=TopN' (or 'Nth'), "
                            "'dollar_volume>NUMBER,TopN' (or ',Nth'), or "
                            "'dollar_volume>NUMBER%,TopN' (or ',Nth').",
                        )
    return (
        minimum_average_dollar_volume_ratio
        if minimum_average_dollar_volume_ratio is not None
        else minimum_average_dollar_volume,
        top_dollar_volume_rank,
        maximum_symbols_per_group,
        buy_strategy_name,
        sell_strategy_name,
        stop_loss_percentage,
        allowed_groups,
    )


def run_daily_tasks(
    buy_strategy_name: str,
    sell_strategy_name: str,
    start_date: str,
    end_date: str,
    symbol_list: Iterable[str] | None = None,
    data_download_function: Callable[[str, str, str], pandas.DataFrame] = download_history,
    data_directory: Path | None = None,
    minimum_average_dollar_volume: float | None = None,
    top_dollar_volume_rank: int | None = None,  # TODO: review
    allowed_fama_french_groups: set[int] | None = None,
    maximum_symbols_per_group: int = 1,
    use_unshifted_signals: bool = False,
    near_delta_range: tuple[float, float] | None = None,
    price_tightness_range: tuple[float, float] | None = None,
) -> Dict[str, List[str]]:
    """Execute the daily workflow using simulation-grade selection logic.

    This implementation aligns with :func:`strategy.evaluate_combined_strategy`
    by applying group-aware ratio thresholds and Top-N selection with an
    optional per-group cap controlled by ``maximum_symbols_per_group``. It
    returns the symbols that have signals on the last available bar at or
    before ``end_date``.

    Notes
    -----
    The parameters ``symbol_list`` and ``data_download_function`` remain in the
    signature for backward compatibility but are not used in this computation.
    Local CSV data under ``data_directory`` is required. When
    ``use_unshifted_signals`` is ``True``, strategy helpers may provide
    ``*_raw_entry_signal`` and ``*_raw_exit_signal`` columns representing
    same-day signals.
    """
    # Determine the evaluation day as the last bar within [start_date, end_date)
    try:
        if use_unshifted_signals:
            evaluation_date = pandas.Timestamp(end_date)
        else:
            evaluation_date = pandas.Timestamp(end_date) - pandas.Timedelta(days=1)
    except Exception:  # noqa: BLE001
        evaluation_date = pandas.Timestamp(end_date)

    # Resolve data directory (default to project data/stock_data if None)
    if data_directory is None:
        data_directory = Path(__file__).resolve().parent.parent.parent / "data" / "stock_data"

    # Interpret minimum_average_dollar_volume parameter: either millions or ratio
    absolute_threshold_millions: float | None = None
    ratio_threshold: float | None = None
    if minimum_average_dollar_volume is not None:
        value = float(minimum_average_dollar_volume)
        if 0 < value < 1.0:
            ratio_threshold = value
        else:
            absolute_threshold_millions = value

    allowed_symbol_set: set[str] | None = set(symbol_list) if symbol_list is not None else None

    if symbol_list is not None and data_directory is not None:
        for symbol_name in symbol_list:
            csv_file_path = data_directory / f"{symbol_name}.csv"
            try:
                data_download_function(
                    symbol_name, start_date, end_date, cache_path=csv_file_path
                )
            except Exception as download_error:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to refresh data for %s: %s", symbol_name, download_error
                )

    return compute_signals_for_date(
        data_directory=data_directory,
        evaluation_date=evaluation_date,
        buy_strategy_name=buy_strategy_name,
        sell_strategy_name=sell_strategy_name,
        minimum_average_dollar_volume=absolute_threshold_millions,
        top_dollar_volume_rank=top_dollar_volume_rank,
        minimum_average_dollar_volume_ratio=ratio_threshold,
        allowed_fama_french_groups=allowed_fama_french_groups,
        allowed_symbols=allowed_symbol_set,
        exclude_other_ff12=True,
        maximum_symbols_per_group=maximum_symbols_per_group,
        use_unshifted_signals=use_unshifted_signals,
        near_delta_range=near_delta_range,
        price_tightness_range=price_tightness_range,
    )
