"""Interactive shell for managing symbol cache and historical data."""

# TODO: review

from __future__ import annotations

import cmd
import csv
import datetime
import gc  # TODO: review
import json
import logging
import re
import shlex
import sys  # TODO: review
import time  # TODO: review
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import pandas
import yfinance  # TODO: review
from pandas import DataFrame

from . import data_loader, symbols, strategy, daily_job, multi_bucket_today
from . import production_ff12_promotion
from . import symbol_seasoning
from . import universe_pipeline
from .simulator import calc_commission
from .strategy_sets import load_strategy_set_mapping, load_strategy_entry_filters
from .daily_job import determine_start_date
from .symbols import SP500_SYMBOL
from stock_indicator.sector_pipeline.overrides import (
    assign_symbol_to_other_if_missing,
)
from stock_indicator.sector_pipeline import pipeline

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
# Live trading state (cron rolling pool + dashboard signal_trades mirror) is
# kept under data/live_state/ so simulation cleanups inside data/ cannot
# accidentally wipe production runtime files.
LIVE_STATE_DIRECTORY = DATA_DIRECTORY / "live_state"
# Store downloaded per-symbol CSVs under a dedicated subfolder to avoid mixing
# with other project CSVs (e.g., sector exports).
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"

# Named data sources for backtesting.  The cron job always uses "stock_data"
# (daily 6-month cache).  Exploratory and full backtests select a source via
# the data_source= token in simulation commands.  Retired historical dataset
# directories are intentionally not listed here so commands fail loudly instead
# of reading stale local research data.
DATA_SOURCE_PATHS: dict[str, Path] = {
    "daily": DATA_DIRECTORY / "stock_data",
    "2010": DATA_DIRECTORY / "stock_data_2010_yf_clean",
    "2010_yf_clean": DATA_DIRECTORY / "stock_data_2010_yf_clean",
    "1994_clean": DATA_DIRECTORY / "stock_data_1994_clean",
}

SYMBOL_LIST_PATHS: dict[str, Path] = {
    "current_stock_universe": DATA_DIRECTORY / "symbols.txt",
    "production": DATA_DIRECTORY / "production_symbols.txt",
    "production_candidate": DATA_DIRECTORY / "production_candidate_symbols.txt",
}


def resolve_data_source(source_name: str | None) -> Path:
    """Return the stock data directory for the given source name.

    Falls back to ``stock_data_2010_yf_clean`` when *source_name* is ``None``.
    """
    if source_name is None:
        source_name = "2010"
    path = DATA_SOURCE_PATHS.get(source_name)
    if path is None:
        raise ValueError(
            f"unknown data source '{source_name}', "
            f"choose from: {', '.join(sorted(DATA_SOURCE_PATHS))}"
        )
    return path


def load_symbol_list(symbol_list_name: str | None) -> set[str] | None:
    """Return an optional symbol whitelist by configured name or file path."""

    if not symbol_list_name:
        return None
    symbol_list_path = SYMBOL_LIST_PATHS.get(symbol_list_name)
    if symbol_list_path is None:
        symbol_list_path = Path(symbol_list_name).expanduser()
    if not symbol_list_path.exists():
        raise ValueError(f"symbol list not found: {symbol_list_path}")
    symbols_to_keep: set[str] = set()
    for line_text in symbol_list_path.read_text(encoding="utf-8").splitlines():
        symbol_name = line_text.strip().upper()
        if symbol_name:
            symbols_to_keep.add(symbol_name)
    return symbols_to_keep


def resolve_ff12_data_path(ff12_data_path_text: object | None) -> Path | None:
    """Return the configured FF12 override path, if one was supplied."""

    if not ff12_data_path_text:
        return None
    ff12_data_path = Path(str(ff12_data_path_text)).expanduser()
    if not ff12_data_path.is_absolute():
        ff12_data_path = _resolve_repository_relative_path(ff12_data_path)
    if not ff12_data_path.exists():
        raise ValueError(f"ff12_data_path not found: {ff12_data_path}")
    return ff12_data_path


def _resolve_repository_relative_path(path_text: object) -> Path:
    """Resolve a path relative to the repository root when needed."""

    resolved_path = Path(str(path_text)).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    repository_root = Path(__file__).resolve().parent.parent.parent
    return repository_root / resolved_path


def load_symbol_seasoning_dates_for_config(
    seasoning_config: symbol_seasoning.SymbolSeasoningConfig,
    *,
    data_directory: Path | None = None,
    allowed_symbols: set[str] | None = None,
) -> tuple[Path, dict[str, datetime.date]] | None:
    """Load symbol seasoning dates when the configured gate is enabled."""

    if not seasoning_config.enabled:
        return None
    if (
        seasoning_config.eligibility_source
        == symbol_seasoning.PRICE_HISTORY_ELIGIBILITY_SOURCE
    ):
        if data_directory is None:
            raise ValueError(
                "symbol_seasoning.eligibility_source=price_history requires "
                "a data directory"
            )
        symbol_first_eligible_trade_dates = (
            symbol_seasoning.build_symbol_first_eligible_trade_dates_from_price_history(
                data_directory,
                allowed_symbols=allowed_symbols,
                quarantine_calendar_days=(
                    seasoning_config.default_new_symbol_quarantine_days
                ),
                quarantine_trading_bars=seasoning_config.quarantine_trading_bars,
            )
        )
        return data_directory, symbol_first_eligible_trade_dates

    repository_root = Path(__file__).resolve().parent.parent.parent
    eligibility_path = symbol_seasoning.resolve_eligibility_path(
        seasoning_config,
        repository_root=repository_root,
    )
    symbol_first_eligible_trade_dates = (
        symbol_seasoning.load_symbol_first_eligible_trade_dates(
            eligibility_path
        )
    )
    return eligibility_path, symbol_first_eligible_trade_dates


def load_risk_score_priority_overrides(
    raw_priority_overrides: object | None,
    raw_risk_score_gate: object | None,
    bucket_labels: set[str],
) -> tuple[dict[str, dict[str, int]] | None, set[int], dict[str, int]]:
    """Load month-keyed bucket priority overrides from risk-score config."""

    if raw_priority_overrides is None:
        return None, set(), {}
    if not isinstance(raw_priority_overrides, dict):
        raise ValueError("risk_score_priority_overrides must be a JSON object")

    raw_scores = raw_priority_overrides.get("scores")
    if not isinstance(raw_scores, list) or not raw_scores:
        raise ValueError("risk_score_priority_overrides.scores must be a list")
    target_scores: set[int] = set()
    for raw_score in raw_scores:
        try:
            target_scores.add(int(raw_score))
        except (TypeError, ValueError) as parse_error:
            raise ValueError(
                "risk_score_priority_overrides.scores must contain integers"
            ) from parse_error

    raw_priorities = raw_priority_overrides.get("priorities")
    if not isinstance(raw_priorities, dict) or not raw_priorities:
        raise ValueError(
            "risk_score_priority_overrides.priorities must be a JSON object"
        )
    priority_by_bucket_label: dict[str, int] = {}
    for raw_bucket_label, raw_priority in raw_priorities.items():
        bucket_label = str(raw_bucket_label)
        if bucket_label not in bucket_labels:
            raise ValueError(
                "risk_score_priority_overrides.priorities contains unknown "
                f"bucket label: {bucket_label}"
            )
        try:
            priority_by_bucket_label[bucket_label] = int(raw_priority)
        except (TypeError, ValueError) as parse_error:
            raise ValueError(
                "risk_score_priority_overrides.priorities values must be integers"
            ) from parse_error

    risk_score_csv_path_text = raw_priority_overrides.get("csv_path")
    if not risk_score_csv_path_text and isinstance(raw_risk_score_gate, dict):
        risk_score_csv_path_text = raw_risk_score_gate.get("csv_path")
    if not risk_score_csv_path_text:
        raise ValueError(
            "risk_score_priority_overrides.csv_path is required when "
            "risk_score_gate.csv_path is not configured"
        )

    risk_score_csv_path = _resolve_repository_relative_path(
        risk_score_csv_path_text
    )
    if not risk_score_csv_path.exists():
        raise ValueError(
            "risk_score_priority_overrides.csv_path not found: "
            f"{risk_score_csv_path}"
        )

    bucket_priority_overrides_by_month: dict[str, dict[str, int]] = {}
    with risk_score_csv_path.open("r", newline="") as risk_score_file:
        reader = csv.DictReader(risk_score_file)
        for row in reader:
            try:
                risk_score = int(row["risk_score"])
                year_month_text = row["year_month"]
            except (KeyError, TypeError, ValueError):
                continue
            if risk_score in target_scores:
                bucket_priority_overrides_by_month[year_month_text] = dict(
                    priority_by_bucket_label
                )

    return (
        bucket_priority_overrides_by_month,
        target_scores,
        priority_by_bucket_label,
    )


def apply_risk_score_priority_override_for_month(
    config: multi_bucket_today.MultiBucketRunConfig,
    evaluation_month: str,
) -> tuple[set[int], dict[str, int]] | None:
    """Apply configured bucket priorities for one risk-score month.

    The daily cron path evaluates one date at a time, so it only needs the
    override for ``evaluation_month`` instead of the simulator's full
    month-keyed mapping.
    """

    (
        bucket_priority_overrides_by_month,
        target_scores,
        priority_by_bucket_label,
    ) = load_risk_score_priority_overrides(
        config.raw_document.get("risk_score_priority_overrides"),
        config.raw_document.get("risk_score_gate"),
        set(config.bucket_definitions),
    )
    if bucket_priority_overrides_by_month is None:
        return None

    priority_override_for_month = bucket_priority_overrides_by_month.get(
        evaluation_month
    )
    if priority_override_for_month is None:
        return target_scores, {}

    for bucket_label, priority_value in priority_override_for_month.items():
        config.bucket_definitions[bucket_label].entry_priority = priority_value

    return target_scores, priority_by_bucket_label


def _resolve_strategy_choice(raw_name: str, allowed: dict) -> str:
    """Return the first supported strategy token from ``raw_name``.

    Configuration values may contain simple logical expressions such as
    ``"ema_a | ema_b"`` or ``"ema_a or ema_b"``. The function splits the
    expression on the recognized separators and returns the first token whose
    base name exists in the ``allowed`` dictionary. If none match, the original
    ``raw_name`` is returned unchanged.
    """
    parts = re.split(r"\s*(?:\bor\b|\||/)\s*", raw_name.strip())
    for token in parts:
        if not token:
            continue
        try:
            base_name, _, _, _, _ = strategy.parse_strategy_name(token)
        except Exception:  # noqa: BLE001
            continue
        if base_name in allowed:
            return token
    return raw_name


def _has_supported_strategy(expression: str, allowed: dict) -> bool:
    """Return ``True`` when ``expression`` references a supported strategy.

    The function first attempts to parse ``expression`` as a single strategy
    name. When that succeeds and the resulting base name is found in
    ``allowed``, the strategy is considered supported. Only if parsing the
    entire expression fails do we split on the recognized separators (``or``,
    ``|``, ``/``) and check each token individually.
    """
    try:
        base_name, _, _, _, _ = strategy.parse_strategy_name(expression)
    except Exception:  # noqa: BLE001
        pass
    else:
        if base_name in allowed:
            return True
        for allowed_name in allowed:
            if expression.startswith(f"{allowed_name}_"):
                return True

    parts = re.split(r"\s*(?:\bor\b|\||/)\s*", expression.strip())
    for token in parts:
        if not token:
            continue
        try:
            base_name, _, _, _, _ = strategy.parse_strategy_name(token)
        except Exception:  # noqa: BLE001
            continue
        if base_name in allowed:
            return True
    return False


def _parse_volume_filter(
    volume_filter: str,
) -> tuple[float | None, float | None, int | None, int]:
    """Parse a dollar-volume filter expression."""

    maximum_symbols_per_group = 1
    pick_match = re.fullmatch(
        r"(.*),Pick(\d+)", volume_filter, flags=re.IGNORECASE
    )
    if pick_match is not None:
        volume_filter = pick_match.group(1)
        maximum_symbols_per_group = int(pick_match.group(2))

    combined_percentage_top_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,Top(\d+)",
        volume_filter,
        flags=re.IGNORECASE,
    )
    combined_percentage_nth_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%,(\d+)th",
        volume_filter,
    )
    if (
        combined_percentage_top_match is not None
        or combined_percentage_nth_match is not None
    ):
        match_obj = combined_percentage_top_match or combined_percentage_nth_match
        minimum_average_dollar_volume_ratio = float(match_obj.group(1)) / 100
        top_dollar_volume_rank = int(match_obj.group(2))
        return (
            None,
            minimum_average_dollar_volume_ratio,
            top_dollar_volume_rank,
            maximum_symbols_per_group,
        )

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
        return (
            minimum_average_dollar_volume,
            None,
            top_dollar_volume_rank,
            maximum_symbols_per_group,
        )

    percentage_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d{1,2})?)%",
        volume_filter,
    )
    if percentage_match is not None:
        minimum_average_dollar_volume_ratio = float(
            percentage_match.group(1)
        ) / 100
        return (
            None,
            minimum_average_dollar_volume_ratio,
            None,
            maximum_symbols_per_group,
        )

    volume_match = re.fullmatch(
        r"dollar_volume>(\d+(?:\.\d+)?)",
        volume_filter,
    )
    if volume_match is not None:
        minimum_average_dollar_volume = float(volume_match.group(1))
        return (
            minimum_average_dollar_volume,
            None,
            None,
            maximum_symbols_per_group,
        )

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
        return (None, None, top_dollar_volume_rank, maximum_symbols_per_group)

    raise ValueError(
        "unsupported filter; expected dollar_volume>NUMBER, "
        "dollar_volume>NUMBER%, dollar_volume=TopN (or Nth), "
        "dollar_volume>NUMBER,TopN (or ,Nth), or "
        "dollar_volume>NUMBER%,TopN (or ,Nth)"
    )


def _parse_stop_take_show(tokens: list[str]) -> tuple[float, float, bool]:
    """Parse optional stop-loss, take-profit, and detail flag tokens."""

    stop_loss_percentage = 1.0
    take_profit_percentage = 0.0
    show_trade_details = True
    remaining_tokens = list(tokens)

    if not remaining_tokens:
        return stop_loss_percentage, take_profit_percentage, show_trade_details

    first_token = remaining_tokens[0].lower()
    if first_token in {"true", "false"}:
        show_trade_details = first_token == "true"
        if len(remaining_tokens) > 1:
            raise ValueError("too many arguments")
        return stop_loss_percentage, take_profit_percentage, show_trade_details

    try:
        stop_loss_percentage = float(remaining_tokens[0])
    except ValueError as error:
        raise ValueError("invalid stop loss or take profit") from error
    remaining_tokens = remaining_tokens[1:]

    if not remaining_tokens:
        return stop_loss_percentage, take_profit_percentage, show_trade_details

    next_token = remaining_tokens[0].lower()
    if next_token in {"true", "false"}:
        show_trade_details = next_token == "true"
        if len(remaining_tokens) > 1:
            raise ValueError("too many arguments")
        return stop_loss_percentage, take_profit_percentage, show_trade_details

    try:
        take_profit_percentage = float(remaining_tokens[0])
    except ValueError as error:
        raise ValueError("invalid stop loss or take profit") from error
    remaining_tokens = remaining_tokens[1:]

    if not remaining_tokens:
        return stop_loss_percentage, take_profit_percentage, show_trade_details

    final_token = remaining_tokens[0].lower()
    if final_token in {"true", "false"} and len(remaining_tokens) == 1:
        show_trade_details = final_token == "true"
        return stop_loss_percentage, take_profit_percentage, show_trade_details
    raise ValueError("too many arguments")


def save_trade_details_to_log(
    evaluation_metrics: strategy.StrategyMetrics,
    log_path: Path,
) -> None:
    """Write trade details to a log file.

    Parameters
    ----------
    evaluation_metrics:
        Aggregated metrics containing trade details for the simulation.
    log_path:
        Directory where the log file should be stored.
    """
    # TODO: review
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = log_path / f"trade_details_{timestamp_string}.log"
    with output_file.open("w", encoding="utf-8") as file_handle:
        trade_details_by_year = evaluation_metrics.trade_details_by_year or {}
        for year in sorted(trade_details_by_year.keys()):
            for trade_detail in trade_details_by_year.get(year, []):
                if trade_detail.action == "close" and trade_detail.result is not None:
                    if trade_detail.percentage_change is not None:
                        result_suffix = (
                            f" {trade_detail.result} "
                            f"{trade_detail.percentage_change:.2%} "
                            f"{trade_detail.exit_reason}"
                        )
                    else:
                        result_suffix = (
                            f" {trade_detail.result} "
                            f"{trade_detail.exit_reason}"
                        )
                else:
                    result_suffix = ""
                open_metrics = ""
                if trade_detail.action == "open":
                    price_score_text = (
                        f"{trade_detail.price_concentration_score:.2f}"
                        if trade_detail.price_concentration_score is not None
                        else "N/A"
                    )
                    near_ratio_text = (
                        f"{trade_detail.near_price_volume_ratio:.2f}"
                        if trade_detail.near_price_volume_ratio is not None
                        else "N/A"
                    )
                    above_ratio_text = (
                        f"{trade_detail.above_price_volume_ratio:.2f}"
                        if trade_detail.above_price_volume_ratio is not None
                        else "N/A"
                    )
                    node_count_text = (
                        f"{trade_detail.histogram_node_count}"
                        if trade_detail.histogram_node_count is not None
                        else "N/A"
                    )
                    sma_angle_text = (
                        f"{trade_detail.sma_angle:.2f}"
                        if trade_detail.sma_angle is not None
                        else "N/A"
                    )
                    d_sma_angle_text = (
                        f"{trade_detail.d_sma_angle:.2f}"
                        if trade_detail.d_sma_angle is not None
                        else "N/A"
                    )
                    ema_angle_text = (
                        f"{trade_detail.ema_angle:.2f}"
                        if trade_detail.ema_angle is not None
                        else "N/A"
                    )
                    d_ema_angle_text = (
                        f"{trade_detail.d_ema_angle:.2f}"
                        if trade_detail.d_ema_angle is not None
                        else "N/A"
                    )
                    signal_bar_open_text = (
                        f"{trade_detail.signal_bar_open:.2f}"
                        if trade_detail.signal_bar_open is not None
                        else "N/A"
                    )
                    open_metrics = (
                        f" signal_open={signal_bar_open_text}"
                        f" price_score={price_score_text}"
                        f" near_pct={near_ratio_text}"
                        f" above_pct={above_ratio_text}"
                        f" node_count={node_count_text}"
                        f" sma_angle={sma_angle_text}"
                        f" d_sma_angle={d_sma_angle_text}"
                        f" ema_angle={ema_angle_text}"
                        f" d_ema_angle={d_ema_angle_text}"
                    )
                position_count = (
                    trade_detail.global_concurrent_position_count
                    if trade_detail.global_concurrent_position_count is not None
                    else trade_detail.concurrent_position_count
                )
                line = (
                    f"  {trade_detail.date.date()} "
                    f"({position_count}) "
                    f"{trade_detail.symbol} {trade_detail.action} {trade_detail.price:.2f} "
                    f"{trade_detail.group_simple_moving_average_dollar_volume_ratio:.4f} "
                    f"{trade_detail.simple_moving_average_dollar_volume / 1_000_000:.2f}M "
                    f"{trade_detail.group_total_simple_moving_average_dollar_volume / 1_000_000:.2f}M"
                    f"{open_metrics}{result_suffix}"
                )
                file_handle.write(line + "\n")


def _cleanup_yfinance_session() -> None:
    """Close shared yfinance session and run garbage collection."""
    session = getattr(yfinance.shared, "_SESSION", None)  # TODO: review
    if session is not None:
        try:
            session.close()  # TODO: review
        except Exception as close_error:  # noqa: BLE001
            LOGGER.debug(
                "Failed to close yfinance session: %s", close_error
            )  # TODO: review
    gc.collect()  # TODO: review


class StockShell(cmd.Cmd):
    """Interactive command shell for stock data maintenance."""

    intro = "Stock Indicator shell. Type help or ? to list commands."
    prompt = "(stock-indicator) "

    def do_update_symbols(self, argument_line: str) -> None:  # noqa: D401
        """update_symbols
        Build the latest common-stock symbol cache from SEC tickers."""
        symbols.update_symbol_cache()
        self.stdout.write("Common-stock symbol cache updated\n")

    # TODO: review
    def help_update_symbols(self) -> None:
        """Display help for the update_symbols command."""
        self.stdout.write(
            "update_symbols\n"
            "Download SEC company tickers and cache common-stock candidates.\n"
            "This command has no parameters.\n"
        )

    def do_update_data_from_yf(self, argument_line: str) -> None:  # noqa: D401
        """update_data_from_yf SYMBOL START END
        Download data from Yahoo Finance for SYMBOL between START and END and store as CSV.

        The END argument is inclusive. One day is added internally to match
        the exclusive end-date semantics of the data source."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 3:
            self.stdout.write("usage: update_data_from_yf SYMBOL START END\n")
            return
        symbol_name, start_date, end_date = argument_parts
        exclusive_end_date = (
            datetime.date.fromisoformat(end_date) + datetime.timedelta(days=1)
        ).isoformat()
        data_frame: DataFrame = data_loader.download_history(
            symbol_name, start_date, exclusive_end_date
        )
        _cleanup_yfinance_session()  # TODO: review
        data_frame_with_date: DataFrame = (
            data_frame.reset_index().rename(columns={"index": "Date"})
        )
        STOCK_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
        with output_path.open("w", encoding="utf-8") as fh:
            data_frame_with_date.to_csv(fh, index=False)
        self.stdout.write(f"Data written to {output_path}\n")
        # If sector data lacks this symbol, classify it as 'Other' (FF12=12)
        try:
            assign_symbol_to_other_if_missing(symbol_name)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning(
                "Could not assign default sector for %s: %s", symbol_name, error
            )
        # Also ensure S&P 500 index data is maintained separately when updating a single symbol
        if symbol_name != SP500_SYMBOL:
            sp_frame: DataFrame = data_loader.download_history(
                SP500_SYMBOL, start_date, exclusive_end_date
            )
            _cleanup_yfinance_session()  # TODO: review
            sp_with_date: DataFrame = (
                sp_frame.reset_index().rename(columns={"index": "Date"})
            )
            sp_output = STOCK_DATA_DIRECTORY / f"{SP500_SYMBOL}.csv"
            with sp_output.open("w", encoding="utf-8") as fh:
                sp_with_date.to_csv(fh, index=False)
            self.stdout.write(f"Data written to {sp_output}\n")

    def help_update_data_from_yf(self) -> None:
        """Display help for the update_data_from_yf command."""
        self.stdout.write(
            "update_data_from_yf SYMBOL START END\n"
            "Download data from Yahoo Finance for SYMBOL and write CSV to data/stock_data/<SYMBOL>.csv.\n"
            "Parameters:\n"
            "  SYMBOL: Ticker symbol for the asset.\n"
            "  START: Start date in YYYY-MM-DD format.\n"
            "  END: End date in YYYY-MM-DD format (inclusive).\n"
        )

    

    def do_update_all_data_from_yf(self, argument_line: str) -> None:  # noqa: D401
        """update_all_data_from_yf START END
        Download data from Yahoo Finance for all cached symbols.

        The END argument is inclusive. One day is added internally to match the
        exclusive end-date semantics of the data source."""
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) != 2:
            self.stdout.write("usage: update_all_data_from_yf START END\n")
            return
        start_date, end_date = argument_parts
        daily_job.update_all_data_from_yf(
            start_date,
            end_date,
            STOCK_DATA_DIRECTORY,
        )
        self.stdout.write(
            "Data refresh completed for sector-safe runtime universe\n"
        )

    def help_update_all_data_from_yf(self) -> None:
        """Display help for the update_all_data_from_yf command."""
        self.stdout.write(
            "update_all_data_from_yf START END\n"
            "Download data from Yahoo Finance for all cached symbols and write CSVs to data/stock_data/.\n"
            "Parameters:\n"
            "  START: Start date in YYYY-MM-DD format.\n"
            "  END: End date in YYYY-MM-DD format (inclusive).\n"
        )

    def do_update_universe_pipeline(self, argument_line: str) -> None:  # noqa: D401
        """update_universe_pipeline [--dry-run] [--maximum-drop-ratio RATIO]
        Rebuild production-candidate symbols and FF12 sector data."""

        try:
            argument_parts = shlex.split(argument_line)
        except ValueError as error:
            self.stdout.write(f"invalid arguments: {error}\n")
            return
        dry_run = False
        maximum_symbol_drop_ratio: float | None = None
        argument_position = 0
        while argument_position < len(argument_parts):
            argument_text = argument_parts[argument_position]
            if argument_text == "--dry-run":
                dry_run = True
                argument_position += 1
                continue
            if argument_text == "--maximum-drop-ratio":
                next_argument_position = argument_position + 1
                if next_argument_position >= len(argument_parts):
                    self.stdout.write(
                        "usage: update_universe_pipeline [--dry-run] "
                        "[--maximum-drop-ratio RATIO]\n"
                    )
                    return
                maximum_drop_ratio_text = argument_parts[next_argument_position]
                argument_position += 2
            elif argument_text.startswith("--maximum-drop-ratio="):
                maximum_drop_ratio_text = argument_text.split("=", maxsplit=1)[1]
                argument_position += 1
            else:
                self.stdout.write(
                    "usage: update_universe_pipeline [--dry-run] "
                    "[--maximum-drop-ratio RATIO]\n"
                )
                return
            try:
                maximum_symbol_drop_ratio = float(maximum_drop_ratio_text)
            except ValueError:
                self.stdout.write("--maximum-drop-ratio must be a number\n")
                return
        try:
            pipeline_options: dict[str, Any] = {"publish_outputs": not dry_run}
            if maximum_symbol_drop_ratio is not None:
                pipeline_options["maximum_symbol_drop_ratio"] = (
                    maximum_symbol_drop_ratio
                )
            report = universe_pipeline.run_universe_pipeline(**pipeline_options)
        except Exception as error:  # noqa: BLE001
            self.stdout.write(f"Universe pipeline failed: {error}\n")
            raise
        self.stdout.write("\n".join(report.to_lines()) + "\n")

    def help_update_universe_pipeline(self) -> None:
        """Display help for the update_universe_pipeline command."""

        self.stdout.write(
            "update_universe_pipeline [--dry-run] [--maximum-drop-ratio RATIO]\n"
            "Refresh the SEC-derived tradable universe, apply LLM/policy/"
            "quarantine layers, "
            "and atomically rebuild production_candidate_symbols plus "
            "production_candidate_symbols_with_sector.\n"
            "Parameters:\n"
            "  --dry-run: validate and print symbol diff without publishing outputs.\n"
            "  --maximum-drop-ratio RATIO: one-run safety threshold for controlled "
            "large universe migrations.\n"
        )

    def do_sync_production_ff12_sector(self, argument_line: str) -> None:  # noqa: D401
        """sync_production_ff12_sector [--dry-run]
        Append promoted candidate FF12 rows into production sector outputs."""

        try:
            argument_parts = shlex.split(argument_line)
        except ValueError as parse_error:
            self.stdout.write(f"invalid arguments: {parse_error}\n")
            return

        dry_run = False
        for argument_text in argument_parts:
            if argument_text == "--dry-run":
                dry_run = True
                continue
            self.stdout.write("usage: sync_production_ff12_sector [--dry-run]\n")
            return

        try:
            report = production_ff12_promotion.sync_production_ff12_sector(
                publish_outputs=not dry_run,
            )
        except (FileNotFoundError, OSError, ValueError) as sync_error:
            self.stdout.write(
                f"Production FF12 promotion sync failed: {sync_error}\n"
            )
            raise
        self.stdout.write("\n".join(report.to_lines()) + "\n")

    def help_sync_production_ff12_sector(self) -> None:
        """Display help for the sync_production_ff12_sector command."""

        self.stdout.write(
            "sync_production_ff12_sector [--dry-run]\n"
            "Compare production_symbols.txt, production_symbols_with_sector, "
            "and production_candidate_symbols_with_sector. Existing production "
            "rows stay frozen; missing promoted symbols are appended from "
            "candidate sector rows, then production parquet and CSV outputs are "
            "written atomically with symbols.txt runtime mirrors.\n"
            "Parameters:\n"
            "  --dry-run: validate and print the promotion diff without publishing.\n"
        )

    def do_update_sector_data(self, argument_line: str) -> None:  # noqa: D401
        """update_sector_data [--ff-map-url=URL OUTPUT_PATH]
        Refresh the local sector classification data set."""
        argument_parts: List[str] = argument_line.split()
        if not argument_parts:
            LOGGER.info(
                "Updating sector classification data using last run configuration",
            )
            try:
                data_frame: DataFrame = pipeline.update_latest_dataset()
            except (FileNotFoundError, ValueError, OSError) as error:
                self.stdout.write(
                    f"Error: {error}\n"
                    "usage: update_sector_data --ff-map-url=URL OUTPUT_PATH\n"
                )
                return
            coverage_report = pipeline.generate_coverage_report(data_frame)
            self.stdout.write(f"{coverage_report}\n")
            return
        mapping_url: str | None = None
        output_path_string: str | None = None
        for token in argument_parts:
            if token.startswith("--ff-map-url="):
                mapping_url = token.split("=", 1)[1]
            else:
                output_path_string = token
        if mapping_url is None or output_path_string is None:
            self.stdout.write(
                "usage: update_sector_data --ff-map-url=URL OUTPUT_PATH\n",
            )
            return
        output_path = Path(output_path_string)
        LOGGER.info(
            "Building sector classification data using %s",
            mapping_url,
        )
        data_frame = pipeline.build_sector_classification_dataset(
            mapping_url,
            output_path,
        )
        coverage_report = pipeline.generate_coverage_report(data_frame)
        self.stdout.write(f"{coverage_report}\n")

    def help_update_sector_data(self) -> None:
        """Display help for the update_sector_data command."""
        self.stdout.write(
            "update_sector_data --ff-map-url=URL OUTPUT_PATH\n"
            "Refresh sector classification data from SEC and Fama-French sources.\n"
            "The ticker universe is sourced from the curated symbols.txt cache.\n"
            "Without parameters, rebuilds data using the last saved configuration.\n"
            "Parameters:\n"
            "  --ff-map-url: URL or file path to SIC to Fama-French mapping.\n"
            "  OUTPUT_PATH: Destination path for the Parquet output file.\n"
        )

    def do_filter_debug_values(self, argument_line: str) -> None:  # noqa: D401
        """filter_debug_values SYMBOL DATE (BUY SELL | strategy=ID)

        Display indicator debug metrics for a symbol on the given date."""
        usage_message = (
            "usage: filter_debug_values SYMBOL DATE (BUY SELL | strategy=ID)\n"
        )
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) < 3:
            self.stdout.write(usage_message)
            return
        symbol_name = argument_parts.pop(0)
        date_string = argument_parts.pop(0)
        strategy_identifier: str | None = None
        non_strategy_parts: list[str] = []
        for part in argument_parts:
            if part.startswith("strategy="):
                strategy_identifier = part.split("=", 1)[1].strip()
            elif part:
                non_strategy_parts.append(part)

        if strategy_identifier:
            mapping = load_strategy_set_mapping()
            if strategy_identifier not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_identifier}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_identifier]
        elif len(non_strategy_parts) == 2:
            buy_strategy_name, sell_strategy_name = non_strategy_parts
        else:
            self.stdout.write(usage_message)
            return  # TODO: review
        result = daily_job.filter_debug_values(
            symbol_name, date_string, buy_strategy_name, sell_strategy_name
        )
        output_row = {"date": date_string, **result}
        output_frame = pandas.DataFrame([output_row])
        self.stdout.write(output_frame.to_string(index=False) + "\n")

    def help_filter_debug_values(self) -> None:
        """Display help for the filter_debug_values command."""
        # TODO: review
        self.stdout.write(
            "filter_debug_values SYMBOL DATE (BUY SELL | strategy=ID)\n"
            "Display indicator debug metrics for SYMBOL on DATE using either explicit "
            "BUY and SELL strategies or a strategy id from data/strategy_sets.csv.\n"
        )

    # TODO: review
    def do_complex_simulation(self, argument_line: str) -> None:  # noqa: D401
        """complex_simulation MAX_POSITION_COUNT [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [margin=NUMBER] SET_A -- SET_B [SHOW_DETAILS]
        Evaluate two strategy sets with shared capital limits."""

        usage_message = (
            "usage: complex_simulation MAX_POSITION_COUNT [starting_cash=NUMBER] "
            "[withdraw=NUMBER] [start=YYYY-MM-DD] [margin=NUMBER] SET_A -- SET_B "
            "[SHOW_DETAILS]\n"
        )

        argument_parts: List[str] = argument_line.split()
        if not argument_parts:
            self.stdout.write(usage_message)
            return

        maximum_position_token = argument_parts.pop(0)
        if maximum_position_token.startswith("maximum_position_count="):
            maximum_position_value = maximum_position_token.split("=", 1)[1]
        else:
            maximum_position_value = maximum_position_token
        try:
            maximum_position_count = int(maximum_position_value)
        except ValueError:
            self.stdout.write("invalid maximum position count\n")
            return
        if maximum_position_count <= 0:
            self.stdout.write("maximum position count must be positive\n")
            return

        show_trade_details = True
        if argument_parts and argument_parts[-1].lower() in {"true", "false"}:
            show_trade_details = argument_parts.pop(-1).lower() == "true"

        starting_cash_value = 3000.0
        withdraw_amount = 0.0
        start_date_string: str | None = None
        margin_multiplier = 1.0
        minimum_holding_bars = 0
        use_confirmation_angle = False
        confirmation_entry_mode = "limit"
        while (
            argument_parts
            and argument_parts[0] != "--"
            and (
                argument_parts[0].startswith(
                    ("starting_cash=", "withdraw=", "start=", "margin=", "min_hold=")
                )
                or argument_parts[0] in {"angle_confirmation_using_limit", "angle_confirmation_using_market"}
            )
        ):
            if argument_parts[0] == "angle_confirmation_using_limit":
                use_confirmation_angle = True
                argument_parts.pop(0)
                continue
            if argument_parts[0] == "angle_confirmation_using_market":
                use_confirmation_angle = True
                confirmation_entry_mode = "market"
                argument_parts.pop(0)
                continue
            parameter_part = argument_parts.pop(0)
            name, value = parameter_part.split("=", 1)
            if name == "start":
                try:
                    datetime.date.fromisoformat(value)
                except ValueError:
                    self.stdout.write("invalid start date\n")
                    return
                start_date_string = value
                continue
            if name == "margin":
                try:
                    margin_multiplier = float(value)
                except ValueError:
                    self.stdout.write("invalid margin multiplier\n")
                    return
                if margin_multiplier < 1.0:
                    self.stdout.write("margin must be >= 1.0\n")
                    return
                continue
            if name == "min_hold":
                try:
                    minimum_holding_bars = int(value)
                except ValueError:
                    self.stdout.write("invalid min_hold\n")
                    return
                if minimum_holding_bars < 0:
                    self.stdout.write("min_hold must be >= 0\n")
                    return
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                self.stdout.write(f"invalid {name}\n")
                return
            if name == "starting_cash":
                starting_cash_value = numeric_value
            elif name == "withdraw":
                withdraw_amount = numeric_value

        set_tokens: list[list[str]] = [[]]
        for token in argument_parts:
            if token == "--":
                if not set_tokens[-1]:
                    self.stdout.write(usage_message)
                    return
                set_tokens.append([])
                continue
            set_tokens[-1].append(token)

        if len(set_tokens) != 2 or any(not tokens for tokens in set_tokens):
            self.stdout.write(usage_message)
            return

        strategy_mapping = load_strategy_set_mapping()
        entry_filters_mapping = load_strategy_entry_filters()

        def build_set_definition(
            label: str, tokens: list[str]
        ) -> strategy.ComplexStrategySetDefinition:
            if not tokens:
                raise ValueError(f"strategy set {label} requires parameters")
            remaining_tokens = tokens.copy()
            volume_filter = remaining_tokens.pop(0)
            try:
                (
                    minimum_average_dollar_volume,
                    minimum_average_dollar_volume_ratio,
                    top_dollar_volume_rank,
                    maximum_symbols_per_group,
                ) = _parse_volume_filter(volume_filter)
            except ValueError as error:
                raise ValueError(str(error)) from error

            strategy_identifier: str | None = None
            for index, token in enumerate(list(remaining_tokens)):
                if token.startswith("strategy="):
                    if strategy_identifier is not None:
                        raise ValueError("only one strategy id may be provided")
                    strategy_identifier = token.split("=", 1)[1].strip()
                    remaining_tokens.pop(index)
                    break

            # Disallow stray strategy= tokens after the first extraction
            if any(part.startswith("strategy=") for part in remaining_tokens):
                raise ValueError("only one strategy id may be provided")

            stop_loss_percentage = 1.0
            take_profit_percentage = 0.0
            if strategy_identifier:
                if strategy_identifier not in strategy_mapping:
                    raise ValueError(
                        f"unknown strategy id: {strategy_identifier}"
                    )
                buy_strategy_name, sell_strategy_name = strategy_mapping[
                    strategy_identifier
                ]
                if len(remaining_tokens) > 2:
                    raise ValueError("invalid stop loss or take profit")
                if remaining_tokens:
                    try:
                        stop_loss_percentage = float(remaining_tokens[0])
                    except ValueError as error:
                        raise ValueError("invalid stop loss or take profit") from error
                    if len(remaining_tokens) == 2:
                        try:
                            take_profit_percentage = float(remaining_tokens[1])
                        except ValueError as error:
                            raise ValueError(
                                "invalid stop loss or take profit"
                            ) from error
            else:
                if len(remaining_tokens) < 2:
                    raise ValueError(
                        f"strategy set {label} requires buy and sell strategies"
                    )
                buy_strategy_name = remaining_tokens.pop(0)
                sell_strategy_name = remaining_tokens.pop(0)
                if not _has_supported_strategy(
                    buy_strategy_name, strategy.BUY_STRATEGIES
                ) or not _has_supported_strategy(
                    sell_strategy_name, strategy.SELL_STRATEGIES
                ):
                    raise ValueError("unsupported strategies")
                if remaining_tokens:
                    if len(remaining_tokens) > 2:
                        raise ValueError("invalid stop loss or take profit")
                    try:
                        stop_loss_percentage = float(remaining_tokens[0])
                    except ValueError as error:
                        raise ValueError("invalid stop loss or take profit") from error
                    if len(remaining_tokens) == 2:
                        try:
                            take_profit_percentage = float(remaining_tokens[1])
                        except ValueError as error:
                            raise ValueError(
                                "invalid stop loss or take profit"
                            ) from error

            d_sma_range = None
            ema_range = None
            d_ema_range = None
            price_score_min_value = None
            price_score_max_value = None
            if strategy_identifier and strategy_identifier in entry_filters_mapping:
                ef = entry_filters_mapping[strategy_identifier]
                if ef.d_sma_min is not None or ef.d_sma_max is not None:
                    d_sma_range = (
                        ef.d_sma_min if ef.d_sma_min is not None else -99.0,
                        ef.d_sma_max if ef.d_sma_max is not None else 99.0,
                    )
                if ef.ema_min is not None or ef.ema_max is not None:
                    ema_range = (
                        ef.ema_min if ef.ema_min is not None else -99.0,
                        ef.ema_max if ef.ema_max is not None else 99.0,
                    )
                if ef.d_ema_min is not None or ef.d_ema_max is not None:
                    d_ema_range = (
                        ef.d_ema_min if ef.d_ema_min is not None else -99.0,
                        ef.d_ema_max if ef.d_ema_max is not None else 99.0,
                    )
                price_score_min_value = ef.price_score_min
                price_score_max_value = ef.price_score_max

            return strategy.ComplexStrategySetDefinition(
                label=label,
                buy_strategy_name=buy_strategy_name,
                sell_strategy_name=sell_strategy_name,
                strategy_identifier=strategy_identifier,
                stop_loss_percentage=stop_loss_percentage,
                take_profit_percentage=take_profit_percentage,
                minimum_average_dollar_volume=minimum_average_dollar_volume,
                minimum_average_dollar_volume_ratio=
                    minimum_average_dollar_volume_ratio,
                top_dollar_volume_rank=top_dollar_volume_rank,
                maximum_symbols_per_group=maximum_symbols_per_group,
                d_sma_range=d_sma_range,
                ema_range=ema_range,
                d_ema_range=d_ema_range,
                price_score_min=price_score_min_value,
                price_score_max=price_score_max_value,
            )

        try:
            set_a_definition = build_set_definition("A", set_tokens[0])
            set_b_definition = build_set_definition("B", set_tokens[1])
        except ValueError as error:
            self.stdout.write(f"{error}\n")
            return

        if start_date_string is None:
            start_date_string = determine_start_date(DATA_DIRECTORY)
        start_timestamp = pandas.Timestamp(start_date_string)

        data_directory = resolve_data_source(None)
        self.stdout.write(f"Data source: {data_directory.name}\n")

        try:
            simulation_metrics = strategy.run_complex_simulation(
                data_directory,
                {"A": set_a_definition, "B": set_b_definition},
                maximum_position_count=maximum_position_count,
                starting_cash=starting_cash_value,
                withdraw_amount=withdraw_amount,
                start_date=start_timestamp,
                margin_multiplier=margin_multiplier,
                margin_interest_annual_rate=0.048,
                use_confirmation_angle=use_confirmation_angle,
                confirmation_entry_mode=confirmation_entry_mode,
                minimum_holding_bars=minimum_holding_bars,
            )
        except ValueError as error:
            self.stdout.write(f"{error}\n")
            return

        self.stdout.write(
            f"Simulation start date: {start_date_string}\n"
        )

        def format_summary_line(
            label: str, metrics: strategy.StrategyMetrics
        ) -> str:
            return (
                f"[{label}] Trades: {metrics.total_trades}, "
                f"Win rate: {metrics.win_rate:.2%}, "
                f"Mean profit %: {metrics.mean_profit_percentage:.2%}, "
                f"Profit % Std Dev: {metrics.profit_percentage_standard_deviation:.2%}, "
                f"Mean loss %: {metrics.mean_loss_percentage:.2%}, "
                f"Loss % Std Dev: {metrics.loss_percentage_standard_deviation:.2%}, "
                f"Mean holding period: {metrics.mean_holding_period:.2f} bars, "
                f"Holding period Std Dev: {metrics.holding_period_standard_deviation:.2f} bars, "
                f"Max concurrent positions: {metrics.maximum_concurrent_positions}, "
                f"Final balance: {metrics.final_balance:.2f}, "
                f"CAGR: {metrics.compound_annual_growth_rate:.2%}, "
                f"Max drawdown: {metrics.maximum_drawdown:.2%}\n"
            )

        def format_trade_detail(detail: strategy.TradeDetail) -> str:
            if detail.action == "close" and detail.result is not None:
                if detail.percentage_change is not None:
                    result_suffix = (
                        f" {detail.result} "
                        f"{detail.percentage_change:.2%} "
                        f"{detail.exit_reason}"
                    )
                else:
                    result_suffix = (
                        f" {detail.result} "
                        f"{detail.exit_reason}"
                    )
            else:
                result_suffix = ""
            open_metrics = ""
            if detail.action == "open":
                price_score_text = (
                    f"{detail.price_concentration_score:.2f}"
                    if detail.price_concentration_score is not None
                    else "N/A"
                )
                near_ratio_text = (
                    f"{detail.near_price_volume_ratio:.2f}"
                    if detail.near_price_volume_ratio is not None
                    else "N/A"
                )
                above_ratio_text = (
                    f"{detail.above_price_volume_ratio:.2f}"
                    if detail.above_price_volume_ratio is not None
                    else "N/A"
                )
                node_count_text = (
                    f"{detail.histogram_node_count}"
                    if detail.histogram_node_count is not None
                    else "N/A"
                )
                sma_angle_text = (
                    f"{detail.sma_angle:.2f}"
                    if detail.sma_angle is not None
                    else "N/A"
                )
                d_sma_angle_text = (
                    f"{detail.d_sma_angle:.2f}"
                    if detail.d_sma_angle is not None
                    else "N/A"
                )
                ema_angle_text = (
                    f"{detail.ema_angle:.2f}"
                    if detail.ema_angle is not None
                    else "N/A"
                )
                d_ema_angle_text = (
                    f"{detail.d_ema_angle:.2f}"
                    if detail.d_ema_angle is not None
                    else "N/A"
                )
                signal_bar_open_text = (
                    f"{detail.signal_bar_open:.2f}"
                    if detail.signal_bar_open is not None
                    else "N/A"
                )
                open_metrics = (
                    f" signal_open={signal_bar_open_text}"
                    f" price_score={price_score_text}"
                    f" near_pct={near_ratio_text}"
                    f" above_pct={above_ratio_text}"
                    f" node_count={node_count_text}"
                    f" sma_angle={sma_angle_text}"
                    f" d_sma_angle={d_sma_angle_text}"
                    f" ema_angle={ema_angle_text}"
                    f" d_ema_angle={d_ema_angle_text}"
            )
            position_count = (
                detail.global_concurrent_position_count
                if detail.global_concurrent_position_count is not None
                else detail.concurrent_position_count
            )
            return (
                f"{detail.date.date()} ({position_count}) "
                f"{detail.symbol} {detail.action} {detail.price:.2f} "
                f"{detail.group_simple_moving_average_dollar_volume_ratio:.4f} "
                f"{detail.simple_moving_average_dollar_volume / 1_000_000:.2f}M "
                f"{detail.group_total_simple_moving_average_dollar_volume / 1_000_000:.2f}M"
                f"{open_metrics}{result_suffix}"
            )

        total_metrics = simulation_metrics.overall_metrics
        self.stdout.write(format_summary_line("Total", total_metrics))
        for year, annual_return in sorted(total_metrics.annual_returns.items()):
            total_trade_count = total_metrics.annual_trade_counts.get(year, 0)
            self.stdout.write(
                f"[Total] Year {year}: {annual_return:.2%}, trade: {total_trade_count}\n"
            )

        for set_label in ("A", "B"):
            metrics = simulation_metrics.metrics_by_set.get(set_label)
            if metrics is None:
                continue
            self.stdout.write(format_summary_line(set_label, metrics))
            for year, annual_return in sorted(metrics.annual_returns.items()):
                trade_count = metrics.annual_trade_counts.get(year, 0)
                self.stdout.write(
                    f"[{set_label}] Year {year}: {annual_return:.2%}, trade: {trade_count}\n"
                )
                if show_trade_details:
                    trade_details = metrics.trade_details_by_year.get(year, [])
                    for trade_detail in trade_details:
                        formatted_detail = format_trade_detail(trade_detail)
                        self.stdout.write(
                            f"[{set_label}]   {formatted_detail}\n"
                        )

        # Export trade details to CSV
        trade_records: List[Dict[str, object]] = []
        for set_label in ("A", "B"):
            metrics = simulation_metrics.metrics_by_set.get(set_label)
            if metrics is None:
                continue
            all_details: List[strategy.TradeDetail] = []
            for year in sorted((metrics.trade_details_by_year or {}).keys()):
                all_details.extend(metrics.trade_details_by_year.get(year, []))
            open_events: Dict[str, strategy.TradeDetail] = {}
            for detail in all_details:
                if detail.action == "open":
                    open_events[detail.symbol] = detail
                elif detail.action == "close":
                    entry_detail = open_events.pop(detail.symbol, None)
                    if entry_detail is None:
                        continue
                    # Commission % from actual portfolio simulation
                    if (
                        detail.total_commission is not None
                        and detail.share_count is not None
                        and detail.share_count > 0
                        and entry_detail.price > 0
                    ):
                        position_value = detail.share_count * entry_detail.price
                        commission_pct = detail.total_commission / position_value
                    else:
                        commission_pct = None
                    trade_records.append(
                        {
                            "set": set_label,
                            "year": detail.date.year,
                            "entry_date": entry_detail.date.date(),
                            "concurrent_position_index": (
                                entry_detail.global_concurrent_position_count
                                if entry_detail.global_concurrent_position_count is not None
                                else entry_detail.concurrent_position_count
                            ),
                            "symbol": entry_detail.symbol,
                            "price_concentration_score": entry_detail.price_concentration_score,
                            "near_price_volume_ratio": entry_detail.near_price_volume_ratio,
                            "above_price_volume_ratio": entry_detail.above_price_volume_ratio,
                            "below_price_volume_ratio": entry_detail.below_price_volume_ratio,
                            "near_delta": entry_detail.near_delta,
                            "price_tightness": entry_detail.price_tightness,
                            "histogram_node_count": entry_detail.histogram_node_count,
                            "sma_angle": entry_detail.sma_angle,
                            "d_sma_angle": entry_detail.d_sma_angle,
                            "ema_angle": entry_detail.ema_angle,
                            "d_ema_angle": entry_detail.d_ema_angle,
                            "slope_60": entry_detail.slope_60,
                            "fuel_drawdown": entry_detail.fuel_drawdown,
                            "phantom": entry_detail.phantom,
                            "signal_bar_open": entry_detail.signal_bar_open,
                            "entry_price": entry_detail.price,
                            "exit_date": detail.date.date(),
                            "exit_price": detail.price,
                            "result": detail.result,
                            "percentage_change": detail.percentage_change,
                            "max_favorable_excursion_pct": detail.max_favorable_excursion_pct,
                            "max_adverse_excursion_pct": detail.max_adverse_excursion_pct,
                            "max_favorable_excursion_date": (
                                detail.max_favorable_excursion_date.date()
                                if detail.max_favorable_excursion_date is not None
                                else None
                            ),
                            "max_adverse_excursion_date": (
                                detail.max_adverse_excursion_date.date()
                                if detail.max_adverse_excursion_date is not None
                                else None
                            ),
                            "commission_pct": commission_pct,
                            "exit_reason": detail.exit_reason,
                            "holding_bars": max(1, (detail.date - entry_detail.date).days * 5 // 7),
                            "profit_per_bar": (
                                detail.percentage_change / max(1, (detail.date - entry_detail.date).days * 5 // 7)
                                if detail.percentage_change is not None
                                else None
                            ),
                        }
                    )
        if trade_records:
            output_directory = Path("logs") / "complex_simulation_result"
            output_directory.mkdir(parents=True, exist_ok=True)
            timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_directory / f"complex_simulation_{timestamp_string}.csv"
            pandas.DataFrame(
                trade_records,
                columns=[
                    "set",
                    "year",
                    "entry_date",
                    "concurrent_position_index",
                    "symbol",
                    "price_concentration_score",
                    "near_price_volume_ratio",
                    "above_price_volume_ratio",
                    "below_price_volume_ratio",
                    "near_delta",
                    "price_tightness",
                    "histogram_node_count",
                    "sma_angle",
                    "d_sma_angle",
                    "ema_angle",
                    "d_ema_angle",
                    "slope_60",
                    "fuel_drawdown",
                    "phantom",
                    "signal_bar_open",
                    "entry_price",
                    "exit_date",
                    "exit_price",
                    "result",
                    "percentage_change",
                    "max_favorable_excursion_pct",
                    "max_adverse_excursion_pct",
                    "max_favorable_excursion_date",
                    "max_adverse_excursion_date",
                    "commission_pct",
                    "exit_reason",
                    "holding_bars",
                    "profit_per_bar",
                ],
            ).to_csv(output_file, index=False)
            self.stdout.write(f"Trade details saved to {output_file}\n")

    # TODO: review
    def help_complex_simulation(self) -> None:
        """Display help for the complex_simulation command."""

        self.stdout.write(
            "complex_simulation MAX_POSITION_COUNT [starting_cash=NUMBER] [withdraw=NUMBER] "
            "[start=YYYY-MM-DD] [margin=NUMBER] SET_A -- SET_B [SHOW_DETAILS]\n"
            "Evaluate two strategy sets using a shared cash balance.\n"
            "Parameters:\n"
            "  MAX_POSITION_COUNT: Maximum concurrent positions for set A. Set B receives half (rounded up, minimum one).\n"
            "  starting_cash: Optional initial cash balance. Defaults to 3000.\n"
            "  withdraw: Optional annual withdrawal amount. Defaults to 0.\n"
            "  start: Optional start date in YYYY-MM-DD format. Defaults to earliest cached data.\n"
            "  margin: Optional leverage multiplier (>= 1.0). When greater than 1, a 4.8% annual interest rate is applied.\n"
            "  SHOW_DETAILS: True (default) or False to control trade detail output.\n"
            "Each SET definition must provide a dollar-volume filter followed by either BUY/SELL strategy names or strategy=ID,\n"
            "optionally followed by stop-loss and take-profit values. Separate the two sets with --.\n"
        )

    def do_multi_bucket_simulation(self, argument_line: str) -> None:  # noqa: D401
        """multi_bucket_simulation CONFIG_PATH [--export-state-on-date YYYY-MM-DD --export-state-out PATH]
        Run a simulation over N parallel strategy buckets defined in a JSON file.

        --export-state-on-date / --export-state-out: cold-start helper for
        the production multi_bucket_today command. Snapshots the rolling
        winners/losers/pending state at the boundary of the given date and
        writes it to PATH (default: data/adaptive_state_export.json)."""

        try:
            tokens = shlex.split(argument_line.strip())
        except ValueError as parse_error:
            self.stdout.write(f"failed to parse arguments: {parse_error}\n")
            return
        if not tokens:
            self.stdout.write(
                "usage: multi_bucket_simulation CONFIG_PATH "
                "[--export-state-on-date YYYY-MM-DD --export-state-out PATH]\n"
                "See help multi_bucket_simulation for the JSON format.\n"
            )
            return

        config_path_text = tokens[0]
        export_state_on_date_str: str | None = None
        export_state_out_path_text: str | None = None
        index_position = 1
        while index_position < len(tokens):
            current_token = tokens[index_position]
            if current_token == "--export-state-on-date" and index_position + 1 < len(tokens):
                export_state_on_date_str = tokens[index_position + 1]
                index_position += 2
            elif current_token == "--export-state-out" and index_position + 1 < len(tokens):
                export_state_out_path_text = tokens[index_position + 1]
                index_position += 2
            else:
                self.stdout.write(f"unknown argument: {current_token}\n")
                return

        if export_state_on_date_str is not None:
            try:
                datetime.date.fromisoformat(export_state_on_date_str)
            except ValueError:
                self.stdout.write(
                    f"--export-state-on-date must be YYYY-MM-DD, got "
                    f"{export_state_on_date_str}\n"
                )
                return

        config_path = Path(config_path_text).expanduser()
        if not config_path.exists():
            self.stdout.write(f"config file not found: {config_path}\n")
            return
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config_document = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            self.stdout.write(f"failed to load config: {error}\n")
            return

        if not isinstance(config_document, dict):
            self.stdout.write("config root must be a JSON object\n")
            return
        raw_buckets = config_document.get("buckets")
        if not isinstance(raw_buckets, list) or not raw_buckets:
            self.stdout.write("config must contain a non-empty 'buckets' array\n")
            return

        try:
            maximum_position_count = int(
                config_document.get("max_position_count", 0)
            )
        except (TypeError, ValueError):
            self.stdout.write("max_position_count must be an integer\n")
            return
        if maximum_position_count <= 0:
            self.stdout.write("max_position_count must be positive\n")
            return

        starting_cash_value = float(config_document.get("starting_cash", 3000.0))
        withdraw_amount = float(config_document.get("withdraw", 0.0))
        margin_multiplier = float(config_document.get("margin", 1.0))
        if margin_multiplier < 1.0:
            self.stdout.write("margin must be >= 1.0\n")
            return
        minimum_holding_bars = int(config_document.get("min_hold", 0))
        if minimum_holding_bars < 0:
            self.stdout.write("min_hold must be >= 0\n")
            return
        show_trade_details = bool(config_document.get("show_trade_details", False))

        start_date_string = config_document.get("start_date")
        if start_date_string is not None:
            try:
                datetime.date.fromisoformat(start_date_string)
            except ValueError:
                self.stdout.write("invalid start_date; expected YYYY-MM-DD\n")
                return

        confirmation_mode = config_document.get("confirmation_mode")
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
            self.stdout.write(
                f"invalid confirmation_mode: {confirmation_mode} "
                "(expected 'market', 'limit', or null)\n"
            )
            return

        # Optional override of the B-layer T+1 sma_angle confirmation range.
        # Defaults to strategy.CONFIRMATION_SMA_ANGLE_RANGE when not provided
        # in the JSON config.
        confirmation_sma_angle_range: tuple[float, float] | None = None
        raw_confirmation_min = config_document.get("confirmation_sma_angle_min")
        raw_confirmation_max = config_document.get("confirmation_sma_angle_max")
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
            except (TypeError, ValueError):
                self.stdout.write(
                    "confirmation_sma_angle_min/max must be numbers\n"
                )
                return
            if resolved_min > resolved_max:
                self.stdout.write(
                    "confirmation_sma_angle_min must be <= max\n"
                )
                return
            confirmation_sma_angle_range = (resolved_min, resolved_max)

        strategy_mapping = load_strategy_set_mapping()
        entry_filters_mapping = load_strategy_entry_filters()

        bucket_definitions: Dict[str, strategy.ComplexStrategySetDefinition] = {}
        seen_labels: set[str] = set()
        for bucket_index, raw_bucket in enumerate(raw_buckets):
            if not isinstance(raw_bucket, dict):
                self.stdout.write(
                    f"bucket[{bucket_index}] must be a JSON object\n"
                )
                return
            label = str(raw_bucket.get("label") or f"bucket{bucket_index+1}")
            if label in seen_labels:
                self.stdout.write(f"duplicate bucket label: {label}\n")
                return
            seen_labels.add(label)
            strategy_identifier = raw_bucket.get("strategy_id")
            if not strategy_identifier:
                self.stdout.write(
                    f"bucket {label} requires 'strategy_id'\n"
                )
                return
            if strategy_identifier not in strategy_mapping:
                self.stdout.write(
                    f"bucket {label}: unknown strategy_id '{strategy_identifier}'\n"
                )
                return
            buy_strategy_name, sell_strategy_name = strategy_mapping[
                strategy_identifier
            ]
            volume_filter_text = raw_bucket.get("dollar_volume_filter")
            if not volume_filter_text:
                self.stdout.write(
                    f"bucket {label} requires 'dollar_volume_filter'\n"
                )
                return
            try:
                (
                    minimum_average_dollar_volume,
                    minimum_average_dollar_volume_ratio,
                    top_dollar_volume_rank,
                    maximum_symbols_per_group,
                ) = _parse_volume_filter(volume_filter_text)
            except ValueError as error:
                self.stdout.write(
                    f"bucket {label} volume filter: {error}\n"
                )
                return

            try:
                stop_loss_percentage = float(raw_bucket.get("stop_loss", 1.0))
                take_profit_percentage = float(
                    raw_bucket.get("take_profit", 0.0)
                )
            except (TypeError, ValueError):
                self.stdout.write(
                    f"bucket {label}: stop_loss/take_profit must be numbers\n"
                )
                return
            try:
                entry_priority = int(raw_bucket.get("priority", 0))
            except (TypeError, ValueError):
                self.stdout.write(
                    f"bucket {label}: priority must be an integer\n"
                )
                return
            raw_max_positions = raw_bucket.get("max_positions")
            if raw_max_positions is None:
                bucket_maximum_positions: int | None = None
            else:
                try:
                    bucket_maximum_positions = int(raw_max_positions)
                except (TypeError, ValueError):
                    self.stdout.write(
                        f"bucket {label}: max_positions must be an integer or null\n"
                    )
                    return
                if bucket_maximum_positions <= 0:
                    self.stdout.write(
                        f"bucket {label}: max_positions must be positive\n"
                    )
                    return
            try:
                skipped_fama_french_groups = multi_bucket_today.parse_skip_ff12_groups(
                    raw_bucket.get("skip_ff12_groups"),
                    bucket_label=label,
                )
            except ValueError as error:
                self.stdout.write(f"{error}\n")
                return

            d_sma_range = None
            ema_range = None
            d_ema_range = None
            price_score_min_value = None
            price_score_max_value = None
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
            else:
                shape_slope_min_value = None
                shape_dev_50_max_value = None
                shape_bsv_lookback_value = None

            raw_exit_alpha_factor = raw_bucket.get("exit_alpha_factor")
            exit_alpha_factor_value: float | None = None
            if raw_exit_alpha_factor is not None:
                try:
                    exit_alpha_factor_value = float(raw_exit_alpha_factor)
                except (TypeError, ValueError):
                    self.stdout.write(
                        f"bucket {label}: exit_alpha_factor must be a number\n"
                    )
                    return

            near_delta_range_value: tuple[float, float] | None = None
            raw_near_delta = raw_bucket.get("near_delta_range")
            if raw_near_delta is not None:
                try:
                    near_delta_range_value = (
                        float(raw_near_delta[0]),
                        float(raw_near_delta[1]),
                    )
                except (TypeError, ValueError, IndexError):
                    self.stdout.write(
                        f"bucket {label}: near_delta_range must be [min, max]\n"
                    )
                    return

            price_tightness_range_value: tuple[float, float] | None = None
            raw_price_tightness = raw_bucket.get("price_tightness_range")
            if raw_price_tightness is not None:
                try:
                    price_tightness_range_value = (
                        float(raw_price_tightness[0]),
                        float(raw_price_tightness[1]),
                    )
                except (TypeError, ValueError, IndexError):
                    self.stdout.write(
                        f"bucket {label}: price_tightness_range must be [min, max]\n"
                    )
                    return

            sma_150_angle_min_value: float | None = None
            raw_sma_150 = raw_bucket.get("sma_150_angle_min")
            if raw_sma_150 is not None:
                try:
                    sma_150_angle_min_value = float(raw_sma_150)
                except (TypeError, ValueError):
                    self.stdout.write(
                        f"bucket {label}: sma_150_angle_min must be a number\n"
                    )
                    return

            bucket_definitions[label] = strategy.ComplexStrategySetDefinition(
                label=label,
                buy_strategy_name=buy_strategy_name,
                sell_strategy_name=sell_strategy_name,
                strategy_identifier=strategy_identifier,
                stop_loss_percentage=stop_loss_percentage,
                take_profit_percentage=take_profit_percentage,
                minimum_average_dollar_volume=minimum_average_dollar_volume,
                minimum_average_dollar_volume_ratio=
                    minimum_average_dollar_volume_ratio,
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
                gate_enabled=bool(
                    raw_bucket.get("gate_enabled", True)
                ),
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
                fuel_priority_threshold=(
                    float(raw_bucket["fuel_priority_threshold"])
                    if "fuel_priority_threshold" in raw_bucket
                    and raw_bucket["fuel_priority_threshold"] is not None
                    else None
                ),
                pre_cross_signal_lookback=bool(
                    raw_bucket.get("pre_cross_signal_lookback", False)
                ),
                tp_slope_amplify=bool(
                    raw_bucket.get("tp_slope_amplify", False)
                ),
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

        if start_date_string is None:
            start_date_string = determine_start_date(DATA_DIRECTORY)
        start_timestamp = pandas.Timestamp(start_date_string)
        data_source_name = config_document.get("data_source")
        if data_source_name == "daily":
            self.stdout.write(
                "multi_bucket_simulation rejects data_source='daily'; "
                "use a non-daily backtest source such as '2010' so simulation "
                "cannot read or mutate the production daily cache.\n"
            )
            return
        try:
            data_directory = resolve_data_source(data_source_name)
        except ValueError as source_error:
            self.stdout.write(f"{source_error}\n")
            return
        if not data_directory.exists():
            self.stdout.write(
                f"data source directory not found: {data_directory}\n"
            )
            return
        self.stdout.write(f"Data source: {data_directory.name}\n")
        try:
            allowed_symbols = load_symbol_list(config_document.get("symbol_list"))
        except ValueError as symbol_list_error:
            self.stdout.write(f"{symbol_list_error}\n")
            return
        if allowed_symbols is not None:
            self.stdout.write(f"Symbol list: {len(allowed_symbols)} symbols\n")

        ff12_data_path_text = (
            config_document.get("ff12_data_path")
            or config_document.get("sector_data_path")
        )
        try:
            ff12_data_path = resolve_ff12_data_path(ff12_data_path_text)
        except ValueError as ff12_path_error:
            self.stdout.write(f"{ff12_path_error}\n")
            return
        if ff12_data_path is not None:
            self.stdout.write(f"FF12 data: {ff12_data_path}\n")

        try:
            seasoning_config = symbol_seasoning.parse_symbol_seasoning_config(
                config_document.get("symbol_seasoning")
            )
            seasoning_dates_result = load_symbol_seasoning_dates_for_config(
                seasoning_config,
                data_directory=data_directory,
                allowed_symbols=allowed_symbols,
            )
        except (FileNotFoundError, ValueError) as seasoning_error:
            self.stdout.write(f"{seasoning_error}\n")
            return
        symbol_first_eligible_trade_dates = None
        if seasoning_dates_result is not None:
            seasoning_source_path, symbol_first_eligible_trade_dates = (
                seasoning_dates_result
            )
            self.stdout.write(
                "Symbol seasoning: enabled "
                f"records={len(symbol_first_eligible_trade_dates)} "
                f"source={seasoning_config.eligibility_source} "
                f"path={seasoning_source_path}\n"
            )

        # Parse adaptive TP/SL configuration.
        adaptive_tp_sl_config: strategy.AdaptiveTPSLConfig | None = None
        raw_adaptive = config_document.get("adaptive_tp_sl")
        if raw_adaptive is not None and raw_adaptive:
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
                # Boolean true -> use defaults.
                adaptive_tp_sl_config = strategy.AdaptiveTPSLConfig()
            sl_sigma_description = (
                adaptive_tp_sl_config.sl_sigma_multiplier
                if adaptive_tp_sl_config.sl_sigma_multiplier is not None
                else adaptive_tp_sl_config.sigma_multiplier
            )
            sl_desc = (
                f"fixed_sl_cap={adaptive_tp_sl_config.fixed_sl}"
                if adaptive_tp_sl_config.fixed_sl is not None
                else f"rolling_loss_sl_sigma={sl_sigma_description}"
            )
            self.stdout.write(
                f"Adaptive TP/SL: window={adaptive_tp_sl_config.window} "
                f"sigma={adaptive_tp_sl_config.sigma_multiplier} "
                f"{sl_desc}\n"
            )

        # WR-synced sizing: entry margin tracks the portfolio's rolling win
        # rate continuously (bounded Kelly). Mutually exclusive with the
        # risk-score reduce margin overrides (enforced inside
        # run_complex_simulation).
        wr_synced_sizing_config: strategy.WRSyncedSizingConfig | None = None
        raw_wr_synced_sizing = config_document.get("wr_synced_sizing")
        if raw_wr_synced_sizing is not None:
            wr_synced_sizing_config = strategy.WRSyncedSizingConfig(
                window=int(raw_wr_synced_sizing.get("window", 40)),
                wr_floor=float(raw_wr_synced_sizing.get("wr_floor", 0.45)),
                wr_healthy=float(raw_wr_synced_sizing.get("wr_healthy", 0.60)),
                curve=str(raw_wr_synced_sizing.get("curve", "linear")),
                z_floor=float(raw_wr_synced_sizing.get("z_floor", -3.0)),
                z_healthy=float(raw_wr_synced_sizing.get("z_healthy", -1.5)),
                sigma_ref_window=int(
                    raw_wr_synced_sizing.get("sigma_ref_window", 252)
                ),
            )
            if wr_synced_sizing_config.curve in ("z_score", "expectancy_z", "dual_z"):
                curve_description = (
                    f"curve={wr_synced_sizing_config.curve} "
                    f"z_floor={wr_synced_sizing_config.z_floor} "
                    f"z_healthy={wr_synced_sizing_config.z_healthy} "
                    f"sigma_ref_window={wr_synced_sizing_config.sigma_ref_window}"
                )
            else:
                curve_description = (
                    f"curve=linear wr_floor={wr_synced_sizing_config.wr_floor} "
                    f"wr_healthy={wr_synced_sizing_config.wr_healthy}"
                )
            self.stdout.write(
                "WR-synced sizing: "
                f"window={wr_synced_sizing_config.window} "
                f"{curve_description}\n"
            )

        # Phantom score gate (action 2): ft-family regime score decides
        # whether gated-bucket entries deploy capital (slot still taken).
        wr_gate_config: strategy.WRGateConfig | None = None
        raw_wr_gate = config_document.get("ft_family_wr_gate")
        if raw_wr_gate is not None:
            wr_gate_config = strategy.WRGateConfig(
                sensor_bucket=str(
                    raw_wr_gate.get(
                        "sensor_bucket", "fish_tail_production"
                    )
                ),
                gated_buckets=tuple(
                    raw_wr_gate.get(
                        "gated_buckets",
                        ["fish_tail_production", "fish_tail_squeeze"],
                    )
                ),
                window=int(raw_wr_gate.get("window", 12)),
                score_threshold=float(
                    raw_wr_gate.get("score_threshold", 0.5)
                ),
                weight_wr=float(raw_wr_gate.get("weight_wr", 0.5)),
                weight_no_tp=float(raw_wr_gate.get("weight_no_tp", 0.5)),
                weight_max_hold=float(
                    raw_wr_gate.get("weight_max_hold", 0.0)
                ),
                curve=str(raw_wr_gate.get("curve", "score")),
            )
            self.stdout.write(
                "Phantom score gate: "
                f"sensor={wr_gate_config.sensor_bucket} "
                f"gated={list(wr_gate_config.gated_buckets)} "
                f"window={wr_gate_config.window} "
                f"threshold={wr_gate_config.score_threshold} "
                f"curve={wr_gate_config.curve} "
                f"weights=(wr={wr_gate_config.weight_wr}, "
                f"no_tp={wr_gate_config.weight_no_tp}, "
                f"max_hold={wr_gate_config.weight_max_hold})\n"
            )

        export_state_at_date_ts: pandas.Timestamp | None = None
        exported_state_holder: Dict[str, Any] | None = None
        if export_state_on_date_str is not None:
            export_state_at_date_ts = pandas.Timestamp(export_state_on_date_str)
            exported_state_holder = {}

        # Risk-score gate for simulation/backtest configs. Stop months
        # remove new entries from gated buckets. Optional reduce months are
        # supported only when `reduce_threshold` is explicitly present for
        # legacy exploratory runs; production stop-only configs omit it. Live
        # order blocking belongs to dashboard.py, not cron/signal rolling.
        risk_score_stop_months: set[str] | None = None
        margin_overrides: dict[str, float] | None = None
        raw_gate = config_document.get("risk_score_gate")
        if raw_gate is not None:
            gate_csv_path_text = str(raw_gate.get("csv_path", ""))
            try:
                stop_threshold = int(raw_gate.get("stop_threshold", 75))
                raw_reduce_threshold = raw_gate.get("reduce_threshold")
                reduce_threshold = (
                    int(raw_reduce_threshold)
                    if raw_reduce_threshold is not None
                    else None
                )
                reduce_margin = float(raw_gate.get("reduce_margin", 1.0))
            except (TypeError, ValueError) as parse_error:
                self.stdout.write(
                    f"risk_score_gate thresholds must be numeric: {parse_error}\n"
                )
                return
            gate_csv_path = Path(gate_csv_path_text).expanduser()
            if not gate_csv_path.is_absolute():
                gate_csv_path = (
                    Path(__file__).resolve().parent.parent.parent
                    / gate_csv_path
                )
            if not gate_csv_path.exists():
                self.stdout.write(
                    f"risk_score_gate.csv_path not found: {gate_csv_path}\n"
                )
                return
            import csv as _csv
            risk_score_stop_months = set()
            margin_overrides = {}
            with gate_csv_path.open("r", newline="") as gate_file:
                reader = _csv.DictReader(gate_file)
                for row in reader:
                    try:
                        score = int(row["risk_score"])
                    except (KeyError, ValueError):
                        continue
                    if score >= stop_threshold:
                        risk_score_stop_months.add(row["year_month"])
                    elif (
                        reduce_threshold is not None
                        and score >= reduce_threshold
                    ):
                        margin_overrides[row["year_month"]] = reduce_margin
            reduce_description = (
                "disabled"
                if reduce_threshold is None
                else f"threshold={reduce_threshold} margin->{reduce_margin} "
                     f"({len(margin_overrides)} reduce months)"
            )
            self.stdout.write(
                f"Risk-score gate: stop_threshold={stop_threshold} "
                f"({len(risk_score_stop_months)} stop months), "
                f"reduce={reduce_description}\n"
            )

        try:
            (
                bucket_priority_overrides_by_month,
                target_priority_scores,
                priority_by_bucket_label,
            ) = load_risk_score_priority_overrides(
                config_document.get("risk_score_priority_overrides"),
                raw_gate,
                set(bucket_definitions),
            )
        except ValueError as priority_error:
            self.stdout.write(f"{priority_error}\n")
            return
        if bucket_priority_overrides_by_month is not None:
            sorted_scores_text = ", ".join(
                str(risk_score) for risk_score in sorted(target_priority_scores)
            )
            sorted_priorities_text = ", ".join(
                f"{bucket_label}->{priority_value}"
                for bucket_label, priority_value in sorted(
                    priority_by_bucket_label.items()
                )
            )
            self.stdout.write(
                "Risk-score priority override: "
                f"scores=[{sorted_scores_text}] "
                f"({len(bucket_priority_overrides_by_month)} months), "
                f"{sorted_priorities_text}\n"
            )

        try:
            with strategy.override_ff12_group_source_path(ff12_data_path):
                simulation_metrics = strategy.run_complex_simulation(
                    data_directory,
                    bucket_definitions,
                    maximum_position_count=maximum_position_count,
                    starting_cash=starting_cash_value,
                    withdraw_amount=withdraw_amount,
                    start_date=start_timestamp,
                    margin_multiplier=margin_multiplier,
                    margin_interest_annual_rate=0.048,
                    use_confirmation_angle=use_confirmation_angle,
                    confirmation_entry_mode=confirmation_entry_mode,
                    minimum_holding_bars=minimum_holding_bars,
                    multi_bucket_mode=True,
                    confirmation_sma_angle_range=confirmation_sma_angle_range,
                    adaptive_tp_sl=adaptive_tp_sl_config,
                    max_same_symbol=int(config_document.get("max_same_symbol", 1)),
                    allowed_symbols=allowed_symbols,
                    export_state_at_date=export_state_at_date_ts,
                    exported_state=exported_state_holder,
                    risk_score_stop_months=risk_score_stop_months,
                    margin_overrides=margin_overrides,
                    bucket_priority_overrides_by_month=(
                        bucket_priority_overrides_by_month
                    ),
                    symbol_first_eligible_trade_dates=(
                        symbol_first_eligible_trade_dates
                    ),
                    wr_synced_sizing=wr_synced_sizing_config,
                    wr_gate=wr_gate_config,
                )
        except ValueError as error:
            self.stdout.write(f"{error}\n")
            return

        if exported_state_holder is not None:
            exported_state_holder.pop("_captured", None)
            export_state_out_path = Path(
                export_state_out_path_text
                if export_state_out_path_text is not None
                else "data/adaptive_state_export.json"
            ).expanduser()
            try:
                with export_state_out_path.open("w", encoding="utf-8") as export_fp:
                    json.dump(exported_state_holder, export_fp, indent=2)
                self.stdout.write(
                    f"Exported rolling state at {export_state_on_date_str} "
                    f"to {export_state_out_path}\n"
                )
            except OSError as write_error:
                self.stdout.write(
                    f"failed to write exported state: {write_error}\n"
                )

        self.stdout.write(
            f"Simulation start date: {start_date_string}\n"
            f"Buckets: {', '.join(bucket_definitions.keys())}\n"
        )
        skipped_group_descriptions = [
            f"{bucket_label}={sorted(bucket_definition.skipped_fama_french_groups)}"
            for bucket_label, bucket_definition in bucket_definitions.items()
            if bucket_definition.skipped_fama_french_groups
        ]
        if skipped_group_descriptions:
            self.stdout.write(
                f"Skipped FF12 groups: {'; '.join(skipped_group_descriptions)}\n"
            )

        def format_summary_line(
            label: str, metrics: strategy.StrategyMetrics
        ) -> str:
            profit_loss_ratio = (
                metrics.mean_profit_percentage / metrics.mean_loss_percentage
                if metrics.mean_loss_percentage
                else 0.0
            )
            return (
                f"[{label}] Trades: {metrics.total_trades}, "
                f"Win rate: {metrics.win_rate:.2%}, "
                f"Mean profit %: {metrics.mean_profit_percentage:.2%}, "
                f"Profit % Std Dev: {metrics.profit_percentage_standard_deviation:.2%}, "
                f"Mean loss %: {metrics.mean_loss_percentage:.2%}, "
                f"Loss % Std Dev: {metrics.loss_percentage_standard_deviation:.2%}, "
                f"P/L: {profit_loss_ratio:.2f}, "
                f"Mean holding period: {metrics.mean_holding_period:.2f} bars, "
                f"Holding period Std Dev: {metrics.holding_period_standard_deviation:.2f} bars, "
                f"Max concurrent positions: {metrics.maximum_concurrent_positions}, "
                f"Final balance: {metrics.final_balance:.2f}, "
                f"CAGR: {metrics.compound_annual_growth_rate:.2%}, "
                f"Max drawdown: {metrics.maximum_drawdown:.2%}\n"
            )

        total_metrics = simulation_metrics.overall_metrics
        self.stdout.write(format_summary_line("Total", total_metrics))
        for year, annual_return in sorted(total_metrics.annual_returns.items()):
            total_trade_count = total_metrics.annual_trade_counts.get(year, 0)
            self.stdout.write(
                f"[Total] Year {year}: {annual_return:.2%}, trade: {total_trade_count}\n"
            )

        for label in bucket_definitions.keys():
            metrics = simulation_metrics.metrics_by_set.get(label)
            if metrics is None:
                continue
            self.stdout.write(format_summary_line(label, metrics))
            for year, annual_return in sorted(metrics.annual_returns.items()):
                trade_count = metrics.annual_trade_counts.get(year, 0)
                self.stdout.write(
                    f"[{label}] Year {year}: {annual_return:.2%}, trade: {trade_count}\n"
                )

        # Export trade details to CSV (generalised to N buckets)
        trade_records: List[Dict[str, object]] = []
        for label in bucket_definitions.keys():
            metrics = simulation_metrics.metrics_by_set.get(label)
            if metrics is None:
                continue
            all_details: List[strategy.TradeDetail] = []
            for year in sorted((metrics.trade_details_by_year or {}).keys()):
                all_details.extend(metrics.trade_details_by_year.get(year, []))
            open_events: Dict[str, strategy.TradeDetail] = {}
            for detail in all_details:
                if detail.action == "open":
                    open_events[detail.symbol] = detail
                elif detail.action == "close":
                    entry_detail = open_events.pop(detail.symbol, None)
                    if entry_detail is None:
                        continue
                    if (
                        detail.total_commission is not None
                        and detail.share_count is not None
                        and detail.share_count > 0
                        and entry_detail.price > 0
                    ):
                        position_value = detail.share_count * entry_detail.price
                        commission_pct = detail.total_commission / position_value
                    else:
                        commission_pct = None
                    trade_records.append(
                        {
                            "bucket": label,
                            "year": detail.date.year,
                            "entry_date": entry_detail.date.date(),
                            "concurrent_position_index": (
                                entry_detail.global_concurrent_position_count
                                if entry_detail.global_concurrent_position_count is not None
                                else entry_detail.concurrent_position_count
                            ),
                            "symbol": entry_detail.symbol,
                            "price_concentration_score": entry_detail.price_concentration_score,
                            "near_price_volume_ratio": entry_detail.near_price_volume_ratio,
                            "above_price_volume_ratio": entry_detail.above_price_volume_ratio,
                            "below_price_volume_ratio": entry_detail.below_price_volume_ratio,
                            "near_delta": entry_detail.near_delta,
                            "price_tightness": entry_detail.price_tightness,
                            "histogram_node_count": entry_detail.histogram_node_count,
                            "sma_angle": entry_detail.sma_angle,
                            "sma_angle_confirmation": entry_detail.sma_angle_confirmation,
                            "d_sma_angle": entry_detail.d_sma_angle,
                            "ema_angle": entry_detail.ema_angle,
                            "d_ema_angle": entry_detail.d_ema_angle,
                            "slope_60": entry_detail.slope_60,
                            "fuel_drawdown": entry_detail.fuel_drawdown,
                            "phantom": entry_detail.phantom,
                            "signal_bar_open": entry_detail.signal_bar_open,
                            "entry_price": entry_detail.price,
                            "exit_date": detail.date.date(),
                            "exit_price": detail.price,
                            "result": detail.result,
                            "percentage_change": detail.percentage_change,
                            "max_favorable_excursion_pct": detail.max_favorable_excursion_pct,
                            "max_adverse_excursion_pct": detail.max_adverse_excursion_pct,
                            "max_favorable_excursion_date": (
                                detail.max_favorable_excursion_date.date()
                                if detail.max_favorable_excursion_date is not None
                                else None
                            ),
                            "max_adverse_excursion_date": (
                                detail.max_adverse_excursion_date.date()
                                if detail.max_adverse_excursion_date is not None
                                else None
                            ),
                            "commission_pct": commission_pct,
                            "exit_reason": detail.exit_reason,
                            "holding_bars": max(1, (detail.date - entry_detail.date).days * 5 // 7),
                            "profit_per_bar": (
                                detail.percentage_change / max(1, (detail.date - entry_detail.date).days * 5 // 7)
                                if detail.percentage_change is not None
                                else None
                            ),
                            "adaptive_tp_pct": detail.adaptive_tp_pct,
                            "adaptive_sl_pct": detail.adaptive_sl_pct,
                        }
                    )
        if trade_records:
            output_directory = Path("logs") / "multi_bucket_simulation_result"
            output_directory.mkdir(parents=True, exist_ok=True)
            timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                output_directory / f"multi_bucket_simulation_{timestamp_string}.csv"
            )
            pandas.DataFrame(
                trade_records,
                columns=[
                    "bucket",
                    "year",
                    "entry_date",
                    "concurrent_position_index",
                    "symbol",
                    "price_concentration_score",
                    "near_price_volume_ratio",
                    "above_price_volume_ratio",
                    "below_price_volume_ratio",
                    "near_delta",
                    "price_tightness",
                    "histogram_node_count",
                    "sma_angle",
                    "sma_angle_confirmation",
                    "d_sma_angle",
                    "ema_angle",
                    "d_ema_angle",
                    "slope_60",
                    "fuel_drawdown",
                    "phantom",
                    "signal_bar_open",
                    "entry_price",
                    "exit_date",
                    "exit_price",
                    "result",
                    "percentage_change",
                    "max_favorable_excursion_pct",
                    "max_adverse_excursion_pct",
                    "max_favorable_excursion_date",
                    "max_adverse_excursion_date",
                    "commission_pct",
                    "exit_reason",
                    "holding_bars",
                    "profit_per_bar",
                    "adaptive_tp_pct",
                    "adaptive_sl_pct",
                ],
            ).to_csv(output_file, index=False)
            self.stdout.write(f"Trade details saved to {output_file}\n")

        if show_trade_details:
            self.stdout.write(
                "(trade-detail stdout dump not implemented for multi_bucket; "
                "see the CSV file above)\n"
            )

    def help_multi_bucket_simulation(self) -> None:
        """Display help for the multi_bucket_simulation command."""

        self.stdout.write(
            "multi_bucket_simulation CONFIG_PATH\n"
            "Run a portfolio simulation over N strategy buckets defined in a JSON file.\n"
            "Each bucket has its own SL/TP, dollar-volume filter, priority, and optional per-bucket cap.\n"
            "All buckets share a global max_position_count and compete for slots first-come-first-served,\n"
            "with bucket priority as the tiebreaker (lower number = higher priority).\n"
            "\n"
            "JSON format:\n"
            '{\n'
            '  "max_position_count": 4,\n'
            '  "starting_cash": 300000,\n'
            '  "start_date": "2014-01-01",\n'
            '  "withdraw": 0,\n'
            '  "min_hold": 5,\n'
            '  "margin": 1.0,\n'
            '  "confirmation_mode": "market",\n'
            '  "show_trade_details": false,\n'
            '  "buckets": [\n'
            '    {\n'
            '      "label": "B1_nearPV",\n'
            '      "strategy_id": "s51",\n'
            '      "dollar_volume_filter": "dollar_volume>0.05%,Top50,Pick2",\n'
            '      "stop_loss": 0.109,\n'
            '      "take_profit": 0.084,\n'
            '      "priority": 1,\n'
            '      "max_positions": null\n'
            '    },\n'
            '    ...\n'
            '  ]\n'
            '}\n'
            "\n"
            "Notes:\n"
            "  - data_source='daily' is rejected; simulations must use a "
            "non-daily backtest cache such as '2010'\n"
            "  - confirmation_mode: 'market', 'limit', or null (no confirmation)\n"
            "  - priority: lower = higher; ties broken by within-bucket quality then insertion order\n"
            "  - risk_score_priority_overrides can change bucket priority for selected monthly scores\n"
            "  - max_positions per bucket is optional; null means no per-bucket cap\n"
            "  - skip_ff12_groups per bucket is optional; values are positive FF group ids removed before ranking\n"
            "  - ff12_data_path is optional; use it for frozen old-universe sector maps\n"
            "  - strategy_id must exist in data/strategy_sets.csv\n"
            "  - output CSV: logs/multi_bucket_simulation_result/multi_bucket_simulation_*.csv\n"
        )

    # TODO: review
    def do_start_simulate(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate [max_positions=NUMBER] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]
        Evaluate trading strategies using cached data.

        STOP_LOSS defaults to 1.0 and TAKE_PROFIT defaults to 0.0 when not provided.
        SHOW_DETAILS defaults to True and controls whether trade details are printed."""
        argument_parts: List[str] = argument_line.split()
        starting_cash_value = 3000.0
        withdraw_amount = 0.0
        start_date_string: str | None = None
        allowed_group_identifiers: set[int] | None = None
        margin_multiplier: float = 1.0
        strategy_id: str | None = None
        maximum_position_count: int = 3
        minimum_holding_bars: int = 0
        use_confirmation_angle: bool = False
        confirmation_entry_mode: str = "limit"
        while argument_parts and (
            argument_parts[0].startswith("starting_cash=")
            or argument_parts[0].startswith("withdraw=")
            or argument_parts[0].startswith("start=")
            or argument_parts[0].startswith("group=")
            or argument_parts[0].startswith("margin=")
            or argument_parts[0].startswith("strategy=")
            or argument_parts[0].startswith("max_positions=")
            or argument_parts[0].startswith("min_hold=")
            or argument_parts[0] in {"angle_confirmation_using_limit", "angle_confirmation_using_market"}
        ):
            parameter_part = argument_parts.pop(0)
            if parameter_part == "angle_confirmation_using_limit":
                use_confirmation_angle = True
                continue
            if parameter_part == "angle_confirmation_using_market":
                use_confirmation_angle = True
                confirmation_entry_mode = "market"
                continue
            name, value = parameter_part.split("=", 1)
            if name == "min_hold":
                try:
                    minimum_holding_bars = int(value)
                except ValueError:
                    self.stdout.write("invalid min_hold\n")
                    return
                if minimum_holding_bars < 0:
                    self.stdout.write("min_hold must be >= 0\n")
                    return
                continue
            if name == "max_positions":
                try:
                    maximum_position_count = int(value)
                except ValueError:
                    self.stdout.write("invalid max_positions\n")
                    return
                if maximum_position_count < 1:
                    self.stdout.write("max_positions must be >= 1\n")
                    return
                continue
            if name == "start":
                try:
                    datetime.date.fromisoformat(value)
                except ValueError:
                    self.stdout.write("invalid start date\n")
                    return
                start_date_string = value
                continue
            if name == "group":
                try:
                    parsed_values = [segment.strip() for segment in value.split(",") if segment.strip()]
                    parsed_integers = {int(segment) for segment in parsed_values}
                except ValueError:
                    self.stdout.write("invalid group list\n")
                    return
                if any(identifier < 1 for identifier in parsed_integers):
                    self.stdout.write("group identifiers must be positive integers\n")
                    return
                allowed_group_identifiers = parsed_integers
                continue
            if name == "margin":
                try:
                    margin_multiplier = float(value)
                except ValueError:
                    self.stdout.write("invalid margin multiplier\n")
                    return
                if margin_multiplier < 1.0:
                    self.stdout.write("margin must be >= 1.0\n")
                    return
                continue
            if name == "strategy":
                strategy_id = value.strip()
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                self.stdout.write(f"invalid {name}\n")
                return
            if name == "starting_cash":
                starting_cash_value = numeric_value
            elif name == "withdraw":
                withdraw_amount = numeric_value
        # Also allow trailing options like strategy=, group=, margin=
        # to appear after the volume filter and before/after STOP/SHOW.
        post_scan_index = 0
        while post_scan_index < len(argument_parts):
            token = argument_parts[post_scan_index]
            if token.startswith("strategy="):
                strategy_id = token.split("=", 1)[1].strip()
                argument_parts.pop(post_scan_index)
                continue
            if token.startswith("group="):
                try:
                    parsed_values = [segment.strip() for segment in token.split("=", 1)[1].split(",") if segment.strip()]
                    parsed_integers = {int(segment) for segment in parsed_values}
                except ValueError:
                    self.stdout.write("invalid group list\n")
                    return
                if any(identifier < 1 for identifier in parsed_integers):
                    self.stdout.write("group identifiers must be positive integers\n")
                    return
                allowed_group_identifiers = parsed_integers
                argument_parts.pop(post_scan_index)
                continue
            if token.startswith("margin="):
                try:
                    margin_multiplier = float(token.split("=", 1)[1])
                except ValueError:
                    self.stdout.write("invalid margin multiplier\n")
                    return
                if margin_multiplier < 1.0:
                    self.stdout.write("margin must be >= 1.0\n")
                    return
                argument_parts.pop(post_scan_index)
                continue
            if token.startswith("max_positions="):
                try:
                    maximum_position_count = int(token.split("=", 1)[1])
                except ValueError:
                    self.stdout.write("invalid max_positions\n")
                    return
                if maximum_position_count < 1:
                    self.stdout.write("max_positions must be >= 1\n")
                    return
                argument_parts.pop(post_scan_index)
                continue
            if token.startswith("min_hold="):
                try:
                    minimum_holding_bars = int(token.split("=", 1)[1])
                except ValueError:
                    self.stdout.write("invalid min_hold\n")
                    return
                if minimum_holding_bars < 0:
                    self.stdout.write("min_hold must be >= 0\n")
                    return
                argument_parts.pop(post_scan_index)
                continue
            if token == "angle_confirmation_using_limit":
                use_confirmation_angle = True
                argument_parts.pop(post_scan_index)
                continue
            if token == "angle_confirmation_using_market":
                use_confirmation_angle = True
                confirmation_entry_mode = "market"
                argument_parts.pop(post_scan_index)
                continue
            post_scan_index += 1
        # Two forms supported:
        # - FILTER BUY SELL [STOP] [SHOW]
        # - FILTER [STOP] [SHOW] with strategy=ID
        stop_loss_percentage = 1.0
        take_profit_percentage = 0.0
        show_trade_details = True
        if strategy_id:
            if not argument_parts:
                self.stdout.write(
                    "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS] strategy=ID [group=...] [margin=NUMBER]\n"
                )
                return
            volume_filter = argument_parts[0]
            try:
                (
                    stop_loss_percentage,
                    take_profit_percentage,
                    show_trade_details,
                ) = _parse_stop_take_show(argument_parts[1:])
            except ValueError as error:
                error_message = str(error)
                if error_message == "too many arguments":
                    self.stdout.write(
                        "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS] strategy=ID [group=...] [margin=NUMBER]\n"
                    )
                else:
                    self.stdout.write(f"{error_message}\n")
                return
            mapping = load_strategy_set_mapping()
            if strategy_id not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_id}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_id]
            # Pass through composite strategy expressions; OR resolution handled in strategy layer
            # Pass through composite strategy expressions
            # Pass through composite strategy expressions
        else:
            if len(argument_parts) < 3:
                self.stdout.write(
                    "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [margin=NUMBER] "
                    "DOLLAR_VOLUME_FILTER (BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS] [group=1,2,...]\n"
                )
                return
            volume_filter, buy_strategy_name, sell_strategy_name = argument_parts[:3]
            try:
                (
                    stop_loss_percentage,
                    take_profit_percentage,
                    show_trade_details,
                ) = _parse_stop_take_show(argument_parts[3:])
            except ValueError as error:
                error_message = str(error)
                if error_message == "too many arguments":
                    self.stdout.write(
                        "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [margin=NUMBER] "
                        "DOLLAR_VOLUME_FILTER (BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS] [group=1,2,...]\n"
                    )
                else:
                    self.stdout.write(f"{error_message}\n")
                return
        minimum_average_dollar_volume: float | None = None  # TODO: review
        minimum_average_dollar_volume_ratio: float | None = None  # TODO: review
        top_dollar_volume_rank: int | None = None  # TODO: review
        maximum_symbols_per_group: int = 1
        pick_match = re.fullmatch(r"(.*),Pick(\d+)", volume_filter, flags=re.IGNORECASE)
        if pick_match is not None:
            volume_filter = pick_match.group(1)
            maximum_symbols_per_group = int(pick_match.group(2))
        # Support both legacy Nth and new TopN syntaxes (case-insensitive)
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
                            self.stdout.write(
                                "unsupported filter; expected dollar_volume>NUMBER, "
                                "dollar_volume>NUMBER%, dollar_volume=TopN (or Nth), "
                                "dollar_volume>NUMBER,TopN (or ,Nth), or "
                                "dollar_volume>NUMBER%,TopN (or ,Nth)\n",
                            )
                            return
        # Validate strategies; allow composite expressions (A or B)
        if not _has_supported_strategy(buy_strategy_name, strategy.BUY_STRATEGIES) or not _has_supported_strategy(
            sell_strategy_name, strategy.SELL_STRATEGIES
        ):
            self.stdout.write("unsupported strategies\n")
            return

        if start_date_string is None:
            start_date_string = determine_start_date(DATA_DIRECTORY)
        start_timestamp = pandas.Timestamp(start_date_string)
        # Load CSV price data from the dedicated stock data directory.
        extra_arguments: dict[str, object] = {}
        if maximum_symbols_per_group != 1:
            extra_arguments["maximum_symbols_per_group"] = maximum_symbols_per_group
        if margin_multiplier != 1.0:
            extra_arguments["margin_multiplier"] = margin_multiplier
            extra_arguments["margin_interest_annual_rate"] = 0.048
        self.stdout.write(f"Data source: {resolve_data_source(None).name}\n")
        evaluation_metrics = strategy.evaluate_combined_strategy(
            resolve_data_source(None),
            buy_strategy_name,
            sell_strategy_name,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            top_dollar_volume_rank=top_dollar_volume_rank,
            minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
            starting_cash=starting_cash_value,
            withdraw_amount=withdraw_amount,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
            minimum_holding_bars=minimum_holding_bars,
            use_confirmation_angle=use_confirmation_angle,
            confirmation_entry_mode=confirmation_entry_mode,
            start_date=start_timestamp,
            maximum_position_count=maximum_position_count,
            allowed_fama_french_groups=allowed_group_identifiers,
            **extra_arguments,
        )
        earliest_valid_googl_date = datetime.date(2014, 4, 3)
        filtered_trade_details_by_year: Dict[int, List[strategy.TradeDetail]] = {}
        removed_any_trade = False
        for year, trade_list in evaluation_metrics.trade_details_by_year.items():
            cleaned_trade_list = []
            for trade_detail in trade_list:
                if (
                    trade_detail.symbol == "GOOGL"
                    and trade_detail.date.date() < earliest_valid_googl_date
                ):
                    removed_any_trade = True
                    continue
                cleaned_trade_list.append(trade_detail)
            if cleaned_trade_list:
                filtered_trade_details_by_year[year] = cleaned_trade_list
        evaluation_metrics.trade_details_by_year = filtered_trade_details_by_year
        if removed_any_trade:
            all_trade_details = sorted(
                (
                    trade_detail
                    for year_trades in filtered_trade_details_by_year.values()
                    for trade_detail in year_trades
                ),
                key=lambda detail: detail.date,
            )
        else:
            # Build a flat, chronologically ordered list of all trade details
            # to support concurrent position counts used in printing below.
            all_trade_details = sorted(
                (
                    trade_detail
                    for year_trades in evaluation_metrics.trade_details_by_year.values()
                    for trade_detail in year_trades
                ),
                key=lambda detail: detail.date,
            )
        # Compute concurrent position counts for each event. Closes remove first,
        # so their count excludes the closed position; opens add, so their count
        # includes the newly opened position.
        if all_trade_details:
            # Ensure stable order for same-day events: process closes before opens
            all_trade_details.sort(
                key=lambda d: (d.date, 0 if d.action == "close" else 1)
            )
            open_symbols: Dict[str, bool] = {}
            for trade_detail in all_trade_details:
                symbol_name = trade_detail.symbol
                if trade_detail.action == "close":
                    currently_open = sum(1 for is_open in open_symbols.values() if is_open)
                    # Exclude this closing position
                    if open_symbols.get(symbol_name, False):
                        trade_detail.concurrent_position_count = max(0, currently_open - 1)
                        open_symbols[symbol_name] = False
                    else:
                        trade_detail.concurrent_position_count = currently_open
                else:  # "open"
                    currently_open = sum(1 for is_open in open_symbols.values() if is_open)
                    trade_detail.concurrent_position_count = currently_open + 1
                    open_symbols[symbol_name] = True
            close_trade_details = [
                trade_detail
                for trade_detail in all_trade_details
                if trade_detail.action == "close"
            ]
            winning_changes = [
                trade_detail.percentage_change
                for trade_detail in close_trade_details
                if trade_detail.result == "win"
                and trade_detail.percentage_change is not None
            ]
            losing_changes = [
                -trade_detail.percentage_change
                for trade_detail in close_trade_details
                if trade_detail.result == "lose"
                and trade_detail.percentage_change is not None
            ]
            open_positions: Dict[str, pandas.Timestamp] = {}
            holding_periods: List[int] = []
            for trade_detail in all_trade_details:
                if trade_detail.action == "open":
                    open_positions[trade_detail.symbol] = trade_detail.date
                elif trade_detail.action == "close":
                    entry_date = open_positions.pop(trade_detail.symbol, None)
                    if entry_date is not None:
                        holding_periods.append(
                            (trade_detail.date - entry_date).days
                        )
            evaluation_metrics.total_trades = len(close_trade_details)
            evaluation_metrics.win_rate = (
                len(winning_changes) / len(close_trade_details)
                if close_trade_details
                else 0.0
            )
            evaluation_metrics.mean_profit_percentage = (
                mean(winning_changes) if winning_changes else 0.0
            )
            evaluation_metrics.profit_percentage_standard_deviation = (
                stdev(winning_changes) if len(winning_changes) > 1 else 0.0
            )
            evaluation_metrics.mean_loss_percentage = (
                mean(losing_changes) if losing_changes else 0.0
            )
            evaluation_metrics.loss_percentage_standard_deviation = (
                stdev(losing_changes) if len(losing_changes) > 1 else 0.0
            )
            evaluation_metrics.mean_holding_period = (
                mean(holding_periods) if holding_periods else 0.0
            )
            evaluation_metrics.holding_period_standard_deviation = (
                stdev(holding_periods) if len(holding_periods) > 1 else 0.0
            )
            evaluation_metrics.annual_trade_counts = {
                year: sum(
                    1 for trade_detail in details if trade_detail.action == "close"
                )
                for year, details in filtered_trade_details_by_year.items()
            }
            evaluation_metrics.annual_returns = {
                year: annual_return
                for year, annual_return in evaluation_metrics.annual_returns.items()
                if year in filtered_trade_details_by_year
            }
        trade_records: List[Dict[str, object]] = []
        open_trade_events: Dict[str, strategy.TradeDetail] = {}
        for detail in all_trade_details:
            if detail.action == "open":
                open_trade_events[detail.symbol] = detail
            elif detail.action == "close":
                entry_detail = open_trade_events.pop(detail.symbol, None)
                if entry_detail is None:
                    continue
                trade_records.append(
                    {
                        "year": detail.date.year,
                        "entry_date": entry_detail.date.date(),
                        "concurrent_position_index": (
                            entry_detail.global_concurrent_position_count
                            if entry_detail.global_concurrent_position_count is not None
                            else entry_detail.concurrent_position_count
                        ),
                        "symbol": entry_detail.symbol,
                        "price_concentration_score": entry_detail.price_concentration_score,
                        "near_price_volume_ratio": entry_detail.near_price_volume_ratio,
                        "above_price_volume_ratio": entry_detail.above_price_volume_ratio,
                        "below_price_volume_ratio": entry_detail.below_price_volume_ratio,
                        "near_delta": entry_detail.near_delta,
                        "price_tightness": entry_detail.price_tightness,
                        "histogram_node_count": entry_detail.histogram_node_count,
                        "sma_angle": entry_detail.sma_angle,
                        "d_sma_angle": entry_detail.d_sma_angle,
                        "ema_angle": entry_detail.ema_angle,
                        "d_ema_angle": entry_detail.d_ema_angle,
                        "slope_60": entry_detail.slope_60,
                        "fuel_drawdown": entry_detail.fuel_drawdown,
                        "phantom": entry_detail.phantom,
                        "signal_bar_open": entry_detail.signal_bar_open,
                        "exit_date": detail.date.date(),
                        "result": detail.result,
                        "percentage_change": detail.percentage_change,
                        "max_favorable_excursion_pct": detail.max_favorable_excursion_pct,
                        "max_adverse_excursion_pct": detail.max_adverse_excursion_pct,
                        "max_favorable_excursion_date": (
                            detail.max_favorable_excursion_date.date()
                            if detail.max_favorable_excursion_date is not None
                            else None
                        ),
                        "max_adverse_excursion_date": (
                            detail.max_adverse_excursion_date.date()
                            if detail.max_adverse_excursion_date is not None
                            else None
                        ),
                        "exit_reason": detail.exit_reason,
                        "holding_bars": max(1, (detail.date - entry_detail.date).days * 5 // 7),
                        "profit_per_bar": (
                            detail.percentage_change / max(1, (detail.date - entry_detail.date).days * 5 // 7)
                            if detail.percentage_change is not None
                            else None
                        ),
                    }
                )
        output_directory = Path("logs") / "simulate_result"
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_directory / f"simulation_{timestamp_string}.csv"
        pandas.DataFrame(
            trade_records,
            columns=[
                "year",
                "entry_date",
                "concurrent_position_index",
                "symbol",
                "price_concentration_score",
                "near_price_volume_ratio",
                "above_price_volume_ratio",
                "below_price_volume_ratio",
                "near_delta",
                "price_tightness",
                "histogram_node_count",
                "sma_angle",
                "d_sma_angle",
                "ema_angle",
                "d_ema_angle",
                "slope_60",
                "fuel_drawdown",
                "phantom",
                "signal_bar_open",
                "exit_date",
                "result",
                "percentage_change",
                "max_favorable_excursion_pct",
                "max_adverse_excursion_pct",
                "max_favorable_excursion_date",
                "max_adverse_excursion_date",
                "exit_reason",
                "holding_bars",
                "profit_per_bar",
            ],
        ).to_csv(output_file, index=False)
        save_trade_details_to_log(
            evaluation_metrics, Path("logs") / "trade_detail"
        )
        self.stdout.write(
            f"Simulation start date: {start_date_string}\n"
        )
        self.stdout.write(
            (
                f"Trades: {evaluation_metrics.total_trades}, "
                f"Win rate: {evaluation_metrics.win_rate:.2%}, "
                f"Mean profit %: {evaluation_metrics.mean_profit_percentage:.2%}, "
                f"Profit % Std Dev: {evaluation_metrics.profit_percentage_standard_deviation:.2%}, "
                f"Mean loss %: {evaluation_metrics.mean_loss_percentage:.2%}, "
                f"Loss % Std Dev: {evaluation_metrics.loss_percentage_standard_deviation:.2%}, "
                f"Mean holding period: {evaluation_metrics.mean_holding_period:.2f} bars, "
                f"Holding period Std Dev: {evaluation_metrics.holding_period_standard_deviation:.2f} bars, "
                f"Max concurrent positions: {evaluation_metrics.maximum_concurrent_positions}, "
                f"Final balance: {evaluation_metrics.final_balance:.2f}, "
                f"CAGR: {evaluation_metrics.compound_annual_growth_rate:.2%}, "
                f"Max drawdown: {evaluation_metrics.maximum_drawdown:.2%}\n"
            )
        )
        for year, annual_return in sorted(
            evaluation_metrics.annual_returns.items()
        ):
            trade_count = evaluation_metrics.annual_trade_counts.get(year, 0)
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, trade: {trade_count}\n"
            )
            if show_trade_details:  # TODO: review
                trade_details = evaluation_metrics.trade_details_by_year.get(year, [])
                for trade_detail in trade_details:
                    if (
                        trade_detail.action == "close"
                        and trade_detail.result is not None
                    ):
                        if trade_detail.percentage_change is not None:
                            result_suffix = (
                                f" {trade_detail.result} "
                                f"{trade_detail.percentage_change:.2%} "
                                f"{trade_detail.exit_reason}"
                            )
                        else:
                            result_suffix = (
                                f" {trade_detail.result} "
                                f"{trade_detail.exit_reason}"
                            )
                    else:
                        result_suffix = ""
                    open_metrics = ""
                    if trade_detail.action == "open":
                        price_score_text = (
                            f"{trade_detail.price_concentration_score:.2f}"
                            if trade_detail.price_concentration_score is not None
                            else "N/A"
                        )
                        near_ratio_text = (
                            f"{trade_detail.near_price_volume_ratio:.2f}"
                            if trade_detail.near_price_volume_ratio is not None
                            else "N/A"
                        )
                        above_ratio_text = (
                            f"{trade_detail.above_price_volume_ratio:.2f}"
                            if trade_detail.above_price_volume_ratio is not None
                            else "N/A"
                        )
                        node_count_text = (
                            f"{trade_detail.histogram_node_count}"
                            if trade_detail.histogram_node_count is not None
                            else "N/A"
                        )
                        open_metrics = (
                            f" price_score={price_score_text}"
                            f" near_pct={near_ratio_text}"
                            f" above_pct={above_ratio_text}"
                            f" node_count={node_count_text}"
                        )
                    self.stdout.write(
                        (
                            f"  {trade_detail.date.date()} ({trade_detail.concurrent_position_count}) "
                            f"{trade_detail.symbol} "
                            f"{trade_detail.action} {trade_detail.price:.2f} "
                            # Show ratio within FF12 group and group total dollar volume
                            f"{trade_detail.group_simple_moving_average_dollar_volume_ratio:.4f} "
                            f"{trade_detail.simple_moving_average_dollar_volume / 1_000_000:.2f}M "
                            f"{trade_detail.group_total_simple_moving_average_dollar_volume / 1_000_000:.2f}M"
                            f"{open_metrics}{result_suffix}\n"
                        )
                    )

    # TODO: review
    def help_start_simulate(self) -> None:
        """Display help for the start_simulate command."""
        available_buy = ", ".join(sorted(strategy.BUY_STRATEGIES.keys()))
        available_sell = ", ".join(sorted(strategy.SELL_STRATEGIES.keys()))
        self.stdout.write(
            "start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [margin=NUMBER] "
            "DOLLAR_VOLUME_FILTER (BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS] [group=1,2,...]\n"
            "Evaluate trading strategies using cached data.\n"
            "Parameters:\n"
            "  starting_cash: Initial cash balance for the simulation. Defaults to 3000.\n"
            "  withdraw: Amount deducted from cash at each year end. Defaults to 0.\n"
            "  start: Date in YYYY-MM-DD format to begin the simulation. Defaults to the earliest available date.\n"
            "  DOLLAR_VOLUME_FILTER: Use dollar_volume>NUMBER (in millions),\n"
            "    dollar_volume>N% (effective market threshold computed per FF12\n"
            "    group as N% divided by that group's market share),\n"
            "    dollar_volume=TopN (global Top-N each day with at most one\n"
            "    symbol per FF12 group), or combine with ranking using\n"
            "    dollar_volume>NUMBER,TopN or dollar_volume>N%,TopN. Legacy 'Nth' is also accepted for\n"
            "    backward compatibility. Known non-stock instruments are excluded.\n"
            "  BUY/SELL or strategy=ID: Either provide explicit buy/sell strategy names, "
            "or a strategy id defined in data/strategy_sets.csv.\n"
            "  STOP_LOSS: Fractional loss for stop orders. If intraday low hits\n"
            "    the stop, exits on the same bar at the stop price; otherwise, if\n"
            "    the close is below the stop, exits on the next day's open.\n"
            "    Defaults to 1.0 (disabled).\n"
            "  TAKE_PROFIT: Fractional gain for profit targets. If intraday high reaches\n"
            "    the target, exits on the same bar at the target price; otherwise, if\n"
            "    the close is above the target, exits on the next day's open.\n"
            "    Defaults to 0.0 (disabled).\n"
            "  SHOW_DETAILS: 'True' to print individual trades, 'False' to suppress them. Defaults to True.\n"
            "  group: Optional comma-separated FF12 group ids (1-11) to restrict\n"
            "    tradable symbols. Group 12 (Other) is selectable. Example:\n"
            "    group=1,2,4,6,7,8,10,11\n"
            "Strategies may be suffixed with _N to set the window size to N; the default window size is 40 when no suffix is provided.\n"
            "Slope-aware strategies follow the ema_sma_signal_with_slope_n_k pattern and accept _LOWER_UPPER bounds after the optional window size; both bounds are floating-point numbers and may be negative.\n"
            "Example: start_simulate start=1990-01-01 dollar_volume>50 ema_sma_cross_20 ema_sma_cross_20\n"
            "Another example: start_simulate dollar_volume>1 ema_sma_signal_with_slope_-0.1_1.2 ema_sma_signal_with_slope_-0.1_1.2\n"
            f"Available buy strategies: {available_buy}.\n"
            f"Available sell strategies: {available_sell}.\n"
        )

    
    def do_start_simulate_single_symbol(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]
        Evaluate strategies on a single symbol using full allocation per position.

        When not provided, STOP_LOSS defaults to 1.0, TAKE_PROFIT defaults to 0.0, and SHOW_DETAILS defaults to True.
        """
        argument_parts: List[str] = argument_line.split()
        symbol_name: str | None = None
        starting_cash_value = 3000.0
        withdraw_amount = 0.0
        start_date_string: str | None = None
        strategy_id: str | None = None
        while argument_parts and (
            argument_parts[0].startswith("symbol=")
            or argument_parts[0].startswith("starting_cash=")
            or argument_parts[0].startswith("withdraw=")
            or argument_parts[0].startswith("start=")
            or argument_parts[0].startswith("strategy=")
        ):
            parameter_part = argument_parts.pop(0)
            name, value = parameter_part.split("=", 1)
            if name == "symbol":
                symbol_name = value.strip().upper()
                continue
            if name == "start":
                try:
                    datetime.date.fromisoformat(value)
                except ValueError:
                    self.stdout.write("invalid start date\n")
                    return
                start_date_string = value
                continue
            if name == "strategy":
                strategy_id = value.strip()
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                self.stdout.write(f"invalid {name}\n")
                return
            if name == "starting_cash":
                starting_cash_value = numeric_value
            elif name == "withdraw":
                withdraw_amount = numeric_value
        # Allow strategy=ID to appear after positional tokens as well
        scan_index = 0
        while scan_index < len(argument_parts):
            token = argument_parts[scan_index]
            if token.startswith("strategy="):
                strategy_id = token.split("=", 1)[1].strip()
                argument_parts.pop(scan_index)
                continue
            scan_index += 1
        # Accept two forms:
        # - BUY SELL [STOP] [SHOW]
        # - [STOP] [SHOW] with strategy=ID
        stop_loss_percentage = 1.0
        take_profit_percentage = 0.0
        show_trade_details = True
        if strategy_id:
            try:
                (
                    stop_loss_percentage,
                    take_profit_percentage,
                    show_trade_details,
                ) = _parse_stop_take_show(argument_parts)
            except ValueError as error:
                error_message = str(error)
                if error_message == "too many arguments":
                    self.stdout.write(
                        "usage: start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS] strategy=ID\n"
                    )
                else:
                    self.stdout.write(f"{error_message}\n")
                return
            mapping = load_strategy_set_mapping()
            if strategy_id not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_id}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_id]
            # Pass through composite strategy expressions
        else:
            if len(argument_parts) < 2:
                self.stdout.write(
                    "usage: start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
                    "(BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]\n"
                )
                return
            buy_strategy_name, sell_strategy_name = argument_parts[:2]
            try:
                (
                    stop_loss_percentage,
                    take_profit_percentage,
                    show_trade_details,
                ) = _parse_stop_take_show(argument_parts[2:])
            except ValueError as error:
                error_message = str(error)
                if error_message == "too many arguments":
                    self.stdout.write(
                        "usage: start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
                        "(BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]\n"
                    )
                else:
                    self.stdout.write(f"{error_message}\n")
                return
        # Validate strategies; support composite expressions
        if not _has_supported_strategy(buy_strategy_name, strategy.BUY_STRATEGIES) or not _has_supported_strategy(
            sell_strategy_name, strategy.SELL_STRATEGIES
        ):
            self.stdout.write("unsupported strategies\n")
            return

        # Determine data directory and ensure the symbol file exists
        base_directory = resolve_data_source(None)
        self.stdout.write(f"Data source: {base_directory.name}\n")
        if symbol_name is None:
            self.stdout.write("symbol is required: provide symbol=SYMBOL\n")
            return
        data_file_path = base_directory / f"{symbol_name}.csv"
        if not data_file_path.exists():
            self.stdout.write(f"data file not found: {data_file_path}\n")
            return

        if start_date_string is None:
            start_date_string = determine_start_date(base_directory)
        start_timestamp = pandas.Timestamp(start_date_string)

        evaluation_metrics = strategy.evaluate_combined_strategy(
            base_directory,
            buy_strategy_name,
            sell_strategy_name,
            starting_cash=starting_cash_value,
            withdraw_amount=withdraw_amount,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
            start_date=start_timestamp,
            maximum_position_count=1,  # full allocation per position
            allowed_symbols={symbol_name},
            exclude_other_ff12=False,
        )

        # Compute concurrent position counts for accurate detail printing
        all_trade_details = sorted(
            (
                trade_detail
                for year_trades in evaluation_metrics.trade_details_by_year.values()
                for trade_detail in year_trades
            ),
            key=lambda detail: detail.date,
        )
        if all_trade_details:
            all_trade_details.sort(
                key=lambda d: (d.date, 0 if d.action == "close" else 1)
            )
            open_state: Dict[str, bool] = {}
            for detail in all_trade_details:
                if detail.action == "close":
                    current_open = sum(1 for is_open in open_state.values() if is_open)
                    if open_state.get(detail.symbol, False):
                        detail.concurrent_position_count = max(0, current_open - 1)
                        open_state[detail.symbol] = False
                    else:
                        detail.concurrent_position_count = current_open
                else:
                    current_open = sum(1 for is_open in open_state.values() if is_open)
                    detail.concurrent_position_count = current_open + 1
                    open_state[detail.symbol] = True

        save_trade_details_to_log(
            evaluation_metrics, Path("logs") / "trade_detail"
        )
        self.stdout.write(
            f"Simulation start date: {start_date_string}\n"
        )
        self.stdout.write(
            (
                f"Trades: {evaluation_metrics.total_trades}, "
                f"Win rate: {evaluation_metrics.win_rate:.2%}, "
                f"Mean profit %: {evaluation_metrics.mean_profit_percentage:.2%}, "
                f"Mean loss %: {evaluation_metrics.mean_loss_percentage:.2%}, "
                f"Max concurrent positions: {evaluation_metrics.maximum_concurrent_positions}, "
                f"Final balance: {evaluation_metrics.final_balance:.2f}, "
                f"CAGR: {evaluation_metrics.compound_annual_growth_rate:.2%}, "
                f"Max drawdown: {evaluation_metrics.maximum_drawdown:.2%}\n"
            )
        )
        for year, annual_return in sorted(
            evaluation_metrics.annual_returns.items()
        ):
            trade_count = evaluation_metrics.annual_trade_counts.get(year, 0)
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, trade: {trade_count}\n"
            )
        if show_trade_details:
            for year in sorted(evaluation_metrics.trade_details_by_year.keys()):
                trade_details = evaluation_metrics.trade_details_by_year.get(year, [])
                for trade_detail in trade_details:
                    if (
                        trade_detail.action == "close"
                        and trade_detail.result is not None
                    ):
                        if trade_detail.percentage_change is not None:
                            result_suffix = (
                                f" {trade_detail.result} "
                                f"{trade_detail.percentage_change:.2%} "
                                f"{trade_detail.exit_reason}"
                            )
                        else:
                            result_suffix = (
                                f" {trade_detail.result} "
                                f"{trade_detail.exit_reason}"
                            )
                    else:
                        result_suffix = ""
                    open_metrics = ""
                    if trade_detail.action == "open":
                        price_score_text = (
                            f"{trade_detail.price_concentration_score:.2f}"
                            if trade_detail.price_concentration_score is not None
                            else "N/A"
                        )
                        near_ratio_text = (
                            f"{trade_detail.near_price_volume_ratio:.2f}"
                            if trade_detail.near_price_volume_ratio is not None
                            else "N/A"
                        )
                        above_ratio_text = (
                            f"{trade_detail.above_price_volume_ratio:.2f}"
                            if trade_detail.above_price_volume_ratio is not None
                            else "N/A"
                        )
                        node_count_text = (
                            f"{trade_detail.histogram_node_count}"
                            if trade_detail.histogram_node_count is not None
                            else "N/A"
                        )
                        open_metrics = (
                            f" price_score={price_score_text}"
                            f" near_pct={near_ratio_text}"
                            f" above_pct={above_ratio_text}"
                            f" node_count={node_count_text}"
                        )
                    self.stdout.write(
                        (
                            f"  {trade_detail.date.date()} ({trade_detail.concurrent_position_count}) "
                            f"{trade_detail.symbol} {trade_detail.action} {trade_detail.price:.2f} "
                            f"{trade_detail.group_simple_moving_average_dollar_volume_ratio:.4f} "
                            f"{trade_detail.simple_moving_average_dollar_volume / 1_000_000:.2f}M "
                            f"{trade_detail.group_total_simple_moving_average_dollar_volume / 1_000_000:.2f}M"
                            f"{open_metrics}{result_suffix}\n"
                        )
                    )

    def help_start_simulate_single_symbol(self) -> None:
        """Display help for the start_simulate_single_symbol command."""
        self.stdout.write(
            "start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
            "(BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]\n"
            "Simulate strategies on a single symbol using full allocation per position.\n"
            "Parameters:\n"
            "  symbol: Ticker symbol to simulate (required).\n"
            "  starting_cash: Initial cash balance. Defaults to 3000.\n"
            "  withdraw: Amount deducted from cash at each year end. Defaults to 0.\n"
            "  start: Start date in YYYY-MM-DD format. Defaults to earliest available.\n"
            "  BUY/SELL or strategy=ID: Either provide explicit strategy names, or a strategy id defined in data/strategy_sets.csv.\n"
            "  STOP_LOSS: Fractional loss for stop orders (same-bar at stop price when low hits; otherwise next-day open). Defaults to 1.0.\n"
            "  TAKE_PROFIT: Fractional gain for profit targets (same-bar at target when high reaches it; otherwise next-day open). Defaults to 0.0 (disabled).\n"
            "  SHOW_DETAILS: 'True' to print trade details, 'False' to suppress. Defaults to True.\n"
        )

    def do_start_simulate_n_symbol(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]
        Evaluate strategies across a provided symbol list. Budget per position uses slot count equal to the number of symbols.

        When not provided, STOP_LOSS defaults to 1.0, TAKE_PROFIT defaults to 0.0, and SHOW_DETAILS defaults to True.
        """
        argument_parts: List[str] = argument_line.split()
        symbol_list_input: str | None = None
        starting_cash_value = 3000.0
        withdraw_amount = 0.0
        start_date_string: str | None = None
        strategy_id: str | None = None
        while argument_parts and (
            argument_parts[0].startswith("symbols=")
            or argument_parts[0].startswith("starting_cash=")
            or argument_parts[0].startswith("withdraw=")
            or argument_parts[0].startswith("start=")
            or argument_parts[0].startswith("strategy=")
        ):
            parameter_part = argument_parts.pop(0)
            name, value = parameter_part.split("=", 1)
            if name == "symbols":
                symbol_list_input = value
                continue
            if name == "start":
                try:
                    datetime.date.fromisoformat(value)
                except ValueError:
                    self.stdout.write("invalid start date\n")
                    return
                start_date_string = value
                continue
            try:
                numeric_value = float(value)
            except ValueError:
                self.stdout.write(f"invalid {name}\n")
                return
            if name == "starting_cash":
                starting_cash_value = numeric_value
            elif name == "withdraw":
                withdraw_amount = numeric_value
        # Allow strategy=ID to appear after positional tokens as well
        scan_index = 0
        while scan_index < len(argument_parts):
            token = argument_parts[scan_index]
            if token.startswith("strategy="):
                strategy_id = token.split("=", 1)[1].strip()
                argument_parts.pop(scan_index)
                continue
            scan_index += 1

        if symbol_list_input is None:
            self.stdout.write(
                "usage: start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
                "(BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]\n"
            )
            return
        stop_loss_percentage = 1.0
        take_profit_percentage = 0.0
        show_trade_details = True
        if strategy_id:
            try:
                (
                    stop_loss_percentage,
                    take_profit_percentage,
                    show_trade_details,
                ) = _parse_stop_take_show(argument_parts)
            except ValueError as error:
                error_message = str(error)
                if error_message == "too many arguments":
                    self.stdout.write(
                        "usage: start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS] strategy=ID\n"
                    )
                else:
                    self.stdout.write(f"{error_message}\n")
                return
            mapping = load_strategy_set_mapping()
            if strategy_id not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_id}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_id]
        else:
            if len(argument_parts) < 2:
                self.stdout.write(
                    "usage: start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
                    "(BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]\n"
                )
                return
            buy_strategy_name, sell_strategy_name = argument_parts[:2]
            try:
                (
                    stop_loss_percentage,
                    take_profit_percentage,
                    show_trade_details,
                ) = _parse_stop_take_show(argument_parts[2:])
            except ValueError as error:
                error_message = str(error)
                if error_message == "too many arguments":
                    self.stdout.write(
                        "usage: start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
                        "(BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]\n"
                    )
                else:
                    self.stdout.write(f"{error_message}\n")
                return

        # Validate strategies; support composite expressions
        if not _has_supported_strategy(buy_strategy_name, strategy.BUY_STRATEGIES) or not _has_supported_strategy(
            sell_strategy_name, strategy.SELL_STRATEGIES
        ):
            self.stdout.write("unsupported strategies\n")
            return

        base_directory = resolve_data_source(None)
        self.stdout.write(f"Data source: {base_directory.name}\n")
        requested_symbols = [
            token.strip().upper()
            for token in symbol_list_input.split(",")
            if token.strip()
        ]
        if not requested_symbols:
            self.stdout.write("no symbols provided\n")
            return
        existing_symbols: List[str] = []
        for symbol_name in requested_symbols:
            if (base_directory / f"{symbol_name}.csv").exists():
                existing_symbols.append(symbol_name)
            else:
                self.stdout.write(f"warning: data file not found for {symbol_name}, skipping\n")
        if not existing_symbols:
            self.stdout.write("no valid symbols with data files found\n")
            return

        if start_date_string is None:
            start_date_string = determine_start_date(base_directory)
        start_timestamp = pandas.Timestamp(start_date_string)

        evaluation_metrics = strategy.evaluate_combined_strategy(
            base_directory,
            buy_strategy_name,
            sell_strategy_name,
            starting_cash=starting_cash_value,
            withdraw_amount=withdraw_amount,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
            start_date=start_timestamp,
            maximum_position_count=len(existing_symbols),  # slots = symbol count
            allowed_symbols=set(existing_symbols),
            exclude_other_ff12=False,  # honor explicit user list
        )

        # Compute concurrent position counts for detail printing
        all_trade_details = sorted(
            (
                trade_detail
                for year_trades in evaluation_metrics.trade_details_by_year.values()
                for trade_detail in year_trades
            ),
            key=lambda detail: detail.date,
        )
        if all_trade_details:
            all_trade_details.sort(
                key=lambda d: (d.date, 0 if d.action == "close" else 1)
            )
            open_state: Dict[str, bool] = {}
            for detail in all_trade_details:
                if detail.action == "close":
                    current_open = sum(1 for is_open in open_state.values() if is_open)
                    if open_state.get(detail.symbol, False):
                        detail.concurrent_position_count = max(0, current_open - 1)
                        open_state[detail.symbol] = False
                    else:
                        detail.concurrent_position_count = current_open
                else:
                    current_open = sum(1 for is_open in open_state.values() if is_open)
                    detail.concurrent_position_count = current_open + 1
                    open_state[detail.symbol] = True

        save_trade_details_to_log(
            evaluation_metrics, Path("logs") / "trade_detail"
        )
        self.stdout.write(
            f"Simulation start date: {start_date_string}\n"
        )
        self.stdout.write(
            (
                f"Trades: {evaluation_metrics.total_trades}, "
                f"Win rate: {evaluation_metrics.win_rate:.2%}, "
                f"Mean profit %: {evaluation_metrics.mean_profit_percentage:.2%}, "
                f"Mean loss %: {evaluation_metrics.mean_loss_percentage:.2%}, "
                f"Max concurrent positions: {evaluation_metrics.maximum_concurrent_positions}, "
                f"Final balance: {evaluation_metrics.final_balance:.2f}, "
                f"CAGR: {evaluation_metrics.compound_annual_growth_rate:.2%}, "
                f"Max drawdown: {evaluation_metrics.maximum_drawdown:.2%}\n"
            )
        )
        for year, annual_return in sorted(
            evaluation_metrics.annual_returns.items()
        ):
            trade_count = evaluation_metrics.annual_trade_counts.get(year, 0)
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, trade: {trade_count}\n"
            )
        if show_trade_details:
            for year in sorted(evaluation_metrics.trade_details_by_year.keys()):
                trade_details = evaluation_metrics.trade_details_by_year.get(year, [])
                for trade_detail in trade_details:
                    if (
                        trade_detail.action == "close"
                        and trade_detail.result is not None
                    ):
                        if trade_detail.percentage_change is not None:
                            result_suffix = (
                                f" {trade_detail.result} "
                                f"{trade_detail.percentage_change:.2%} "
                                f"{trade_detail.exit_reason}"
                            )
                        else:
                            result_suffix = (
                                f" {trade_detail.result} "
                                f"{trade_detail.exit_reason}"
                            )
                    else:
                        result_suffix = ""
                    open_metrics = ""
                    if trade_detail.action == "open":
                        price_score_text = (
                            f"{trade_detail.price_concentration_score:.2f}"
                            if trade_detail.price_concentration_score is not None
                            else "N/A"
                        )
                        near_ratio_text = (
                            f"{trade_detail.near_price_volume_ratio:.2f}"
                            if trade_detail.near_price_volume_ratio is not None
                            else "N/A"
                        )
                        above_ratio_text = (
                            f"{trade_detail.above_price_volume_ratio:.2f}"
                            if trade_detail.above_price_volume_ratio is not None
                            else "N/A"
                        )
                        node_count_text = (
                            f"{trade_detail.histogram_node_count}"
                            if trade_detail.histogram_node_count is not None
                            else "N/A"
                        )
                        open_metrics = (
                            f" price_score={price_score_text}"
                            f" near_pct={near_ratio_text}"
                            f" above_pct={above_ratio_text}"
                            f" node_count={node_count_text}"
                        )
                    self.stdout.write(
                        (
                            f"  {trade_detail.date.date()} ({trade_detail.concurrent_position_count}) "
                            f"{trade_detail.symbol} {trade_detail.action} {trade_detail.price:.2f} "
                            f"{trade_detail.group_simple_moving_average_dollar_volume_ratio:.4f} "
                            f"{trade_detail.simple_moving_average_dollar_volume / 1_000_000:.2f}M "
                            f"{trade_detail.group_total_simple_moving_average_dollar_volume / 1_000_000:.2f}M"
                            f"{open_metrics}{result_suffix}\n"
                        )
                    )

    def help_start_simulate_n_symbol(self) -> None:
        """Display help for the start_simulate_n_symbol command."""
        self.stdout.write(
            "start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
            "(BUY SELL | [strategy=ID]) [STOP_LOSS] [TAKE_PROFIT] [SHOW_DETAILS]\n"
            "Simulate strategies across a list of symbols; budget per position uses slots equal to the list length.\n"
            "Parameters:\n"
            "  symbols: Comma-separated ticker symbols to simulate (required).\n"
            "  starting_cash: Initial cash balance. Defaults to 3000.\n"
            "  withdraw: Amount deducted from cash at each year end. Defaults to 0.\n"
            "  start: Start date in YYYY-MM-DD format. Defaults to earliest available.\n"
            "  BUY/SELL or strategy=ID: Either provide explicit strategy names, or a strategy id defined in data/strategy_sets.csv.\n"
            "  STOP_LOSS: Fractional loss for stop orders (same-bar at stop price when low hits; otherwise next-day open). Defaults to 1.0.\n"
            "  TAKE_PROFIT: Fractional gain for profit targets (same-bar at target when high reaches it; otherwise next-day open). Defaults to 0.0 (disabled).\n"
            "  SHOW_DETAILS: 'True' to print trade details, 'False' to suppress. Defaults to True.\n"
        )


    # TODO: review
    def do_find_history_signal(self, argument_line: str) -> None:  # noqa: D401
        """find_history_signal [DATE] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY STOP_LOSS
        [group=...] or find_history_signal [DATE] DOLLAR_VOLUME_FILTER STOP_LOSS strategy=ID
        [group=...]

        Display the entry and exit signals generated for DATE or the latest trading day when DATE is omitted."""
        usage_message = (
            "usage: find_history_signal [DATE] DOLLAR_VOLUME_FILTER (BUY SELL STOP_LOSS | STOP_LOSS strategy=ID) [group=1,2,...]\n"
        )
        argument_parts: List[str] = argument_line.split()
        if len(argument_parts) < 3:
            self.stdout.write(usage_message)
            return  # TODO: review
        # Optional group token may appear in any position after DATE; normalize
        allowed_group_identifiers: set[int] | None = None
        tokens: List[str] = []
        strategy_id: str | None = None
        take_profit_display: str | None = None
        max_positions_display: int | None = None
        min_hold_bars: int = 0
        for token in argument_parts:
            if token.startswith("group="):
                try:
                    raw = token.split("=", 1)[1]
                    parts = [p.strip() for p in raw.split(",") if p.strip()]
                    parsed = {int(p) for p in parts}
                except ValueError:
                    self.stdout.write("invalid group list\n")
                    return
                if any(identifier < 1 for identifier in parsed):
                    self.stdout.write("group identifiers must be positive integers\n")
                    return
                allowed_group_identifiers = parsed
            elif token.startswith("strategy="):
                strategy_id = token.split("=", 1)[1].strip()
            elif token.startswith("tp="):
                tp_val = float(token.split("=", 1)[1])
                take_profit_display = f"{tp_val:.1%}" if tp_val > 0 else "NOPE"
            elif token.startswith("max_pos="):
                max_positions_display = int(token.split("=", 1)[1])
            elif token.startswith("min_hold="):
                min_hold_bars = int(token.split("=", 1)[1])
            else:
                tokens.append(token)
        try:
            datetime.date.fromisoformat(tokens[0])
            date_string = tokens.pop(0)
        except ValueError:
            date_string = None
        # Support two forms:
        # 1) [DATE] FILTER BUY SELL STOP
        # 2) [DATE] FILTER STOP with strategy=ID
        if strategy_id:
            if len(tokens) != 2:
                self.stdout.write(usage_message)
                return
            (
                dollar_volume_filter,
                stop_loss_string,
            ) = tokens
            mapping = load_strategy_set_mapping()
            if strategy_id not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_id}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_id]
        else:
            if len(tokens) != 4:
                self.stdout.write(usage_message)
                return
            (
                dollar_volume_filter,
                buy_strategy_name,
                sell_strategy_name,
                stop_loss_string,
            ) = tokens
        if date_string is not None:
            try:
                datetime.date.fromisoformat(date_string)
            except ValueError:
                self.stdout.write(usage_message)
                return
        try:
            stop_loss_value = float(stop_loss_string)
        except ValueError:
            self.stdout.write("invalid stop loss\n")
            return
        # Load entry filters for the strategy (near_delta_range, etc.)
        near_delta_range_for_signal: tuple[float, float] | None = None
        price_tightness_range_for_signal: tuple[float, float] | None = None
        if strategy_id:
            entry_filters_mapping = load_strategy_entry_filters()
            if strategy_id in entry_filters_mapping:
                ef = entry_filters_mapping[strategy_id]
                if ef.near_delta_min is not None or ef.near_delta_max is not None:
                    near_delta_range_for_signal = (
                        ef.near_delta_min if ef.near_delta_min is not None else -99.0,
                        ef.near_delta_max if ef.near_delta_max is not None else 99.0,
                    )
                if ef.price_tightness_min is not None or ef.price_tightness_max is not None:
                    price_tightness_range_for_signal = (
                        ef.price_tightness_min if ef.price_tightness_min is not None else 0.0,
                        ef.price_tightness_max if ef.price_tightness_max is not None else 99.0,
                    )
        signal_data: Dict[str, Any] = daily_job.find_history_signal(
            date_string,
            dollar_volume_filter,
            buy_strategy_name,
            sell_strategy_name,
            stop_loss_value,
            allowed_group_identifiers,
            near_delta_range=near_delta_range_for_signal,
            price_tightness_range=price_tightness_range_for_signal,
        )
        filtered_symbol_list: List[tuple[str, int | None]] = signal_data.get(
            "filtered_symbols", []
        )
        # Strategy header
        self.stdout.write(f"--- {strategy_id or 'signal'} ---\n")
        self.stdout.write(f"{argument_line}\n")
        self.stdout.write(f"filtered symbols: {filtered_symbol_list}\n")
        entry_signal_list: List[str] = signal_data.get("entry_signals", [])
        if strategy_id in {"s4", "s6"} and entry_signal_list:
            if date_string is None:
                effective_date_string = (
                    daily_job.determine_latest_trading_date().isoformat()
                )
            else:
                effective_date_string = date_string
            above_ratio_by_symbol: Dict[str, float | None] = {}
            for symbol_name in entry_signal_list:
                try:
                    debug_values = daily_job.filter_debug_values(
                        symbol_name,
                        effective_date_string,
                        buy_strategy_name,
                        sell_strategy_name,
                    )
                except Exception:  # noqa: BLE001
                    debug_values = {}
                raw_ratio_value = debug_values.get("above_price_volume_ratio")
                ratio_value: float | None
                if raw_ratio_value is None:
                    ratio_value = None
                else:
                    try:
                        ratio_value = float(raw_ratio_value)
                    except (TypeError, ValueError):
                        ratio_value = None
                above_ratio_by_symbol[symbol_name] = ratio_value
            entry_signal_list = sorted(
                entry_signal_list,
                key=lambda name: (
                    above_ratio_by_symbol.get(name) is None,
                    above_ratio_by_symbol.get(name, float("inf")),
                    name,
                ),
            )
        filter_exit_signal_list: List[str] = signal_data.get("exit_signals", [])

        # Sort entry signals by dollar volume rank (filtered_symbols is
        # already sorted biggest-first by compute_signals_for_date).
        filtered_symbol_order: Dict[str, int] = {
            sym: idx for idx, (sym, _) in enumerate(filtered_symbol_list)
        }
        entry_signal_list = sorted(
            entry_signal_list,
            key=lambda name: filtered_symbol_order.get(name, len(filtered_symbol_order)),
        )

        # Signal trade tracking: load signal_trades.json to know which symbols
        # have active entry signals.  When a signal exit fires, the trade is
        # recorded into adaptive_state.json for rolling TP/SL computation.
        # This is NOT live portfolio state — actual positions come from Futu API.
        # Format: {"strategy_id": [{"symbol": "X", "entry_date": "YYYY-MM-DD"}, ...]}
        positions_path = LIVE_STATE_DIRECTORY / "signal_trades.json"
        all_positions: Dict[str, List[Dict[str, str]]] = {}
        if positions_path.exists():
            try:
                with positions_path.open("r", encoding="utf-8") as fp:
                    raw_positions = json.load(fp)
                # Migrate legacy format (list of strings) to new format
                for key, val in raw_positions.items():
                    if isinstance(val, list):
                        migrated: List[Dict[str, str]] = []
                        for item in val:
                            if isinstance(item, str):
                                migrated.append({"symbol": item, "entry_date": ""})
                            elif isinstance(item, dict):
                                migrated.append(item)
                        all_positions[key] = migrated
                    else:
                        all_positions[key] = []
            except (json.JSONDecodeError, OSError):
                all_positions = {}
        held_entries: List[Dict[str, str]] = all_positions.get(
            strategy_id or "", []
        )
        held_symbols: List[str] = [e["symbol"] for e in held_entries]
        held_entry_dates: Dict[str, str] = {
            e["symbol"]: e.get("entry_date", "") for e in held_entries
        }

        # Determine evaluation date for exit checks and bar counting
        if date_string is None:
            effective_date_for_exit = (
                daily_job.determine_latest_trading_date().isoformat()
            )
        else:
            effective_date_for_exit = date_string

        # Count trading bars between entry_date and evaluation date
        def _bars_held(entry_date_str: str) -> int:
            if not entry_date_str:
                return 9999  # unknown entry date, assume long enough
            try:
                entry_ts = pandas.Timestamp(entry_date_str)
                eval_ts = pandas.Timestamp(effective_date_for_exit)
                # Approximate trading bars (weekdays)
                trading_days = pandas.bdate_range(entry_ts, eval_ts)
                return max(0, len(trading_days) - 1)
            except Exception:  # noqa: BLE001
                return 9999

        # Check exit for held symbols not already in filter exit list
        held_exit_signals: List[str] = []
        held_min_hold_blocked: List[str] = []
        for held_symbol in held_symbols:
            bars = _bars_held(held_entry_dates.get(held_symbol, ""))
            has_exit = held_symbol in filter_exit_signal_list
            if not has_exit:
                try:
                    debug_values = daily_job.filter_debug_values(
                        held_symbol,
                        effective_date_for_exit,
                        buy_strategy_name,
                        sell_strategy_name,
                    )
                except Exception:  # noqa: BLE001
                    debug_values = {}
                has_exit = debug_values.get("exit", False)
            if has_exit:
                if bars < min_hold_bars:
                    held_min_hold_blocked.append(held_symbol)
                else:
                    held_exit_signals.append(held_symbol)

        # Merge exit signals: filter + held positions
        all_exit_signals: List[str] = list(filter_exit_signal_list)
        for sym in held_exit_signals:
            if sym not in all_exit_signals:
                all_exit_signals.append(sym)

        self.stdout.write(f"entry signals: {entry_signal_list}\n")
        self.stdout.write(f"exit signals: {all_exit_signals}\n")

        # Update adaptive_state.json with closed positions' raw P/L.
        # The user must manually fill in exit_price after closing;
        # alternatively, compute_adaptive_tp_sl can be called with
        # the actual exit price.
        if held_exit_signals:
            state_path = LIVE_STATE_DIRECTORY / "adaptive_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            adaptive_state: dict = {"raw_trade_profits": [], "closed_trades": []}
            if state_path.exists():
                try:
                    with state_path.open("r", encoding="utf-8") as fp:
                        adaptive_state = json.load(fp)
                except (json.JSONDecodeError, OSError):
                    pass
            raw_profits = adaptive_state.get("raw_trade_profits", [])
            closed_list = adaptive_state.get("closed_trades", [])
            for exit_sym in held_exit_signals:
                entry_rec = next(
                    (e for e in held_entries if e["symbol"] == exit_sym), None
                )
                if entry_rec:
                    # Record closed trade. entry_price may be None if
                    # compute_adaptive_tp_sl hasn't run yet; it will be
                    # auto-filled on the next compute_adaptive_tp_sl call.
                    closed_list.append({
                        "symbol": exit_sym,
                        "entry_date": entry_rec.get("entry_date", ""),
                        "exit_date": effective_date_for_exit,
                        "entry_price": entry_rec.get("entry_price"),
                        "exit_price": None,
                        "raw_pct": None,
                    })
            adaptive_state["raw_trade_profits"] = raw_profits[-20:]
            adaptive_state["closed_trades"] = closed_list
            try:
                with state_path.open("w", encoding="utf-8") as fp:
                    json.dump(adaptive_state, fp, indent=2)
            except OSError:
                pass

        # Compute expected positions after today's actions
        expected_entries = [e for e in held_entries if e["symbol"] not in held_exit_signals]
        for symbol_name in entry_signal_list:
            if symbol_name not in [e["symbol"] for e in expected_entries]:
                expected_entries.append({
                    "symbol": symbol_name,
                    "entry_date": effective_date_for_exit,
                })

        # Apply max_positions cap to expected positions
        if max_positions_display is not None:
            expected_entries = expected_entries[:max_positions_display]

        expected_symbols = [e["symbol"] for e in expected_entries]

        # Save to signal_trades.json
        if strategy_id:
            all_positions[strategy_id] = expected_entries
            try:
                positions_path.parent.mkdir(parents=True, exist_ok=True)
                with positions_path.open("w", encoding="utf-8") as fp:
                    json.dump(all_positions, fp, indent=2)
            except OSError as write_error:
                self.stdout.write(
                    f"warning: failed to write signal_trades.json: {write_error}\n"
                )

        # Action instructions
        self.stdout.write(f"\n--- {strategy_id or ''} actions ---\n")
        self.stdout.write(
            "[SIGNAL LAYER — signal_trades.json sim positions, "
            "NOT live Futu portfolio. Live positions managed by place_tp_sl.py.]\n"
        )
        self.stdout.write(
            "TP/SL: adaptive (will be computed in next day's "
            "compute_adaptive_tp_sl after entry)\n"
        )
        if max_positions_display is not None:
            self.stdout.write(
                f"No new positions can be opened when there are more than {max_positions_display}.\n"
            )
        if min_hold_bars > 0:
            self.stdout.write(f"Minimum hold: {min_hold_bars} bars\n")
        self.stdout.write("Entry priority is based on the displayed order.\n")
        if entry_signal_list:
            entry_names = ", ".join(f"'{s}'" for s in entry_signal_list)
            self.stdout.write(f"  BUY  {entry_names}\n")
        if held_exit_signals:
            exit_names = ", ".join(f"'{s}'" for s in held_exit_signals)
            self.stdout.write(f"  SELL {exit_names}\n")
        if held_min_hold_blocked:
            blocked_names = ", ".join(f"'{s}'" for s in held_min_hold_blocked)
            self.stdout.write(f"  HOLD (min_hold) {blocked_names}\n")
        if not held_exit_signals and not entry_signal_list:
            self.stdout.write("  (no action)\n")


    # TODO: review
    def help_find_history_signal(self) -> None:
        """Display help for the find_history_signal command."""
        self.stdout.write(
            "find_history_signal [DATE] DOLLAR_VOLUME_FILTER (BUY SELL STOP_LOSS | STOP_LOSS strategy=ID) [group=1,2,...]\n"
            "Display entry and exit signals for DATE or the latest trading day when DATE is omitted using the provided strategies or a strategy id from data/strategy_sets.csv.\n"
            "Signal calculation uses the same group dynamic ratio and Top-N rule as start_simulate.\n"
        )

    def do_multi_bucket_daily_signal(self, argument_line: str) -> None:  # noqa: D401
        """multi_bucket_daily_signal CONFIG_PATH [DATE] [--shadow]
        Production today-slice signal generator that reproduces the
        simulator's multi-bucket decision in one cron run. See
        help multi_bucket_daily_signal for details."""
        try:
            tokens = shlex.split(argument_line.strip())
        except ValueError as parse_error:
            self.stdout.write(f"failed to parse arguments: {parse_error}\n")
            return
        if not tokens:
            self.stdout.write(
                "usage: multi_bucket_daily_signal CONFIG_PATH [DATE] [--shadow]\n"
            )
            return

        config_path_text = tokens[0]
        date_string: str | None = None
        shadow_mode = False
        for token in tokens[1:]:
            if token == "--shadow":
                shadow_mode = True
            elif date_string is None:
                date_string = token
            else:
                self.stdout.write(f"unexpected argument: {token}\n")
                return

        config_path = Path(config_path_text).expanduser()
        try:
            config = multi_bucket_today.load_multi_bucket_config(config_path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as parse_error:
            self.stdout.write(f"{parse_error}\n")
            return

        if config.adaptive_tp_sl is None:
            self.stdout.write("config must define adaptive_tp_sl\n")
            return

        try:
            data_directory = resolve_data_source(config.data_source_name)
        except ValueError as source_error:
            self.stdout.write(f"{source_error}\n")
            return
        if not data_directory.exists():
            self.stdout.write(
                f"data source directory not found: {data_directory}\n"
            )
            return
        try:
            allowed_symbols = load_symbol_list(config.symbol_list_name)
        except ValueError as symbol_list_error:
            self.stdout.write(f"{symbol_list_error}\n")
            return
        try:
            ff12_data_path = resolve_ff12_data_path(config.ff12_data_path_text)
        except ValueError as ff12_path_error:
            self.stdout.write(f"{ff12_path_error}\n")
            return
        if ff12_data_path is not None:
            self.stdout.write(f"FF12 data: {ff12_data_path}\n")

        try:
            seasoning_dates_result = load_symbol_seasoning_dates_for_config(
                config.symbol_seasoning or symbol_seasoning.SymbolSeasoningConfig(),
                data_directory=data_directory,
                allowed_symbols=allowed_symbols,
            )
        except (FileNotFoundError, ValueError) as seasoning_error:
            self.stdout.write(f"{seasoning_error}\n")
            return
        symbol_first_eligible_trade_dates = None
        if seasoning_dates_result is not None:
            seasoning_source_path, symbol_first_eligible_trade_dates = (
                seasoning_dates_result
            )
            seasoning_config = (
                config.symbol_seasoning or symbol_seasoning.SymbolSeasoningConfig()
            )
            self.stdout.write(
                "Symbol seasoning: enabled "
                f"records={len(symbol_first_eligible_trade_dates)} "
                f"source={seasoning_config.eligibility_source} "
                f"path={seasoning_source_path}\n"
            )

        if date_string is None:
            eval_date_string = daily_job.determine_latest_trading_date().isoformat()
        else:
            try:
                datetime.date.fromisoformat(date_string)
            except ValueError:
                self.stdout.write(
                    f"invalid date: {date_string} (expected YYYY-MM-DD)\n"
                )
                return
            eval_date_string = date_string
        eval_date_timestamp = pandas.Timestamp(eval_date_string)
        evaluation_month = eval_date_timestamp.strftime("%Y-%m")

        try:
            daily_priority_override = (
                apply_risk_score_priority_override_for_month(
                    config,
                    evaluation_month,
                )
            )
        except ValueError as priority_error:
            self.stdout.write(f"{priority_error}\n")
            return
        if daily_priority_override is not None:
            target_priority_scores, priority_by_bucket_label = (
                daily_priority_override
            )
            if priority_by_bucket_label:
                score_text = ", ".join(
                    str(risk_score)
                    for risk_score in sorted(target_priority_scores)
                )
                priority_text = ", ".join(
                    f"{bucket_label}->{priority_value}"
                    for bucket_label, priority_value in sorted(
                        priority_by_bucket_label.items()
                    )
                )
                self.stdout.write(
                    "Risk-score priority override active: "
                    f"month={evaluation_month} scores=[{score_text}], "
                    f"{priority_text}\n"
                )

        suffix = "_shadow" if shadow_mode else ""
        state_path = LIVE_STATE_DIRECTORY / f"adaptive_state{suffix}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        state = multi_bucket_today.load_state(state_path)

        # Held positions come from cron's own virtual ledger
        # (state.accepted_entries), NOT from signal_trades.json.
        # signal_trades.json belongs to the order layer (dashboard) and
        # tracks real Futu fills — a separate concern from the rolling
        # pool's virtual-trade simulation. Coupling them caused two
        # bugs:
        #   1. Recorded-but-unfilled signals suppressed re-emission.
        #   2. Real positions Cal manually opens would never appear in
        #      the virtual ledger, and vice versa, leaking into either
        #      side.
        # After this split, cron only reads/writes adaptive_state.
        held_positions: Dict[str, List[Dict[str, str]]] = {}
        for accepted_entry in state.get("accepted_entries", []):
            strategy_identifier = accepted_entry.get("strategy_id", "")
            if not strategy_identifier:
                continue
            held_position_record = {
                "symbol": accepted_entry.get("symbol", ""),
                "entry_date": accepted_entry.get("entry_date", ""),
            }
            # Bucket attribution keeps buckets that share a strategy_id
            # (fish_tail_squeeze / fish_tail_production) separable in
            # compute_today_signals' held-position handling.
            if accepted_entry.get("bucket"):
                held_position_record["bucket"] = accepted_entry["bucket"]
            held_positions.setdefault(strategy_identifier, []).append(
                held_position_record
            )

        try:
            with strategy.override_ff12_group_source_path(ff12_data_path):
                result = multi_bucket_today.compute_today_signals(
                    config=config,
                    eval_date=eval_date_timestamp,
                    held_positions=held_positions,
                    state=state,
                    data_directory=data_directory,
                    allowed_symbols=allowed_symbols,
                    symbol_first_eligible_trade_dates=(
                        symbol_first_eligible_trade_dates
                    ),
                )
        except ValueError as run_error:
            self.stdout.write(f"compute_today_signals failed: {run_error}\n")
            return

        multi_bucket_today.save_state_atomically(state_path, state)

        self.stdout.write(
            f"[multi_bucket_daily_signal mode="
            f"{'shadow' if shadow_mode else 'live'} "
            f"state={state_path.name}]\n"
        )
        for log_line in result.log_lines:
            self.stdout.write(f"{log_line}\n")

    def help_multi_bucket_daily_signal(self) -> None:
        """Display help for the multi_bucket_daily_signal command."""
        self.stdout.write(
            "multi_bucket_daily_signal CONFIG_PATH [DATE] [--shadow]\n"
            "Today-slice multi-bucket signal generator. Reproduces the\n"
            "simulator's single-day decision in production:\n"
            "  - per-bucket signal generation via compute_signals_for_date\n"
            "  - shared frozen TP/SL via compute_frozen_tp_sl_for_bucket\n"
            "  - cross-bucket slot competition (priority + dollar_volume)\n"
            "Reads/writes data/adaptive_state.json (schema_version=2);\n"
            "signal_trades.json is owned by the order layer (dashboard)\n"
            "and is no longer touched by cron. With --shadow, all I/O\n"
            "is suffixed _shadow so the live cron path is untouched. Emits\n"
            "[ENTRY_SIGNAL], [EXIT_SIGNAL], [FROZEN_TP_SL], and\n"
            "[ROLLING_TP_SL_STATE] log lines for the dashboard/order layer.\n"
            "Honors optional ff12_data_path from the JSON config so live "
            "selection uses the same sector map as the matching simulation.\n"
            "Honors optional risk_score_priority_overrides for the evaluated "
            "month.\n"
        )

    def do_compute_adaptive_tp_sl(self, argument_line: str) -> None:  # noqa: D401
        """compute_adaptive_tp_sl
        System A: compute adaptive TP/SL from rolling raw trade stats.

        Reads data/adaptive_state.json for the rolling window of raw signal
        trade profits (open-to-open, no TP/SL adjustment).  Computes current
        TP/SL percentages and writes them back to adaptive_state.json for
        System B (place_tp_sl.py) to read.

        Config: window=20, sigma=0.5, SL sensor=median rolling loss.
        """
        state_path = LIVE_STATE_DIRECTORY / "adaptive_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # Adaptive parameters (matching new design):
        # - sl_pct computed as robust rolling median loss indicator
        # - SL never actually placed in live (place_tp_sl.py skips placement)
        # - sl_pct still recorded in adaptive_state for diagnostic / future use
        window = 20
        sigma_mult = 0.5
        fixed_sl_cap = 1.0  # large enough to never bind in practice
        min_tp = 0.02
        min_sl = 0.01
        min_samples = 5

        # Load rolling state
        raw_profits: list[float] = []
        closed_trades: list[dict] = []
        if state_path.exists():
            try:
                with state_path.open("r", encoding="utf-8") as fp:
                    state = json.load(fp)
                raw_profits = state.get("raw_trade_profits", [])
                closed_trades = state.get("closed_trades", [])
            except (json.JSONDecodeError, OSError):
                pass

        # Process closed trades: auto-fill entry/exit prices from stock
        # data (open price on T+1 for entry, open price on exit_date for
        # exit — matching simulation's open-to-open convention), compute
        # raw_pct and move into raw_trade_profits.
        stock_data_dir = DATA_DIRECTORY / "stock_data"
        updated_closed: list[dict] = []
        state_changed = False

        def _read_open_price(symbol: str, date_str: str) -> float | None:
            csv_path = stock_data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                return None
            try:
                price_frame = pandas.read_csv(
                    csv_path, index_col=0, parse_dates=True
                )
                ts = pandas.Timestamp(date_str)
                if ts not in price_frame.index:
                    return None
                open_col = next(
                    (c for c in price_frame.columns if c.lower() == "open"),
                    None,
                )
                if open_col is None:
                    return None
                val = float(price_frame.loc[ts, open_col])
                return val if not pandas.isna(val) else None
            except Exception:  # noqa: BLE001
                return None

        for ct in closed_trades:
            if ct.get("raw_pct") is not None:
                updated_closed.append(ct)
                continue
            symbol = ct.get("symbol", "")
            # Auto-fill entry_price: open on T+1 (one bday after entry_date).
            if ct.get("entry_price") is None and ct.get("entry_date"):
                t1 = (
                    pandas.Timestamp(ct["entry_date"]) + pandas.offsets.BDay(1)
                ).date().isoformat()
                entry_p = _read_open_price(symbol, t1)
                if entry_p is not None:
                    ct["entry_price"] = round(entry_p, 4)
            # Auto-fill exit_price: open on exit_date.
            if ct.get("exit_price") is None and ct.get("exit_date"):
                exit_p = _read_open_price(symbol, ct["exit_date"])
                if exit_p is not None:
                    ct["exit_price"] = round(exit_p, 4)
            # Compute raw_pct if both prices available.
            if ct.get("exit_price") is not None and ct.get("entry_price"):
                entry_p = float(ct["entry_price"])
                exit_p = float(ct["exit_price"])
                if entry_p > 0:
                    pct = (exit_p - entry_p) / entry_p
                    ct["raw_pct"] = round(pct, 6)
                    raw_profits.append(pct)
                    state_changed = True
                    self.stdout.write(
                        f"  Processed closed trade: {symbol} "
                        f"pct={pct:+.2%}\n"
                    )
            updated_closed.append(ct)
        # Trim rolling window
        raw_profits = raw_profits[-window:]
        # Compute current TP/SL
        tp_pct = min_tp
        sl_pct = min_sl
        if len(raw_profits) >= min_samples:
            from statistics import median as _median
            from statistics import stdev as _stdev

            wins = [
                profit_percentage
                for profit_percentage in raw_profits
                if profit_percentage > 0
            ]
            if len(wins) >= 3:
                mean_profit_percentage = sum(wins) / len(wins)
                profit_standard_deviation = (
                    _stdev(wins) if len(wins) >= 2 else 0.0
                )
                tp_pct = max(
                    min_tp,
                    mean_profit_percentage
                    + sigma_mult * profit_standard_deviation,
                )

            losses = [
                abs(loss_percentage)
                for loss_percentage in raw_profits
                if loss_percentage < 0
            ]
            if len(losses) >= 3:
                sl_pct = max(min_sl, _median(losses))
                sl_pct = min(sl_pct, fixed_sl_cap)

        # Always write back: rolling history, closed trades, and TP/SL %.
        # Merge into existing state rather than overwriting so multi_bucket
        # schema_version=2 fields (winners/losers/pending_rolling) survive
        # this call when multi_bucket_daily_signal runs in the same cron.
        merged_state: dict = {}
        if state_path.exists():
            try:
                with state_path.open("r", encoding="utf-8") as fp_read:
                    merged_state = json.load(fp_read)
                if not isinstance(merged_state, dict):
                    merged_state = {}
            except (json.JSONDecodeError, OSError):
                merged_state = {}
        merged_state["raw_trade_profits"] = [round(p, 8) for p in raw_profits]
        merged_state["closed_trades"] = updated_closed
        merged_state["tp_pct"] = round(tp_pct, 6)
        merged_state["sl_pct"] = round(sl_pct, 6)
        try:
            with state_path.open("w", encoding="utf-8") as fp:
                json.dump(merged_state, fp, indent=2)
        except OSError:
            pass

        self.stdout.write(
            f"\n--- Adaptive TP/SL (window={window}, "
            f"samples={len(raw_profits)}) ---\n"
        )
        self.stdout.write(f"  TP: {tp_pct:.2%}\n")
        self.stdout.write(f"  SL: {sl_pct:.2%}\n")
        if len(raw_profits) >= min_samples:
            wins = [p for p in raw_profits if p > 0]
            losses = [p for p in raw_profits if p <= 0]
            if wins:
                self.stdout.write(
                    f"  Rolling MP: {sum(wins)/len(wins):.2%} "
                    f"(n={len(wins)})\n"
                )
            if losses:
                self.stdout.write(
                    f"  Rolling ML: {sum(losses)/len(losses):.2%} "
                    f"(n={len(losses)})\n"
                )
        else:
            self.stdout.write(
                f"  (insufficient data: {len(raw_profits)}/{min_samples} "
                f"samples, using defaults)\n"
            )

    def do_show_positions(self, argument_line: str) -> None:  # noqa: D401
        """show_positions
        Print active signal trades from signal_trades.json (not live portfolio)."""
        positions_path = LIVE_STATE_DIRECTORY / "signal_trades.json"
        all_positions: Dict[str, list] = {}
        if positions_path.exists():
            try:
                with positions_path.open("r", encoding="utf-8") as fp:
                    all_positions = json.load(fp)
            except (json.JSONDecodeError, OSError):
                all_positions = {}
        total_count = sum(len(v) for v in all_positions.values())
        self.stdout.write(
            f"\n--- Concurrent positions after entry ({total_count} total) ---\n"
        )
        self.stdout.write(
            "[SIGNAL LAYER — signal_trades.json sim positions used for rolling "
            "TP/SL computation. NOT live Futu portfolio. Live positions queried "
            "from Futu API by place_tp_sl.py — they may not match this list.]\n"
        )
        for strat_id, strat_positions in all_positions.items():
            if strat_positions:
                symbols = []
                for item in strat_positions:
                    if isinstance(item, dict):
                        symbols.append(item.get("symbol", "?"))
                    else:
                        symbols.append(str(item))
                names = ", ".join(f"'{s}'" for s in symbols)
                self.stdout.write(f"  {strat_id}: {names}\n")
            else:
                self.stdout.write(f"  {strat_id}: (none)\n")



    def do_exit(self, argument_line: str) -> bool:  # noqa: D401
        """exit
        Exit the shell."""
        self.stdout.write("Bye\n")
        return True

    # TODO: review
    def help_exit(self) -> None:
        """Display help for the exit command."""
        self.stdout.write("exit\nExit the shell.\n")

    # TODO: review
    def do_EOF(self, arg: str) -> bool:
        """Exit the shell when an end-of-file (EOF) condition is reached."""
        self.stdout.write("Bye\n")
        return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )  # TODO: review
    if sys.argv[1:]:  # TODO: review
        command_text = " ".join(sys.argv[1:])  # TODO: review
        StockShell().onecmd(command_text)  # TODO: review
    else:  # TODO: review
        StockShell().cmdloop()
