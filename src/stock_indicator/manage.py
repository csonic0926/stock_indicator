"""Interactive shell for managing symbol cache and historical data."""

# TODO: review

from __future__ import annotations

import cmd
import datetime
import gc  # TODO: review
import logging
import re
import sys  # TODO: review
import time  # TODO: review
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import pandas
import yfinance  # TODO: review
from pandas import DataFrame

from . import data_loader, symbols, strategy, daily_job
from .strategy_sets import load_strategy_set_mapping
from .daily_job import determine_start_date
from .symbols import SP500_SYMBOL
from stock_indicator.sector_pipeline.overrides import (
    assign_symbol_to_other_if_missing,
)
from stock_indicator.sector_pipeline import pipeline

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
# Store downloaded per-symbol CSVs under a dedicated subfolder to avoid mixing
# with other project CSVs (e.g., sector exports).
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"


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
                    open_metrics = (
                        f" price_score={price_score_text}"
                        f" near_pct={near_ratio_text}"
                        f" above_pct={above_ratio_text}"
                        f" node_count={node_count_text}"
                    )
                line = (
                    f"  {trade_detail.date.date()} "
                    f"({trade_detail.concurrent_position_count}) "
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
        Download the latest list of ticker symbols."""
        symbols.update_symbol_cache()
        self.stdout.write("Symbol cache updated\n")

    # TODO: review
    def help_update_symbols(self) -> None:
        """Display help for the update_symbols command."""
        self.stdout.write(
            "update_symbols\n"
            "Download the latest list of ticker symbols.\n"
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
        exclusive_end_date = (
            datetime.date.fromisoformat(end_date) + datetime.timedelta(days=1)
        ).isoformat()
        symbol_list = symbols.load_symbols()
        # Ensure ^GSPC is also downloaded, but not stored in symbols.txt
        if SP500_SYMBOL not in symbol_list:
            symbol_list.append(SP500_SYMBOL)
        for symbol_name in symbol_list:
            data_frame: DataFrame = data_loader.download_history(
                symbol_name, start_date, exclusive_end_date
            )
            _cleanup_yfinance_session()  # TODO: review
            data_frame_with_date: DataFrame = (
                data_frame.reset_index().rename(columns={"index": "Date"})
            )
            STOCK_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
            output_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
            try:
                with output_path.open("w", encoding="utf-8") as file_handle:  # TODO: review
                    data_frame_with_date.to_csv(file_handle, index=False)  # TODO: review
            except OSError as first_error:  # TODO: review
                LOGGER.error(
                    "Error writing CSV for %s: %s", symbol_name, first_error
                )  # TODO: review
                time.sleep(1)  # TODO: review
                try:
                    with output_path.open("w", encoding="utf-8") as file_handle:  # TODO: review
                        data_frame_with_date.to_csv(file_handle, index=False)  # TODO: review
                except OSError as second_error:  # TODO: review
                    LOGGER.error(
                        "Skipping %s due to repeated write failure: %s",
                        symbol_name,
                        second_error,
                    )  # TODO: review
                    continue
            self.stdout.write(f"Data written to {output_path}\n")

    def help_update_all_data_from_yf(self) -> None:
        """Display help for the update_all_data_from_yf command."""
        self.stdout.write(
            "update_all_data_from_yf START END\n"
            "Download data from Yahoo Finance for all cached symbols and write CSVs to data/stock_data/.\n"
            "Parameters:\n"
            "  START: Start date in YYYY-MM-DD format.\n"
            "  END: End date in YYYY-MM-DD format (inclusive).\n"
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
            "The ticker universe is sourced from the SEC company tickers dataset.\n"
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
        if len(argument_parts) == 1 and argument_parts[0].startswith("strategy="):
            strategy_identifier = argument_parts[0].split("=", 1)[1].strip()
        elif len(argument_parts) == 2:
            buy_strategy_name, sell_strategy_name = argument_parts
        else:
            self.stdout.write(usage_message)
            return  # TODO: review
        if strategy_identifier:
            mapping = load_strategy_set_mapping()
            if strategy_identifier not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_identifier}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_identifier]
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
        while (
            argument_parts
            and argument_parts[0] != "--"
            and argument_parts[0].startswith(
                ("starting_cash=", "withdraw=", "start=", "margin=")
            )
        ):
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
            if strategy_identifier:
                if strategy_identifier not in strategy_mapping:
                    raise ValueError(
                        f"unknown strategy id: {strategy_identifier}"
                    )
                buy_strategy_name, sell_strategy_name = strategy_mapping[
                    strategy_identifier
                ]
                if len(remaining_tokens) > 1:
                    raise ValueError("invalid stop loss")
                if remaining_tokens:
                    try:
                        stop_loss_percentage = float(remaining_tokens[0])
                    except ValueError as error:
                        raise ValueError("invalid stop loss") from error
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
                    if len(remaining_tokens) > 1:
                        raise ValueError("invalid stop loss")
                    try:
                        stop_loss_percentage = float(remaining_tokens[0])
                    except ValueError as error:
                        raise ValueError("invalid stop loss") from error

            return strategy.ComplexStrategySetDefinition(
                label=label,
                buy_strategy_name=buy_strategy_name,
                sell_strategy_name=sell_strategy_name,
                stop_loss_percentage=stop_loss_percentage,
                minimum_average_dollar_volume=minimum_average_dollar_volume,
                minimum_average_dollar_volume_ratio=
                    minimum_average_dollar_volume_ratio,
                top_dollar_volume_rank=top_dollar_volume_rank,
                maximum_symbols_per_group=maximum_symbols_per_group,
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

        data_directory = (
            STOCK_DATA_DIRECTORY
            if STOCK_DATA_DIRECTORY.exists()
            else DATA_DIRECTORY
        )

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
                open_metrics = (
                    f" price_score={price_score_text}"
                    f" near_pct={near_ratio_text}"
                    f" above_pct={above_ratio_text}"
                    f" node_count={node_count_text}"
            )
            return (
                f"{detail.date.date()} ({detail.concurrent_position_count}) "
                f"{detail.symbol} {detail.action} {detail.price:.2f} "
                f"{detail.group_simple_moving_average_dollar_volume_ratio:.4f} "
                f"{detail.simple_moving_average_dollar_volume / 1_000_000:.2f}M "
                f"{detail.group_total_simple_moving_average_dollar_volume / 1_000_000:.2f}M"
                f"{open_metrics}{result_suffix}"
            )

        total_metrics = simulation_metrics.overall_metrics
        self.stdout.write(format_summary_line("Total", total_metrics))

        for set_label in ("A", "B"):
            metrics = simulation_metrics.metrics_by_set.get(set_label)
            if metrics is None:
                continue
            self.stdout.write(format_summary_line(set_label, metrics))
            for year, annual_return in sorted(metrics.annual_returns.items()):
                trade_count = metrics.annual_trade_counts.get(year, 0)
                annual_profit = metrics.annual_profit_totals.get(year, 0.0)
                self.stdout.write(
                    f"[{set_label}] Year {year}: {annual_return:.2%}, "
                    f"profit: {annual_profit:.2f}, trade: {trade_count}\n"
                )
                if show_trade_details:
                    trade_details = metrics.trade_details_by_year.get(year, [])
                    for trade_detail in trade_details:
                        formatted_detail = format_trade_detail(trade_detail)
                        self.stdout.write(
                            f"[{set_label}]   {formatted_detail}\n"
                        )

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
            "optionally followed by a stop-loss value. Separate the two sets with --.\n"
        )

    # TODO: review
    def do_start_simulate(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [SHOW_DETAILS]
        Evaluate trading strategies using cached data.

        STOP_LOSS defaults to 1.0 when not provided.
        SHOW_DETAILS defaults to True and controls whether trade details are printed."""
        argument_parts: List[str] = argument_line.split()
        starting_cash_value = 3000.0
        withdraw_amount = 0.0
        start_date_string: str | None = None
        allowed_group_identifiers: set[int] | None = None
        margin_multiplier: float = 1.0
        strategy_id: str | None = None
        while argument_parts and (
            argument_parts[0].startswith("starting_cash=")
            or argument_parts[0].startswith("withdraw=")
            or argument_parts[0].startswith("start=")
            or argument_parts[0].startswith("group=")
            or argument_parts[0].startswith("margin=")
            or argument_parts[0].startswith("strategy=")
        ):
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
            if name == "group":
                try:
                    parsed_values = [segment.strip() for segment in value.split(",") if segment.strip()]
                    parsed_integers = {int(segment) for segment in parsed_values}
                except ValueError:
                    self.stdout.write("invalid group list\n")
                    return
                # Disallow 12 (Other); all identifiers must be between 1 and 11.
                if any(identifier < 1 or identifier > 11 for identifier in parsed_integers):
                    self.stdout.write("group identifiers must be between 1 and 11\n")
                    return
                if 12 in parsed_integers:
                    self.stdout.write("group list must not include 12 (Other)\n")
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
                if any(identifier < 1 or identifier > 11 for identifier in parsed_integers):
                    self.stdout.write("group identifiers must be between 1 and 11\n")
                    return
                if 12 in parsed_integers:
                    self.stdout.write("group list must not include 12 (Other)\n")
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
            post_scan_index += 1
        # Two forms supported:
        # - FILTER BUY SELL [STOP] [SHOW]
        # - FILTER [STOP] [SHOW] with strategy=ID
        stop_loss_percentage = 1.0
        show_trade_details = True
        if strategy_id:
            if len(argument_parts) not in (1, 2, 3):
                self.stdout.write(
                    "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] DOLLAR_VOLUME_FILTER [STOP_LOSS] [SHOW_DETAILS] strategy=ID [group=...] [margin=NUMBER]\n"
                )
                return
            volume_filter = argument_parts[0]
            if len(argument_parts) >= 2:
                try:
                    stop_loss_percentage = float(argument_parts[1])
                except ValueError:
                    self.stdout.write("invalid stop loss\n")
                    return
            if len(argument_parts) == 3:
                show_trade_details = argument_parts[2].lower() == "true"
            mapping = load_strategy_set_mapping()
            if strategy_id not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_id}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_id]
            # Pass through composite strategy expressions; OR resolution handled in strategy layer
            # Pass through composite strategy expressions
            # Pass through composite strategy expressions
        else:
            if len(argument_parts) not in (3, 4, 5):
                self.stdout.write(
                    "usage: start_simulate [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [margin=NUMBER] "
                    "DOLLAR_VOLUME_FILTER (BUY SELL | [strategy=ID]) [STOP_LOSS] [SHOW_DETAILS] [group=1,2,...]\n"
                )
                return
            volume_filter, buy_strategy_name, sell_strategy_name = argument_parts[:3]
            if len(argument_parts) >= 4:
                try:
                    stop_loss_percentage = float(argument_parts[3])
                except ValueError:
                    self.stdout.write("invalid stop loss\n")
                    return
            if len(argument_parts) == 5:
                show_trade_details = argument_parts[4].lower() == "true"
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
        evaluation_metrics = strategy.evaluate_combined_strategy(
            STOCK_DATA_DIRECTORY if STOCK_DATA_DIRECTORY.exists() else DATA_DIRECTORY,
            buy_strategy_name,
            sell_strategy_name,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            top_dollar_volume_rank=top_dollar_volume_rank,
            minimum_average_dollar_volume_ratio=minimum_average_dollar_volume_ratio,
            starting_cash=starting_cash_value,
            withdraw_amount=withdraw_amount,
            stop_loss_percentage=stop_loss_percentage,
            start_date=start_timestamp,
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
            evaluation_metrics.annual_profit_totals = {
                year: evaluation_metrics.annual_profit_totals.get(year, 0.0)
                for year in filtered_trade_details_by_year
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
                        "concurrent_position_index": entry_detail.concurrent_position_count,
                        "symbol": entry_detail.symbol,
                        "price_concentration_score": entry_detail.price_concentration_score,
                        "near_price_volume_ratio": entry_detail.near_price_volume_ratio,
                        "above_price_volume_ratio": entry_detail.above_price_volume_ratio,
                        "histogram_node_count": entry_detail.histogram_node_count,
                        "sma_angle": entry_detail.sma_angle,
                        "exit_date": detail.date.date(),
                        "result": detail.result,
                        "percentage_change": detail.percentage_change,
                        "exit_reason": detail.exit_reason,
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
                "histogram_node_count",
                "sma_angle",
                "exit_date",
                "result",
                "percentage_change",
                "exit_reason",
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
            annual_profit = evaluation_metrics.annual_profit_totals.get(
                year, 0.0
            )
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, "
                f"profit: {annual_profit:.2f}, trade: {trade_count}\n"
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
            "DOLLAR_VOLUME_FILTER (BUY SELL | [strategy=ID]) [STOP_LOSS] [SHOW_DETAILS] [group=1,2,...]\n"
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
            "    backward compatibility. 'Other' (FF12=12) is excluded.\n"
            "  BUY/SELL or strategy=ID: Either provide explicit buy/sell strategy names, "
            "or a strategy id defined in data/strategy_sets.csv.\n"
            "  STOP_LOSS: Fractional loss for stop orders. If intraday low hits\n"
            "    the stop, exits on the same bar at the stop price; otherwise, if\n"
            "    the close is below the stop, exits on the next day's open.\n"
            "    Defaults to 1.0 (disabled).\n"
            "  SHOW_DETAILS: 'True' to print individual trades, 'False' to suppress them. Defaults to True.\n"
            "  group: Optional comma-separated FF12 group ids (1-11) to restrict\n"
            "    tradable symbols. Group 12 (Other) is always excluded. Example:\n"
            "    group=1,2,4,6,7,8,10,11\n"
            "Strategies may be suffixed with _N to set the window size to N; the default window size is 40 when no suffix is provided.\n"
            "Slope-aware strategies follow the ema_sma_signal_with_slope_n_k pattern and accept _LOWER_UPPER bounds after the optional window size; both bounds are floating-point numbers and may be negative.\n"
            "Example: start_simulate start=1990-01-01 dollar_volume>50 ema_sma_cross_20 ema_sma_cross_20\n"
            "Another example: start_simulate dollar_volume>1 ema_sma_signal_with_slope_-0.1_1.2 ema_sma_signal_with_slope_-0.1_1.2\n"
            f"Available buy strategies: {available_buy}.\n"
            f"Available sell strategies: {available_sell}.\n"
        )

    
    def do_start_simulate_single_symbol(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [SHOW_DETAILS]
        Evaluate strategies on a single symbol using full allocation per position.

        When not provided, STOP_LOSS defaults to 1.0 and SHOW_DETAILS defaults to True.
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
        show_trade_details = True
        if strategy_id:
            if len(argument_parts) not in (1, 2):
                self.stdout.write(
                    "usage: start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [STOP_LOSS] [SHOW_DETAILS] strategy=ID\n"
                )
                return
            if len(argument_parts) >= 1:
                try:
                    stop_loss_percentage = float(argument_parts[0])
                except ValueError:
                    self.stdout.write("invalid stop loss\n")
                    return
            if len(argument_parts) == 2:
                show_trade_details = argument_parts[1].lower() == "true"
            mapping = load_strategy_set_mapping()
            if strategy_id not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_id}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_id]
            # Pass through composite strategy expressions
        else:
            if len(argument_parts) not in (2, 3, 4):
                self.stdout.write(
                    "usage: start_simulate_single_symbol [symbol=SYMBOL] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
                    "(BUY SELL | [strategy=ID]) [STOP_LOSS] [SHOW_DETAILS]\n"
                )
                return
            buy_strategy_name, sell_strategy_name = argument_parts[:2]
            if len(argument_parts) >= 3:
                try:
                    stop_loss_percentage = float(argument_parts[2])
                except ValueError:
                    self.stdout.write("invalid stop loss\n")
                    return
            if len(argument_parts) == 4:
                show_trade_details = argument_parts[3].lower() == "true"
        # Validate strategies; support composite expressions
        if not _has_supported_strategy(buy_strategy_name, strategy.BUY_STRATEGIES) or not _has_supported_strategy(
            sell_strategy_name, strategy.SELL_STRATEGIES
        ):
            self.stdout.write("unsupported strategies\n")
            return

        # Determine data directory and ensure the symbol file exists
        base_directory = STOCK_DATA_DIRECTORY if STOCK_DATA_DIRECTORY.exists() else DATA_DIRECTORY
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
            annual_profit = evaluation_metrics.annual_profit_totals.get(
                year, 0.0
            )
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, "
                f"profit: {annual_profit:.2f}, trade: {trade_count}\n"
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
            "(BUY SELL | [strategy=ID]) [STOP_LOSS] [SHOW_DETAILS]\n"
            "Simulate strategies on a single symbol using full allocation per position.\n"
            "Parameters:\n"
            "  symbol: Ticker symbol to simulate (required).\n"
            "  starting_cash: Initial cash balance. Defaults to 3000.\n"
            "  withdraw: Amount deducted from cash at each year end. Defaults to 0.\n"
            "  start: Start date in YYYY-MM-DD format. Defaults to earliest available.\n"
            "  BUY/SELL or strategy=ID: Either provide explicit strategy names, or a strategy id defined in data/strategy_sets.csv.\n"
            "  STOP_LOSS: Fractional loss for stop orders (same-bar at stop price when low hits; otherwise next-day open). Defaults to 1.0.\n"
            "  SHOW_DETAILS: 'True' to print trade details, 'False' to suppress. Defaults to True.\n"
        )

    def do_start_simulate_n_symbol(self, argument_line: str) -> None:  # noqa: D401
        """start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] BUY_STRATEGY SELL_STRATEGY [STOP_LOSS] [SHOW_DETAILS]
        Evaluate strategies across a provided symbol list. Budget per position uses slot count equal to the number of symbols.

        When not provided, STOP_LOSS defaults to 1.0 and SHOW_DETAILS defaults to True.
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
                "(BUY SELL | [strategy=ID]) [STOP_LOSS] [SHOW_DETAILS]\n"
            )
            return
        stop_loss_percentage = 1.0
        show_trade_details = True
        if strategy_id:
            if len(argument_parts) not in (1, 2):
                self.stdout.write(
                    "usage: start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] [STOP_LOSS] [SHOW_DETAILS] strategy=ID\n"
                )
                return
            if len(argument_parts) >= 1:
                try:
                    stop_loss_percentage = float(argument_parts[0])
                except ValueError:
                    self.stdout.write("invalid stop loss\n")
                    return
            if len(argument_parts) == 2:
                show_trade_details = argument_parts[1].lower() == "true"
            mapping = load_strategy_set_mapping()
            if strategy_id not in mapping:
                self.stdout.write(f"unknown strategy id: {strategy_id}\n")
                return
            buy_strategy_name, sell_strategy_name = mapping[strategy_id]
        else:
            if len(argument_parts) not in (2, 3, 4):
                self.stdout.write(
                    "usage: start_simulate_n_symbol symbols=AAA,BBB[,CCC...] [starting_cash=NUMBER] [withdraw=NUMBER] [start=YYYY-MM-DD] "
                    "(BUY SELL | [strategy=ID]) [STOP_LOSS] [SHOW_DETAILS]\n"
                )
                return
            buy_strategy_name, sell_strategy_name = argument_parts[:2]
            if len(argument_parts) >= 3:
                try:
                    stop_loss_percentage = float(argument_parts[2])
                except ValueError:
                    self.stdout.write("invalid stop loss\n")
                    return
            if len(argument_parts) == 4:
                show_trade_details = argument_parts[3].lower() == "true"

        # Validate strategies; support composite expressions
        if not _has_supported_strategy(buy_strategy_name, strategy.BUY_STRATEGIES) or not _has_supported_strategy(
            sell_strategy_name, strategy.SELL_STRATEGIES
        ):
            self.stdout.write("unsupported strategies\n")
            return

        base_directory = STOCK_DATA_DIRECTORY if STOCK_DATA_DIRECTORY.exists() else DATA_DIRECTORY
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
            annual_profit = evaluation_metrics.annual_profit_totals.get(
                year, 0.0
            )
            self.stdout.write(
                f"Year {year}: {annual_return:.2%}, "
                f"profit: {annual_profit:.2f}, trade: {trade_count}\n"
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
            "(BUY SELL | [strategy=ID]) [STOP_LOSS] [SHOW_DETAILS]\n"
            "Simulate strategies across a list of symbols; budget per position uses slots equal to the list length.\n"
            "Parameters:\n"
            "  symbols: Comma-separated ticker symbols to simulate (required).\n"
            "  starting_cash: Initial cash balance. Defaults to 3000.\n"
            "  withdraw: Amount deducted from cash at each year end. Defaults to 0.\n"
            "  start: Start date in YYYY-MM-DD format. Defaults to earliest available.\n"
            "  BUY/SELL or strategy=ID: Either provide explicit strategy names, or a strategy id defined in data/strategy_sets.csv.\n"
            "  STOP_LOSS: Fractional loss for stop orders (same-bar at stop price when low hits; otherwise next-day open). Defaults to 1.0.\n"
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
        for token in argument_parts:
            if token.startswith("group="):
                try:
                    raw = token.split("=", 1)[1]
                    parts = [p.strip() for p in raw.split(",") if p.strip()]
                    parsed = {int(p) for p in parts}
                except ValueError:
                    self.stdout.write("invalid group list\n")
                    return
                if any(identifier < 1 or identifier > 11 for identifier in parsed):
                    self.stdout.write("group identifiers must be between 1 and 11\n")
                    return
                if 12 in parsed:
                    self.stdout.write("group list must not include 12 (Other)\n")
                    return
                allowed_group_identifiers = parsed
            elif token.startswith("strategy="):
                strategy_id = token.split("=", 1)[1].strip()
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
        signal_data: Dict[str, Any] = daily_job.find_history_signal(
            date_string,
            dollar_volume_filter,
            buy_strategy_name,
            sell_strategy_name,
            stop_loss_value,
            allowed_group_identifiers,
        )
        filtered_symbol_list: List[tuple[str, int | None]] = signal_data.get(
            "filtered_symbols", []
        )
        self.stdout.write(f"filtered symbols: {filtered_symbol_list}\n")
        entry_signal_list: List[str] = signal_data.get("entry_signals", [])
        exit_signal_list: List[str] = signal_data.get("exit_signals", [])
        self.stdout.write(f"entry signals: {entry_signal_list}\n")
        self.stdout.write(f"exit signals: {exit_signal_list}\n")
        entry_budgets: Dict[str, float] | None = signal_data.get("entry_budgets")
        if entry_budgets:
            self.stdout.write(f"budget suggestions: {entry_budgets}\n")

    # TODO: review
    def help_find_history_signal(self) -> None:
        """Display help for the find_history_signal command."""
        self.stdout.write(
            "find_history_signal [DATE] DOLLAR_VOLUME_FILTER (BUY SELL STOP_LOSS | STOP_LOSS strategy=ID) [group=1,2,...]\n"
            "Display entry and exit signals for DATE or the latest trading day when DATE is omitted using the provided strategies or a strategy id from data/strategy_sets.csv.\n"
            "Signal calculation uses the same group dynamic ratio and Top-N rule as start_simulate.\n"
        )



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
