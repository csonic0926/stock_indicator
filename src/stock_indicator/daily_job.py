"""Entry point for running the daily cron tasks."""
# TODO: review

from __future__ import annotations

import argparse
import json
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas
from pandas.tseries.offsets import BDay

from . import cron, strategy_sets
from .data_loader import download_history

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2019-01-01"
# Earliest date used when refreshing historical data; this guards against
# missing rows in local caches. Modify only when extending the supported
# history range.
MINIMUM_HISTORY_DATE = "2014-01-01"
DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
STOCK_DATA_DIRECTORY = DATA_DIRECTORY / "stock_data"
LOG_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"
LOCAL_DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "local_data"
CURRENT_STATUS_FILE = LOCAL_DATA_DIRECTORY / "current_status.json"


def determine_latest_trading_date(
    now: datetime.datetime | None = None,
) -> datetime.date:
    """Return the most recent trading date based on Eastern Time.

    Parameters
    ----------
    now:
        Current timestamp. When ``None`` the system time in the ``US/Eastern``
        timezone is used.

    Returns
    -------
    datetime.date
        The prior business day if the time is earlier than 16:00 Eastern,
        otherwise the current date.
    """

    eastern_zone = ZoneInfo("US/Eastern")
    if now is None:
        current_time = datetime.datetime.now(tz=eastern_zone)
    else:
        current_time = (
            now.astimezone(eastern_zone)
            if now.tzinfo is not None
            else now.replace(tzinfo=eastern_zone)
        )

    if current_time.time() < datetime.time(16, 0):
        previous_business_day = (
            pandas.Timestamp(current_time.date()) - BDay(1)
        ).date()
        return previous_business_day
    return current_time.date()


def determine_start_date(data_directory: Path) -> str:
    """Return the earliest date across all CSV files in ``data_directory``.

    When no CSV files are available, ``DEFAULT_START_DATE`` is returned.
    """
    earliest_date: datetime.date | None = None
    if not data_directory.exists():
        return DEFAULT_START_DATE
    for csv_file_path in data_directory.glob("*.csv"):
        try:
            date_frame = pandas.read_csv(
                csv_file_path, usecols=[0], parse_dates=[0]
            )
        except Exception as read_error:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)
            continue
        if date_frame.empty:
            continue
        column_minimum = date_frame.iloc[:, 0].min()
        if not hasattr(column_minimum, "date"):
            continue
        earliest_candidate = column_minimum.date()
        if earliest_date is None or earliest_candidate < earliest_date:
            earliest_date = earliest_candidate
    if earliest_date is None:
        return DEFAULT_START_DATE
    return earliest_date.isoformat()


def _expand_strategy_argument_line(argument_line: str) -> str:
    """Expand a ``strategy=ID`` token into explicit buy and sell names.

    The ``argument_line`` may optionally contain a ``group=`` token. When a
    ``strategy=`` token is present, its identifier is looked up in
    ``data/strategy_sets.csv`` via
    :func:`strategy_sets.load_strategy_set_mapping`. The token is replaced by
    the corresponding buy and sell strategy names while preserving the trailing
    stop-loss value. The ``group=`` token, when provided, is kept at the
    beginning of the returned string.

    Parameters
    ----------
    argument_line:
        Raw argument line possibly containing a ``strategy=`` token.

    Returns
    -------
    str
        Argument line with any ``strategy=ID`` token expanded.

    Raises
    ------
    ValueError
        If the provided strategy identifier is not found in the mapping.
    """

    parts = argument_line.split()
    group_token: str | None = None
    strategy_identifier: str | None = None
    remaining_tokens: list[str] = []
    for part in parts:
        if part.startswith("group=") and group_token is None:
            group_token = part
        elif part.startswith("strategy=") and strategy_identifier is None:
            strategy_identifier = part.split("=", 1)[1].strip()
        else:
            remaining_tokens.append(part)

    if strategy_identifier is None:
        tokens: list[str] = []
        if group_token is not None:
            tokens.append(group_token)
        tokens.extend(remaining_tokens)
        return " ".join(tokens)

    if not remaining_tokens:
        raise ValueError("missing dollar volume filter")
    volume_filter = remaining_tokens[0]
    stop_loss_token: str | None = remaining_tokens[1] if len(remaining_tokens) > 1 else None

    mapping = strategy_sets.load_strategy_set_mapping()
    if strategy_identifier not in mapping:
        raise ValueError(f"unknown strategy id: {strategy_identifier}")
    buy_name, sell_name = mapping[strategy_identifier]

    tokens = []
    if group_token is not None:
        tokens.append(group_token)
    tokens.extend([volume_filter, buy_name, sell_name])
    if stop_loss_token is not None:
        tokens.append(stop_loss_token)
    return " ".join(tokens)


def run_daily_job(
    argument_line: str,
    *,
    data_directory: Path | None = None,
    log_directory: Path | None = None,
    current_date: datetime.date | None = None,
) -> Path:
    """Execute daily tasks and record the signals to a log file.

    Parameters
    ----------
    argument_line: str
        Argument string in the format accepted by
        :func:`cron.run_daily_tasks_from_argument`.
    data_directory: Path | None, optional
        Directory where downloaded per-symbol CSVs are written. Defaults to
        ``data/stock_data`` under the project root.
    log_directory: Path | None, optional
        Directory where the log file is stored. Defaults to the ``logs``
        directory in the project root.
    current_date: datetime.date | None, optional
        Date used for data retrieval and log file naming. When ``None`` the
        system date is used.

    Returns
    -------
    Path
        Path to the log file containing the entry and exit signals.
    """
    global STOCK_DATA_DIRECTORY
    if current_date is None:
        current_date = datetime.date.today()
    # Track whether caller used defaults; only then add budget lines to logs
    used_default_paths = data_directory is None and log_directory is None
    if data_directory is None:
        data_directory = STOCK_DATA_DIRECTORY
    if log_directory is None:
        log_directory = LOG_DIRECTORY

    signal_date_string = current_date.isoformat()
    trade_date = (pandas.Timestamp(current_date) + BDay(1)).date()
    trade_date_string = trade_date.isoformat()
    LOGGER.info(
        "Starting daily tasks for trade date %s using signals from %s",
        trade_date_string,
        signal_date_string,
    )
    normalized_argument_line = _expand_strategy_argument_line(argument_line)

    token_list = normalized_argument_line.split()
    allowed_groups: set[int] | None = None
    if token_list and token_list[0].startswith("group="):
        group_values = token_list.pop(0).split("=", 1)[1]
        allowed_groups = {
            int(value) for value in group_values.split(",") if value
        }

    if len(token_list) < 3:
        raise ValueError(
            "argument line must include dollar volume filter, buy strategy, and sell strategy",
        )

    dollar_volume_filter = token_list[0]
    buy_strategy_name = token_list[1]
    sell_strategy_name = token_list[2]
    stop_loss_token = token_list[3] if len(token_list) > 3 else "1.0"
    try:
        stop_loss_value = float(stop_loss_token)
    except ValueError as stop_loss_error:
        raise ValueError(
            f"invalid stop loss: {stop_loss_token}",
        ) from stop_loss_error

    original_stock_directory = STOCK_DATA_DIRECTORY
    try:
        STOCK_DATA_DIRECTORY = data_directory
        signal_result: Dict[str, list[str]] = find_history_signal(
            signal_date_string,
            dollar_volume_filter,
            buy_strategy_name,
            sell_strategy_name,
            stop_loss_value,
            allowed_groups,
        )
    finally:
        STOCK_DATA_DIRECTORY = original_stock_directory
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file_path = log_directory / f"{signal_date_string}.log"
    entry_signals: List[str] = signal_result.get("entry_signals", [])
    exit_signals: List[str] = signal_result.get("exit_signals", [])

    # Compute budget per position using simulator sizing: budget = equity * min(margin/slots, 1.0)
    equity_value: float | None = None
    margin_multiplier: float | None = None
    slot_count: int | None = None
    slot_weight: float | None = None
    budget_per_entry: float | None = None
    budgets_by_symbol: Dict[str, float] | None = None
    if entry_signals and used_default_paths:
        try:
            portfolio_status = _load_portfolio_status(CURRENT_STATUS_FILE)
            equity_value, margin_multiplier, slot_count, slot_weight = _compute_sizing_inputs(
                portfolio_status, data_directory, current_date
            )
            if equity_value is not None and margin_multiplier is not None and slot_weight is not None:
                budget_per_entry = equity_value * slot_weight
                budgets_by_symbol = {symbol: budget_per_entry for symbol in entry_signals}
            else:
                LOGGER.warning(
                    "Budget calculation skipped (equity=%s, margin=%s, slot_weight=%s)",
                    equity_value,
                    margin_multiplier,
                    slot_weight,
                )
        except Exception as status_error:  # noqa: BLE001
            LOGGER.warning("Could not compute budgets from current_status.json: %s", status_error)

    with log_file_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"entry_signals: {', '.join(entry_signals)}\n")
        if budget_per_entry is not None and budgets_by_symbol is not None:
            if equity_value is not None:
                log_file.write(f"equity_usd: {equity_value:.2f}\n")
            if margin_multiplier is not None:
                log_file.write(f"margin_multiplier: {margin_multiplier:.2f}\n")
            if slot_count is not None:
                log_file.write(f"slot_count: {slot_count}\n")
            if slot_weight is not None:
                log_file.write(f"slot_weight: {slot_weight:.4f}\n")
            log_file.write(f"budget_per_entry_usd: {budget_per_entry:.2f}\n")
            formatted_budgets = ", ".join(
                f"{symbol}:{amount:.2f}" for symbol, amount in budgets_by_symbol.items()
            )
            log_file.write(f"entry_budgets: {formatted_budgets}\n")
        log_file.write(f"exit_signals: {', '.join(exit_signals)}\n")
    LOGGER.info("Daily tasks completed; results written to %s", log_file_path)
    return log_file_path


def _load_portfolio_status(status_file_path: Path) -> Dict[str, object]:
    """Load and validate the current portfolio status JSON.

    The expected minimal schema is::

        {
            "cash": <number>,
            "margin": <number>,
            "positions": [
                {"symbol": "SYM", "Qty": <number>}, ...
            ]
        }

    Only ``cash`` and ``margin`` are required for budget calculation. A missing
    ``positions`` list does not prevent budget computation.

    Parameters
    ----------
    status_file_path:
        Path to ``local_data/current_status.json``.

    Returns
    -------
    Dict[str, object]
        Parsed JSON content.

    Raises
    ------
    ValueError
        If the JSON cannot be parsed or required fields are invalid.
    """
    if not status_file_path.exists():
        raise ValueError(f"status file not found: {status_file_path}")
    try:
        raw_text = status_file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception as parse_error:  # noqa: BLE001
        raise ValueError(f"invalid JSON in {status_file_path}") from parse_error

    # Validate required fields
    if "cash" not in data:
        raise ValueError("missing 'cash' in current_status.json")
    if "margin" not in data:
        raise ValueError("missing 'margin' in current_status.json")
    try:
        float(data["cash"])  # ensure numeric
        float(data["margin"])  # ensure numeric
    except Exception as type_error:  # noqa: BLE001
        raise ValueError("'cash' and 'margin' must be numbers") from type_error
    return data


def _compute_sizing_inputs(
    portfolio_status: Dict[str, Any],
    data_directory: Path,
    valuation_date: datetime.date,
) -> Tuple[float | None, float | None, int | None, float | None]:
    """Compute equity, margin, slot count and slot weight for sizing.

    - equity: cash + sum(position_qty * latest_close)
    - margin: taken from current_status.json (default 1.0)
    - slots: prefer JSON ('maximum_position_count'|'max_concurrent_positions'|'slots');
      fallback to default 3 to match simulator defaults.
    - slot_weight: min(margin / slots, 1.0) to mirror simulator sizing.

    Returns a tuple (equity, margin, slots, slot_weight). Any None indicates
    inputs were insufficient to compute the value.
    """
    try:
        cash_balance = float(portfolio_status.get("cash", 0.0))
    except Exception:  # noqa: BLE001
        cash_balance = 0.0
    try:
        margin_multiplier = float(portfolio_status.get("margin", 1.0))
    except Exception:  # noqa: BLE001
        margin_multiplier = 1.0

    equity_value = cash_balance
    positions_list = portfolio_status.get("positions") or []
    if isinstance(positions_list, list):
        for position_item in positions_list:
            try:
                symbol_name = str(position_item.get("symbol"))
                quantity_value = position_item.get("Qty")
                if quantity_value is None:
                    quantity_value = position_item.get("qty")
                if quantity_value is None:
                    quantity_value = position_item.get("quantity")
                if not symbol_name or quantity_value is None:
                    continue
                position_size = float(quantity_value)
                latest_close = _read_latest_close(symbol_name, data_directory, valuation_date)
                if latest_close is None:
                    LOGGER.warning(
                        "No recent close for %s on or before %s; treating as 0",
                        symbol_name,
                        valuation_date,
                    )
                    continue
                equity_value += position_size * latest_close
            except Exception as position_error:  # noqa: BLE001
                LOGGER.warning("Skipping position due to error: %s", position_error)

    slot_count = _determine_slot_count(portfolio_status)
    if slot_count is None or slot_count <= 0:
        return equity_value, margin_multiplier, None, None

    slot_weight = min(margin_multiplier / slot_count, 1.0)
    return equity_value, margin_multiplier, slot_count, slot_weight


def _determine_slot_count(portfolio_status: Dict[str, Any]) -> int | None:
    """Determine the slot count from status JSON using common keys.

    Recognized keys (first found wins):
    - maximum_position_count
    - max_concurrent_positions
    - slots
    Fallback: 3 (simulator default when unspecified).
    """
    for key_name in ("maximum_position_count", "max_concurrent_positions", "slots"):
        if key_name in portfolio_status:
            try:
                value = int(portfolio_status[key_name])
                if value > 0:
                    return value
            except Exception:  # noqa: BLE001
                continue
    return 3


def _read_latest_close(
    symbol: str,
    data_directory: Path,
    valuation_date: datetime.date,
) -> float | None:
    """Read the last available close on or before the valuation date.

    Accepts both 'close' and 'Close' column names; assumes the CSV has a date
    index in the first column (as written by our data loader).
    """
    csv_path = data_directory / f"{symbol}.csv"
    if not csv_path.exists():
        return None
    try:
        frame = pandas.read_csv(csv_path, index_col=0, parse_dates=True)
        if frame.empty:
            return None
        timestamp = pandas.Timestamp(valuation_date)
        eligible = frame.loc[frame.index <= timestamp]
        if eligible.empty:
            return None
        last_row = eligible.iloc[-1]
        if "close" in eligible.columns:
            return float(last_row["close"])  # type: ignore[index]
        if "Close" in eligible.columns:
            return float(last_row["Close"])  # type: ignore[index]
        for column_name in eligible.columns:
            if str(column_name).lower() == "close":
                return float(last_row[column_name])  # type: ignore[index]
        return None
    except Exception:  # noqa: BLE001
        return None

def find_history_signal(
    date_string: str | None,
    dollar_volume_filter: str,
    buy_strategy: str,
    sell_strategy: str,
    stop_loss: float,
    allowed_fama_french_groups: set[int] | None = None,
) -> Dict[str, Any]:
    """Run daily tasks for a single date and return signals and budget data.

    When ``date_string`` is ``None`` the most recent trading date is determined
    via :func:`determine_latest_trading_date` and used for evaluation. Entries
    based on generated signals occur on the next trading day's open.

    Parameters
    ----------
    date_string:
        ISO formatted date string representing the signal date. ``None``
        triggers evaluation for the latest trading day.
    dollar_volume_filter:
        Filter applied to select symbols based on dollar volume.
    buy_strategy:
        Name of the strategy used to generate entry signals.
    sell_strategy:
        Name of the strategy used to generate exit signals.
    stop_loss:
        Fractional loss used for downstream simulations; not used in signal
        detection here but preserved for parity with other entry points.
    allowed_fama_french_groups:
        Optional set of FF12 group identifiers (1–11) used to restrict the
        tradable universe. Group 12 (Other) is always excluded when sector data
        is available.
    Historical data starting from either the earliest cached date in
    ``data/stock_data`` or ``2014-01-01``—whichever is earlier—is used to
    ensure sufficient look-back.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the entry and exit signal lists as well as
        optional budget information derived from ``current_status.json``.
        Budget fields are ``equity``, ``margin``, ``slot_count``,
        ``slot_weight``, ``budget_per_entry``, and ``entry_budgets``.
    """
    # TODO: review
    if date_string is None:
        date_string = determine_latest_trading_date().isoformat()
    group_token = (
        ""
        if not allowed_fama_french_groups
        else "group=" + ",".join(str(i) for i in sorted(allowed_fama_french_groups)) + " "
    )
    argument_line = f"{group_token}{dollar_volume_filter} {buy_strategy} {sell_strategy} {stop_loss}"
    start_date_string = determine_start_date(STOCK_DATA_DIRECTORY)
    minimum_timestamp = pandas.Timestamp(MINIMUM_HISTORY_DATE)
    if pandas.Timestamp(start_date_string) > minimum_timestamp:
        start_date_string = MINIMUM_HISTORY_DATE
    # The downloader uses a half-open interval [start, end), therefore set the
    # end date to the evaluation day so that it is included in the range.
    try:
        evaluation_timestamp = pandas.Timestamp(date_string)
        evaluation_end_date_string = evaluation_timestamp.date().isoformat()
    except Exception:  # noqa: BLE001
        evaluation_timestamp = pandas.Timestamp.today()
        evaluation_end_date_string = evaluation_timestamp.date().isoformat()
    required_start = evaluation_timestamp - pandas.Timedelta(days=150)

    # Align symbol universe with simulator: evaluate all locally cached CSVs.
    try:
        local_symbols = [
            csv_path.stem
            for csv_path in STOCK_DATA_DIRECTORY.glob("*.csv")
            if csv_path.stem and csv_path.stem != "^GSPC"
        ]
    except Exception:  # noqa: BLE001
        local_symbols = None

    # Refresh local history files so cron can rely on up-to-date data without
    # needing to perform network requests for each symbol.
    if local_symbols is not None:
        current_date_string = datetime.date.today().isoformat()
        for symbol_name in local_symbols:
            csv_file_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
            needs_refresh = True
            try:
                cached_index_frame = pandas.read_csv(
                    csv_file_path, usecols=[0], index_col=0, parse_dates=True
                )
                if not cached_index_frame.index.empty:
                    earliest_cached_timestamp = cached_index_frame.index.min()
                    latest_cached_timestamp = cached_index_frame.index.max()
                    if (
                        earliest_cached_timestamp <= required_start
                        and latest_cached_timestamp >= evaluation_timestamp
                    ):
                        needs_refresh = False
            except Exception as read_error:  # noqa: BLE001
                LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)

            if needs_refresh:
                try:
                    download_history(
                        symbol_name,
                        start=start_date_string,
                        end=current_date_string,
                        cache_path=csv_file_path,
                    )
                    try:
                        cached_frame = pandas.read_csv(
                            csv_file_path, index_col=0, parse_dates=True
                        )
                        deduplicated_frame = cached_frame.loc[
                            ~cached_frame.index.duplicated(keep="last")
                        ]
                        deduplicated_frame.to_csv(csv_file_path)
                    except Exception as cache_error:  # noqa: BLE001
                        LOGGER.warning(
                            "Failed to deduplicate %s: %s", csv_file_path, cache_error
                        )
                except Exception as download_error:  # noqa: BLE001
                    LOGGER.warning(
                        "Failed to refresh data for %s: %s", symbol_name, download_error
                    )

    if local_symbols is not None:
        # Ensure every symbol has data for the requested evaluation date.
        missing_symbols: List[str] = []
        for symbol_name in local_symbols:
            csv_file_path = STOCK_DATA_DIRECTORY / f"{symbol_name}.csv"
            try:
                history_frame = pandas.read_csv(
                    csv_file_path, index_col=0, parse_dates=True
                )
            except Exception:  # noqa: BLE001
                missing_symbols.append(symbol_name)
                continue
            if evaluation_timestamp not in history_frame.index:
                missing_symbols.append(symbol_name)
        if missing_symbols:
            warning_symbol_list = ", ".join(sorted(missing_symbols))
            LOGGER.warning(
                "Skipping symbols missing %s: %s",
                date_string,
                warning_symbol_list,
            )
            local_symbols = [
                symbol_name
                for symbol_name in local_symbols
                if symbol_name not in missing_symbols
            ]

    signal_result: Dict[str, List[str]] = cron.run_daily_tasks_from_argument(
        argument_line,
        start_date=start_date_string,
        end_date=evaluation_end_date_string,
        symbol_list=local_symbols,
        data_directory=STOCK_DATA_DIRECTORY,
        use_unshifted_signals=True,
    )

    entry_signals: List[str] = signal_result.get("entry_signals", [])
    exit_signals: List[str] = signal_result.get("exit_signals", [])
    equity_value: float | None = None
    margin_multiplier: float | None = None
    slot_count: int | None = None
    slot_weight: float | None = None
    budget_per_entry: float | None = None
    entry_budgets: Dict[str, float] | None = None
    if entry_signals:
        try:
            portfolio_status = _load_portfolio_status(CURRENT_STATUS_FILE)
            valuation_date = datetime.date.fromisoformat(date_string)
            (
                equity_value,
                margin_multiplier,
                slot_count,
                slot_weight,
            ) = _compute_sizing_inputs(
                portfolio_status, STOCK_DATA_DIRECTORY, valuation_date
            )
            if equity_value is not None and slot_weight is not None:
                budget_per_entry = equity_value * slot_weight
                entry_budgets = {
                    symbol_name: budget_per_entry for symbol_name in entry_signals
                }
        except Exception as status_error:  # noqa: BLE001
            LOGGER.warning(
                "Could not compute budgets from current_status.json: %s",
                status_error,
            )

    return {
        "entry_signals": entry_signals,
        "exit_signals": exit_signals,
        "equity": equity_value,
        "margin": margin_multiplier,
        "slot_count": slot_count,
        "slot_weight": slot_weight,
        "budget_per_entry": budget_per_entry,
        "entry_budgets": entry_budgets,
    }


def main() -> None:
    """Parse command line arguments and run the daily job."""
    parser = argparse.ArgumentParser(description="Run daily cron tasks")
    parser.add_argument(
        "argument_line",
        help=(
            "Task description: 'dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume>NUMBER% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume=TopN BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]' (or legacy 'Nth'), "
            "'dollar_volume>NUMBER,TopN' (or ',Nth'), or 'dollar_volume>NUMBER%,TopN' (or ',Nth')."
        ),
    )
    parsed_arguments = parser.parse_args()
    run_daily_job(parsed_arguments.argument_line)


if __name__ == "__main__":
    main()
