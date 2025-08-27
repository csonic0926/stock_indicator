"""Entry point for running the daily cron tasks."""
# TODO: review

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List

import pandas

from . import cron

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2019-01-01"
DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
LOG_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"


def determine_start_date(data_directory: Path) -> str:
    """Return the oldest date found in ``data_directory``.

    When no CSV files are available, ``DEFAULT_START_DATE`` is returned.
    """
    earliest_date: datetime.date | None = None
    if not data_directory.exists():
        return DEFAULT_START_DATE
    for csv_file_path in data_directory.glob("*.csv"):
        try:
            date_frame = pandas.read_csv(
                csv_file_path, usecols=[0], nrows=1, parse_dates=[0]
            )
        except Exception as read_error:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)
            continue
        if date_frame.empty:
            continue
        first_date = date_frame.iloc[0, 0].date()
        if earliest_date is None or first_date < earliest_date:
            earliest_date = first_date
    if earliest_date is None:
        return DEFAULT_START_DATE
    return earliest_date.isoformat()


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
        Directory where downloaded price history data is written. Defaults to the
        project's ``data`` directory.
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
    if current_date is None:
        current_date = datetime.date.today()
    if data_directory is None:
        data_directory = DATA_DIRECTORY
    if log_directory is None:
        log_directory = LOG_DIRECTORY

    current_date_string = current_date.isoformat()
    LOGGER.info("Starting daily tasks for %s", current_date_string)
    start_date_string = determine_start_date(data_directory)
    signal_result: Dict[str, list[str]] = cron.run_daily_tasks_from_argument(
        argument_line,
        start_date=start_date_string,
        end_date=current_date_string,
        data_directory=data_directory,
    )
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file_path = log_directory / f"{current_date_string}.log"
    with log_file_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            f"entry_signals: {', '.join(signal_result['entry_signals'])}\n",
        )
        log_file.write(
            f"exit_signals: {', '.join(signal_result['exit_signals'])}\n",
        )
    LOGGER.info("Daily tasks completed; results written to %s", log_file_path)
    return log_file_path


def find_signal(
    date_string: str,
    dollar_volume_filter: str,
    buy_strategy: str,
    sell_strategy: str,
    stop_loss: float,
) -> Dict[str, List[str]]:
    """Run daily tasks for a single date and return the signals.

    Parameters
    ----------
    date_string:
        ISO formatted date string representing the day to evaluate.
    dollar_volume_filter:
        Filter applied to select symbols based on dollar volume.
    buy_strategy:
        Name of the strategy used to generate entry signals.
    sell_strategy:
        Name of the strategy used to generate exit signals.
    stop_loss:
        Fractional loss that triggers an exit on the next day's open.
    Historical data from the earliest available date in ``DATA_DIRECTORY`` is
    used to ensure sufficient look-back.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing two keys: ``"entry_signals"`` and
        ``"exit_signals"``. Each maps to the list of symbols produced by
        the strategies.
    """
    # TODO: review
    argument_line = (
        f"{dollar_volume_filter} {buy_strategy} {sell_strategy} {stop_loss}"
    )
    start_date_string = determine_start_date(DATA_DIRECTORY)
    signal_result: Dict[str, List[str]] = cron.run_daily_tasks_from_argument(
        argument_line,
        start_date=start_date_string,
        end_date=date_string,
        data_directory=DATA_DIRECTORY,
    )
    return signal_result


def main() -> None:
    """Parse command line arguments and run the daily job."""
    parser = argparse.ArgumentParser(description="Run daily cron tasks")
    parser.add_argument(
        "argument_line",
        help=(
            "Task description: 'dollar_volume>N%,K% BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]'"
        ),
    )
    parsed_arguments = parser.parse_args()
    run_daily_job(parsed_arguments.argument_line)


if __name__ == "__main__":
    main()
