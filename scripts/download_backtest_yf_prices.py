"""Download one-off Yahoo Finance price history for clean backtest symbols."""

from __future__ import annotations

import csv
import datetime as datetime_module
import logging
import time
from pathlib import Path
from typing import Iterable

import pandas
import yfinance

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_SYMBOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtest_universe_alpha_vantage"
    / "backtest_common_stock_yf_symbols_2010_2026_plus_runtime.txt"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "stock_data_2010_yf_clean"
LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtest_universe_alpha_vantage"
    / "yf_full_download_2010_2026_log.csv"
)
START_DATE = "2010-01-01"
END_DATE = "2026-05-03"
SLEEP_SECONDS_AFTER_SUCCESS = 0.05
SLEEP_SECONDS_AFTER_FAILURE = 0.5


def load_symbols(symbol_path: Path) -> list[str]:
    """Return unique symbols from a newline-delimited symbol file."""

    seen_symbols: set[str] = set()
    ordered_symbols: list[str] = []
    for line_text in symbol_path.read_text(encoding="utf-8").splitlines():
        symbol_name = line_text.strip().upper()
        if not symbol_name or symbol_name in seen_symbols:
            continue
        seen_symbols.add(symbol_name)
        ordered_symbols.append(symbol_name)
    return ordered_symbols


def flatten_yfinance_frame(price_frame: pandas.DataFrame, symbol_name: str) -> pandas.DataFrame:
    """Return a single-symbol OHLCV frame with repository column names."""

    if price_frame.empty:
        return price_frame
    normalized_frame = price_frame.copy()
    if isinstance(normalized_frame.columns, pandas.MultiIndex):
        # yfinance commonly returns (Price, Ticker) for single-symbol downloads.
        normalized_frame.columns = normalized_frame.columns.get_level_values(0)
    normalized_frame.columns = [str(column_name).lower().replace(" ", "_") for column_name in normalized_frame.columns]
    required_columns = ["close", "high", "low", "open", "volume"]
    missing_columns = [column_name for column_name in required_columns if column_name not in normalized_frame.columns]
    if missing_columns:
        raise ValueError(f"{symbol_name} missing columns: {missing_columns}")
    normalized_frame = normalized_frame[required_columns].copy()
    normalized_frame.index.name = "Date"
    normalized_frame = normalized_frame.sort_index()
    normalized_frame = normalized_frame.loc[~normalized_frame.index.duplicated(keep="last")]
    return normalized_frame


def load_completed_symbols(log_path: Path) -> set[str]:
    """Return symbols already logged with a terminal status."""

    if not log_path.exists() or log_path.stat().st_size == 0:
        return set()
    try:
        log_frame = pandas.read_csv(log_path)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Could not read existing log %s: %s", log_path, error)
        return set()
    if "symbol" not in log_frame.columns or "status" not in log_frame.columns:
        return set()
    terminal_statuses = {"success", "empty", "error"}
    completed_frame = log_frame[log_frame["status"].isin(terminal_statuses)]
    return set(completed_frame["symbol"].dropna().astype(str).str.upper())


def append_log_row(log_path: Path, row: dict[str, object]) -> None:
    """Append one CSV log row."""

    field_names = [
        "timestamp",
        "symbol",
        "status",
        "row_count",
        "first_date",
        "last_date",
        "output_path",
        "error",
    ]
    file_exists = log_path.exists() and log_path.stat().st_size > 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def download_symbol(symbol_name: str) -> tuple[pandas.DataFrame, str]:
    """Download and normalize one symbol from Yahoo Finance."""

    raw_frame = yfinance.download(
        symbol_name,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw_frame.empty:
        return raw_frame, "empty"
    return flatten_yfinance_frame(raw_frame, symbol_name), "success"


def main() -> None:
    """Run the one-off download with resumable success logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    symbol_list = load_symbols(INPUT_SYMBOL_PATH)
    completed_symbols = load_completed_symbols(LOG_PATH)
    total_symbol_count = len(symbol_list)
    LOGGER.info("Starting YF download: %s symbols, %s already logged", total_symbol_count, len(completed_symbols))

    for symbol_index, symbol_name in enumerate(symbol_list, start=1):
        output_path = OUTPUT_DIRECTORY / f"{symbol_name}.csv"
        if symbol_name in completed_symbols:
            continue
        timestamp_text = datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()
        try:
            price_frame, status = download_symbol(symbol_name)
            if status == "success" and not price_frame.empty:
                price_frame.to_csv(output_path)
                first_date = price_frame.index.min().date().isoformat()
                last_date = price_frame.index.max().date().isoformat()
                row_count = len(price_frame)
                append_log_row(
                    LOG_PATH,
                    {
                        "timestamp": timestamp_text,
                        "symbol": symbol_name,
                        "status": "success",
                        "row_count": row_count,
                        "first_date": first_date,
                        "last_date": last_date,
                        "output_path": str(output_path),
                        "error": "",
                    },
                )
                LOGGER.info("[%s/%s] %s success rows=%s %s..%s", symbol_index, total_symbol_count, symbol_name, row_count, first_date, last_date)
                time.sleep(SLEEP_SECONDS_AFTER_SUCCESS)
            else:
                append_log_row(
                    LOG_PATH,
                    {
                        "timestamp": timestamp_text,
                        "symbol": symbol_name,
                        "status": "empty",
                        "row_count": 0,
                        "first_date": "",
                        "last_date": "",
                        "output_path": "",
                        "error": "empty download",
                    },
                )
                LOGGER.info("[%s/%s] %s empty", symbol_index, total_symbol_count, symbol_name)
                time.sleep(SLEEP_SECONDS_AFTER_FAILURE)
        except Exception as error:  # noqa: BLE001
            append_log_row(
                LOG_PATH,
                {
                    "timestamp": timestamp_text,
                    "symbol": symbol_name,
                    "status": "error",
                    "row_count": 0,
                    "first_date": "",
                    "last_date": "",
                    "output_path": "",
                    "error": repr(error),
                },
            )
            LOGGER.warning("[%s/%s] %s error: %s", symbol_index, total_symbol_count, symbol_name, error)
            time.sleep(SLEEP_SECONDS_AFTER_FAILURE)

    LOGGER.info("Finished YF download")


if __name__ == "__main__":
    main()
