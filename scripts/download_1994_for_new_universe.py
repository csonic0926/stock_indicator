"""Download 1994-2025 yf data for symbols missing from stock_data_1994_clean.

Targets new universe (symbols.txt = 7,447). Only downloads symbols NOT already
present in stock_data_1994_clean (delta = 2,854 missing). Resumable via log.
Writes per-symbol CSV in the same format as existing stock_data_1994_clean files.
"""

from __future__ import annotations

import csv
import datetime as datetime_module
import logging
import time
from pathlib import Path

import pandas
import yfinance

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_TXT = PROJECT_ROOT / "data" / "symbols.txt"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "stock_data_1994_clean"
LOG_PATH = PROJECT_ROOT / "data" / "yf_1994_download_for_new_universe_log.csv"
START_DATE = "1994-01-01"
END_DATE = "2025-12-31"
SLEEP_SECONDS_AFTER_SUCCESS = 0.05
SLEEP_SECONDS_AFTER_FAILURE = 0.5


def load_missing_symbols() -> list[str]:
    """Return symbols in symbols.txt but not yet in stock_data_1994_clean."""
    universe_symbols: set[str] = set()
    for line_text in SYMBOLS_TXT.read_text(encoding="utf-8").splitlines():
        symbol_name = line_text.strip().upper()
        if symbol_name:
            universe_symbols.add(symbol_name)
    existing_symbols = {p.stem.upper() for p in OUTPUT_DIRECTORY.glob("*.csv")}
    missing_symbols = sorted(universe_symbols - existing_symbols)
    return missing_symbols


def flatten_yfinance_frame(price_frame: pandas.DataFrame, symbol_name: str) -> pandas.DataFrame:
    if price_frame.empty:
        return price_frame
    normalized_frame = price_frame.copy()
    if isinstance(normalized_frame.columns, pandas.MultiIndex):
        normalized_frame.columns = normalized_frame.columns.get_level_values(0)
    normalized_frame.columns = [str(c).lower().replace(" ", "_") for c in normalized_frame.columns]
    required_columns = ["close", "high", "low", "open", "volume"]
    missing_cols = [c for c in required_columns if c not in normalized_frame.columns]
    if missing_cols:
        raise ValueError(f"{symbol_name} missing columns: {missing_cols}")
    normalized_frame = normalized_frame[required_columns].copy()
    normalized_frame.index.name = "Date"
    normalized_frame = normalized_frame.sort_index()
    normalized_frame = normalized_frame.loc[~normalized_frame.index.duplicated(keep="last")]
    return normalized_frame


def load_completed_symbols() -> set[str]:
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return set()
    try:
        log_frame = pandas.read_csv(LOG_PATH)
    except Exception:
        return set()
    if "symbol" not in log_frame.columns or "status" not in log_frame.columns:
        return set()
    terminal = {"success", "empty", "error"}
    completed = log_frame[log_frame["status"].isin(terminal)]
    return set(completed["symbol"].dropna().astype(str).str.upper())


def append_log_row(row: dict) -> None:
    field_names = ["timestamp", "symbol", "status", "row_count", "first_date", "last_date", "output_path", "error"]
    file_exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def download_symbol(symbol_name: str):
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    missing_symbols = load_missing_symbols()
    completed_symbols = load_completed_symbols()
    pending = [s for s in missing_symbols if s not in completed_symbols]
    LOGGER.info(
        "1994 universe-fill download: %s missing total, %s already logged, %s pending",
        len(missing_symbols),
        len(completed_symbols),
        len(pending),
    )

    success_count = 0
    empty_count = 0
    error_count = 0
    for index, symbol_name in enumerate(pending, start=1):
        output_path = OUTPUT_DIRECTORY / f"{symbol_name}.csv"
        timestamp = datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()
        try:
            price_frame, status = download_symbol(symbol_name)
            if status == "success" and not price_frame.empty:
                price_frame.to_csv(output_path)
                first_date = price_frame.index.min().date().isoformat()
                last_date = price_frame.index.max().date().isoformat()
                row_count = len(price_frame)
                append_log_row(dict(
                    timestamp=timestamp, symbol=symbol_name, status="success",
                    row_count=row_count, first_date=first_date, last_date=last_date,
                    output_path=str(output_path), error="",
                ))
                success_count += 1
                LOGGER.info("[%d/%d] %s success rows=%d %s→%s", index, len(pending), symbol_name, row_count, first_date, last_date)
                time.sleep(SLEEP_SECONDS_AFTER_SUCCESS)
            else:
                append_log_row(dict(
                    timestamp=timestamp, symbol=symbol_name, status="empty",
                    row_count=0, first_date="", last_date="", output_path="", error="empty_dataframe",
                ))
                empty_count += 1
                LOGGER.info("[%d/%d] %s EMPTY", index, len(pending), symbol_name)
                time.sleep(SLEEP_SECONDS_AFTER_FAILURE)
        except Exception as error:
            append_log_row(dict(
                timestamp=timestamp, symbol=symbol_name, status="error",
                row_count=0, first_date="", last_date="", output_path="", error=str(error)[:500],
            ))
            error_count += 1
            LOGGER.warning("[%d/%d] %s ERROR: %s", index, len(pending), symbol_name, str(error)[:200])
            time.sleep(SLEEP_SECONDS_AFTER_FAILURE)

        # Progress checkpoint every 100
        if index % 100 == 0:
            LOGGER.info(
                "PROGRESS [%d/%d] success=%d empty=%d error=%d",
                index, len(pending), success_count, empty_count, error_count,
            )

    LOGGER.info(
        "DONE. success=%d empty=%d error=%d total_processed=%d",
        success_count, empty_count, error_count, len(pending),
    )


if __name__ == "__main__":
    main()
