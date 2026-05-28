"""Prepare the 2010 Yahoo backtest data source for the current universe.

The script makes ``data/stock_data_2010_yf_clean`` cover every symbol in
``data/symbols.txt``. It first fills gaps from existing local long-history
sources, merges the current daily Yahoo cache for fresh rows, then optionally
uses Yahoo Finance only for symbols that still need a historical backfill.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import logging
from pathlib import Path
import sys
from typing import Any

import pandas

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from stock_indicator.data_loader import download_history  # noqa: E402
from stock_indicator.daily_job import determine_latest_trading_date  # noqa: E402

LOGGER = logging.getLogger(__name__)

DATA_DIRECTORY = REPOSITORY_ROOT / "data"
SYMBOLS_PATH = DATA_DIRECTORY / "symbols.txt"
TARGET_DIRECTORY = DATA_DIRECTORY / "stock_data_2010_yf_clean"
CURRENT_DAILY_DIRECTORY = DATA_DIRECTORY / "stock_data"
LOCAL_HISTORY_SOURCE_DIRECTORIES = [
    DATA_DIRECTORY / "stock_data_1994",
    DATA_DIRECTORY / "stock_data_2014",
]
AUDIT_PATH = DATA_DIRECTORY / "stock_data_2010_yf_clean_prepare_audit.csv"
EXPECTED_PRICE_COLUMNS = ["close", "high", "low", "open", "volume"]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    default_end_date = (
        determine_latest_trading_date() + datetime.timedelta(days=1)
    ).isoformat()
    parser = argparse.ArgumentParser(
        description="Prepare stock_data_2010_yf_clean for current symbols.txt.",
    )
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=default_end_date)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Only fill from local data sources; do not call Yahoo Finance.",
    )
    parser.add_argument(
        "--clean-extras",
        action="store_true",
        help="Move target CSVs not in symbols.txt to an excluded subdirectory.",
    )
    return parser.parse_args()


def load_current_symbols(symbols_path: Path = SYMBOLS_PATH) -> list[str]:
    """Return the current universe symbols."""

    return sorted(
        {
            line_text.strip().upper()
            for line_text in symbols_path.read_text(encoding="utf-8").splitlines()
            if line_text.strip()
        }
    )


def normalize_price_frame(raw_frame: pandas.DataFrame) -> pandas.DataFrame:
    """Return a normalized OHLCV frame indexed by date."""

    if raw_frame.empty:
        return pandas.DataFrame(columns=EXPECTED_PRICE_COLUMNS)
    frame = raw_frame.copy()
    if "Date" in frame.columns:
        frame["Date"] = pandas.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date")
    else:
        first_column_name = str(frame.columns[0])
        frame[first_column_name] = pandas.to_datetime(
            frame[first_column_name],
            errors="coerce",
        )
        frame = frame.dropna(subset=[first_column_name]).set_index(first_column_name)
    if isinstance(frame.columns, pandas.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [
        str(column_name).strip().lower().replace(" ", "_")
        for column_name in frame.columns
    ]
    selected_columns = [
        column_name
        for column_name in EXPECTED_PRICE_COLUMNS
        if column_name in frame.columns
    ]
    if not selected_columns:
        return pandas.DataFrame(columns=EXPECTED_PRICE_COLUMNS)
    frame = frame[selected_columns].copy()
    for expected_column in EXPECTED_PRICE_COLUMNS:
        if expected_column not in frame.columns:
            frame[expected_column] = pandas.NA
    frame = frame[EXPECTED_PRICE_COLUMNS]
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()
    frame.index.name = "Date"
    return frame


def read_price_csv(csv_path: Path) -> pandas.DataFrame:
    """Read and normalize one price CSV file."""

    try:
        raw_frame = pandas.read_csv(csv_path)
    except (OSError, pandas.errors.EmptyDataError, pandas.errors.ParserError) as error:
        LOGGER.warning("Could not read %s: %s", csv_path, error)
        return pandas.DataFrame(columns=EXPECTED_PRICE_COLUMNS)
    return normalize_price_frame(raw_frame)


def write_price_csv(price_frame: pandas.DataFrame, csv_path: Path) -> None:
    """Write a normalized price frame to ``csv_path``."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame = price_frame.copy()
    output_frame.index.name = "Date"
    output_frame.to_csv(csv_path)


def trim_to_start_date(price_frame: pandas.DataFrame, start_date: str) -> pandas.DataFrame:
    """Return rows on or after ``start_date``."""

    if price_frame.empty:
        return price_frame
    return price_frame.loc[price_frame.index >= pandas.Timestamp(start_date)]


def merge_price_frames(price_frames: list[pandas.DataFrame]) -> pandas.DataFrame:
    """Merge price frames with later frames overriding duplicate dates."""

    non_empty_frames = [price_frame for price_frame in price_frames if not price_frame.empty]
    if not non_empty_frames:
        return pandas.DataFrame(columns=EXPECTED_PRICE_COLUMNS)
    merged_frame = pandas.concat(non_empty_frames).sort_index()
    merged_frame = merged_frame.loc[~merged_frame.index.duplicated(keep="last")]
    merged_frame = merged_frame[EXPECTED_PRICE_COLUMNS]
    merged_frame.index.name = "Date"
    return merged_frame


def first_and_last_date(price_frame: pandas.DataFrame) -> tuple[str, str]:
    """Return first and last dates for audit output."""

    if price_frame.empty:
        return "", ""
    return (
        pandas.Timestamp(price_frame.index.min()).date().isoformat(),
        pandas.Timestamp(price_frame.index.max()).date().isoformat(),
    )


def build_local_history_frame(symbol: str, start_date: str) -> tuple[pandas.DataFrame, str]:
    """Return the best local long-history frame for ``symbol``."""

    for source_directory in LOCAL_HISTORY_SOURCE_DIRECTORIES:
        source_path = source_directory / f"{symbol}.csv"
        if not source_path.exists():
            continue
        source_frame = trim_to_start_date(read_price_csv(source_path), start_date)
        if not source_frame.empty:
            return source_frame, source_directory.name
    return pandas.DataFrame(columns=EXPECTED_PRICE_COLUMNS), ""


def seed_symbol_from_local_sources(
    symbol: str,
    *,
    start_date: str,
    originally_missing: bool,
) -> dict[str, Any]:
    """Create or update one target CSV from local sources."""

    target_path = TARGET_DIRECTORY / f"{symbol}.csv"
    existing_frame = (
        trim_to_start_date(read_price_csv(target_path), start_date)
        if target_path.exists()
        else pandas.DataFrame(columns=EXPECTED_PRICE_COLUMNS)
    )
    long_history_frame, long_history_source = build_local_history_frame(
        symbol,
        start_date,
    )
    daily_path = CURRENT_DAILY_DIRECTORY / f"{symbol}.csv"
    daily_frame = (
        read_price_csv(daily_path)
        if daily_path.exists()
        else pandas.DataFrame(columns=EXPECTED_PRICE_COLUMNS)
    )
    merged_frame = merge_price_frames(
        [existing_frame, long_history_frame, daily_frame],
    )
    if not merged_frame.empty:
        write_price_csv(merged_frame, target_path)
    first_date, last_date = first_and_last_date(merged_frame)
    needs_network_backfill = (
        long_history_source == ""
        and not merged_frame.empty
        and first_date > start_date
    )
    return {
        "symbol": symbol,
        "local_status": "seeded" if not merged_frame.empty else "missing_local_data",
        "long_history_source": long_history_source,
        "row_count_after_local": len(merged_frame),
        "first_date_after_local": first_date,
        "last_date_after_local": last_date,
        "needs_network_backfill": needs_network_backfill,
        "network_status": "not_requested",
        "network_error": "",
        "row_count_final": len(merged_frame),
        "first_date_final": first_date,
        "last_date_final": last_date,
    }


def backfill_symbol_from_yahoo(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Backfill one symbol from Yahoo using the target CSV as cache."""

    target_path = TARGET_DIRECTORY / f"{symbol}.csv"
    try:
        downloaded_frame = download_history(
            symbol,
            start=start_date,
            end=end_date,
            cache_path=target_path,
        )
        normalized_frame = normalize_price_frame(downloaded_frame.reset_index())
        if not normalized_frame.empty:
            write_price_csv(trim_to_start_date(normalized_frame, start_date), target_path)
        final_frame = read_price_csv(target_path)
        first_date, last_date = first_and_last_date(final_frame)
        return {
            "symbol": symbol,
            "network_status": "downloaded",
            "network_error": "",
            "row_count_final": len(final_frame),
            "first_date_final": first_date,
            "last_date_final": last_date,
        }
    except Exception as error:  # noqa: BLE001
        final_frame = read_price_csv(target_path) if target_path.exists() else pandas.DataFrame()
        first_date, last_date = first_and_last_date(final_frame)
        return {
            "symbol": symbol,
            "network_status": "failed",
            "network_error": str(error),
            "row_count_final": len(final_frame),
            "first_date_final": first_date,
            "last_date_final": last_date,
        }


def move_extra_csvs(current_symbols: set[str]) -> int:
    """Move target CSV files that are not part of the current universe."""

    excluded_directory = TARGET_DIRECTORY.with_name(
        f"{TARGET_DIRECTORY.name}_excluded_non_current"
    )
    moved_count = 0
    for csv_path in TARGET_DIRECTORY.glob("*.csv"):
        if csv_path.stem in current_symbols:
            continue
        excluded_directory.mkdir(parents=True, exist_ok=True)
        destination_path = excluded_directory / csv_path.name
        csv_path.replace(destination_path)
        moved_count += 1
    return moved_count


def write_audit(records: list[dict[str, Any]], audit_path: Path = AUDIT_PATH) -> None:
    """Write preparation audit records."""

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "symbol",
        "local_status",
        "long_history_source",
        "row_count_after_local",
        "first_date_after_local",
        "last_date_after_local",
        "needs_network_backfill",
        "network_status",
        "network_error",
        "row_count_final",
        "first_date_final",
        "last_date_final",
    ]
    with audit_path.open("w", newline="", encoding="utf-8") as audit_file:
        writer = csv.DictWriter(audit_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Prepare the current-universe 2010 Yahoo data source."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_arguments()
    current_symbols = load_current_symbols()
    existing_target_symbols = {
        csv_path.stem for csv_path in TARGET_DIRECTORY.glob("*.csv")
    }
    originally_missing_symbols = set(current_symbols) - existing_target_symbols
    LOGGER.info("Current symbols: %d", len(current_symbols))
    LOGGER.info("Target CSVs before: %d", len(existing_target_symbols))
    LOGGER.info("Missing before local fill: %d", len(originally_missing_symbols))

    audit_records: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(current_symbols, start=1):
        record = seed_symbol_from_local_sources(
            symbol,
            start_date=arguments.start_date,
            originally_missing=symbol in originally_missing_symbols,
        )
        audit_records.append(record)
        if symbol_index % 250 == 0 or symbol_index == len(current_symbols):
            LOGGER.info("Local fill progress: %d/%d", symbol_index, len(current_symbols))

    records_by_symbol = {str(record["symbol"]): record for record in audit_records}
    network_symbols = [
        str(record["symbol"])
        for record in audit_records
        if bool(record["needs_network_backfill"])
    ]
    LOGGER.info("Network backfill candidates: %d", len(network_symbols))
    if network_symbols and not arguments.skip_network:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, arguments.max_workers),
        ) as executor:
            future_to_symbol = {
                executor.submit(
                    backfill_symbol_from_yahoo,
                    symbol,
                    start_date=arguments.start_date,
                    end_date=arguments.end_date,
                ): symbol
                for symbol in network_symbols
            }
            completed_count = 0
            for completed_future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[completed_future]
                network_result = completed_future.result()
                records_by_symbol[symbol].update(network_result)
                completed_count += 1
                LOGGER.info(
                    "Network backfill progress: %d/%d %s %s rows=%s",
                    completed_count,
                    len(network_symbols),
                    symbol,
                    network_result["network_status"],
                    network_result["row_count_final"],
                )

    if arguments.clean_extras:
        moved_count = move_extra_csvs(set(current_symbols))
        LOGGER.info("Moved non-current extra CSVs: %d", moved_count)

    final_target_symbols = {
        csv_path.stem for csv_path in TARGET_DIRECTORY.glob("*.csv")
    }
    missing_after = sorted(set(current_symbols) - final_target_symbols)
    extra_after = sorted(final_target_symbols - set(current_symbols))
    write_audit(list(records_by_symbol.values()))
    LOGGER.info("Target CSVs after: %d", len(final_target_symbols))
    LOGGER.info("Missing after: %d", len(missing_after))
    LOGGER.info("Extra after: %d", len(extra_after))
    LOGGER.info("Audit written: %s", AUDIT_PATH)
    if missing_after:
        raise SystemExit(f"Missing current symbols after preparation: {missing_after[:20]}")


if __name__ == "__main__":
    main()
