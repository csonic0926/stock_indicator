"""Append promoted-symbol FF12 rows into the production sector contract.

The production-old sector file is an audited trading contract. Existing rows
must remain frozen, while newly promoted production symbols receive their
already-audited research classification rows from the research-new sector
output.
"""

# TODO: review

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
from typing import Iterable

import pandas

from stock_indicator.sector_pipeline.config import DATA_DIRECTORY
from stock_indicator.symbols import normalize_symbol_for_cache

LOGGER = logging.getLogger(__name__)

PRODUCTION_SYMBOLS_FILE_NAME = "production_old_symbols.txt"
PRODUCTION_SECTOR_PARQUET_FILE_NAME = "production_old_symbols_with_sector.parquet"
PRODUCTION_SECTOR_CSV_FILE_NAME = "production_old_symbols_with_sector.csv"
RESEARCH_SECTOR_PARQUET_FILE_NAME = "research_new_symbols_with_sector.parquet"
RESEARCH_SECTOR_CSV_FILE_NAME = "research_new_symbols_with_sector.csv"

REQUIRED_SECTOR_COLUMNS = {
    "ticker",
    "ff12",
    "ff12_source",
    "classification_confidence",
}
LOW_CONFIDENCE_VALUE = "low"
MISSING_SIC_FALLBACK_SOURCE = "missing_sic_fallback"


@dataclass(frozen=True)
class ProductionFf12PromotionPaths:
    """Filesystem paths used by production FF12 promotion."""

    data_directory: Path

    @property
    def production_symbols_path(self) -> Path:
        """Return the active production symbol contract path."""

        return self.data_directory / PRODUCTION_SYMBOLS_FILE_NAME

    @property
    def production_sector_parquet_path(self) -> Path:
        """Return the active production sector parquet path."""

        return self.data_directory / PRODUCTION_SECTOR_PARQUET_FILE_NAME

    @property
    def production_sector_csv_path(self) -> Path:
        """Return the active production sector CSV path."""

        return self.data_directory / PRODUCTION_SECTOR_CSV_FILE_NAME

    @property
    def research_sector_parquet_path(self) -> Path:
        """Return the research sector parquet source path."""

        return self.data_directory / RESEARCH_SECTOR_PARQUET_FILE_NAME

    @property
    def research_sector_csv_path(self) -> Path:
        """Return the research sector CSV source path."""

        return self.data_directory / RESEARCH_SECTOR_CSV_FILE_NAME


@dataclass(frozen=True)
class ProductionFf12PromotionReport:
    """Summary of a production FF12 promotion sync."""

    published: bool
    production_symbol_count: int
    original_sector_row_count: int
    final_sector_row_count: int
    appended_symbols: list[str]
    removed_sector_symbols: list[str]
    ff12_source_counts: dict[str, int]
    output_paths: dict[str, str]

    def to_lines(self) -> list[str]:
        """Return a compact human-readable report."""

        action_label = "published" if self.published else "dry run completed"
        appended_sample = ", ".join(self.appended_symbols[:20]) or "none"
        removed_sample = ", ".join(self.removed_sector_symbols[:20]) or "none"
        return [
            f"Production FF12 promotion sync {action_label}",
            f"production symbols: {self.production_symbol_count}",
            f"original sector rows: {self.original_sector_row_count}",
            f"final sector rows: {self.final_sector_row_count}",
            f"appended promoted symbols: {len(self.appended_symbols)} ({appended_sample})",
            f"removed inactive sector rows: {len(self.removed_sector_symbols)} ({removed_sample})",
            f"ff12 sources: {_format_count_mapping(self.ff12_source_counts)}",
        ]


@dataclass(frozen=True)
class ProductionFf12PromotionBuildResult:
    """In-memory promoted sector frame plus its symbol-level diff."""

    sector_frame: pandas.DataFrame
    appended_symbols: list[str]
    removed_sector_symbols: list[str]


def _format_count_mapping(count_mapping: dict[str, int]) -> str:
    """Return a deterministic compact count mapping."""

    if not count_mapping:
        return "{}"
    count_parts = [
        f"{count_key}={count_mapping[count_key]}"
        for count_key in sorted(count_mapping)
    ]
    return ", ".join(count_parts)


def _count_values(frame: pandas.DataFrame, column_name: str) -> dict[str, int]:
    """Return value counts for ``column_name`` as a plain dictionary."""

    if column_name not in frame.columns:
        return {}
    count_series = frame[column_name].astype(str).value_counts()
    return {str(value): int(count) for value, count in count_series.items()}


def load_production_symbols(production_symbols_path: Path) -> list[str]:
    """Return production symbols in file order, rejecting duplicates."""

    if not production_symbols_path.exists():
        raise FileNotFoundError(
            f"production symbols file not found: {production_symbols_path}"
        )

    production_symbols: list[str] = []
    seen_symbols: set[str] = set()
    duplicate_symbols: list[str] = []
    for line_text in production_symbols_path.read_text(
        encoding="utf-8"
    ).splitlines():
        symbol_name = normalize_symbol_for_cache(line_text)
        if not symbol_name:
            continue
        if symbol_name in seen_symbols:
            duplicate_symbols.append(symbol_name)
            continue
        seen_symbols.add(symbol_name)
        production_symbols.append(symbol_name)

    if duplicate_symbols:
        raise ValueError(
            "production symbols file has duplicate symbols: "
            f"{duplicate_symbols[:10]}"
        )
    return production_symbols


def _read_sector_frame_from_pair(
    *,
    parquet_path: Path,
    csv_path: Path,
    source_label: str,
) -> tuple[pandas.DataFrame, Path]:
    """Read a sector frame, preferring parquet and falling back to CSV."""

    if parquet_path.exists():
        sector_frame = pandas.read_parquet(parquet_path)
        selected_path = parquet_path
    elif csv_path.exists():
        sector_frame = pandas.read_csv(csv_path, keep_default_na=False)
        selected_path = csv_path
    else:
        raise FileNotFoundError(
            f"{source_label} sector file not found: "
            f"{parquet_path} or {csv_path}"
        )

    normalized_frame = sector_frame.copy()
    normalized_frame.columns = [
        str(column_name).strip()
        for column_name in normalized_frame.columns
    ]
    return normalized_frame, selected_path


def _normalized_ticker_series(sector_frame: pandas.DataFrame) -> pandas.Series:
    """Return normalized non-empty tickers from a sector frame."""

    return sector_frame["ticker"].map(normalize_symbol_for_cache)


def _validate_sector_source_frame(
    sector_frame: pandas.DataFrame,
    *,
    source_label: str,
) -> None:
    """Validate columns and duplicate tickers for a sector source frame."""

    missing_columns = REQUIRED_SECTOR_COLUMNS - set(sector_frame.columns)
    if missing_columns:
        raise ValueError(
            f"{source_label} sector frame missing columns: "
            f"{sorted(missing_columns)}"
        )

    normalized_tickers = _normalized_ticker_series(sector_frame)
    blank_ticker_count = int((normalized_tickers == "").sum())
    if blank_ticker_count:
        raise ValueError(
            f"{source_label} sector frame has blank tickers: {blank_ticker_count}"
        )

    duplicate_tickers = normalized_tickers.loc[
        normalized_tickers.duplicated()
    ].tolist()
    if duplicate_tickers:
        raise ValueError(
            f"{source_label} sector frame has duplicate tickers: "
            f"{duplicate_tickers[:10]}"
        )


def _build_row_index_by_ticker(
    sector_frame: pandas.DataFrame,
) -> dict[str, int]:
    """Return a normalized ticker to row-index lookup."""

    normalized_tickers = _normalized_ticker_series(sector_frame)
    return {
        str(ticker_symbol): row_position
        for row_position, ticker_symbol in enumerate(normalized_tickers.tolist())
    }


def validate_production_ff12_sector_contract(
    sector_frame: pandas.DataFrame,
    production_symbols: Iterable[str],
) -> None:
    """Validate the active production FF12 sector contract before publish."""

    missing_columns = REQUIRED_SECTOR_COLUMNS - set(sector_frame.columns)
    if missing_columns:
        raise ValueError(
            "production FF12 sector frame missing columns: "
            f"{sorted(missing_columns)}"
        )

    normalized_production_symbols: list[str] = []
    seen_production_symbols: set[str] = set()
    duplicate_production_symbols: list[str] = []
    for symbol_name in production_symbols:
        normalized_symbol_name = normalize_symbol_for_cache(symbol_name)
        if not normalized_symbol_name:
            continue
        if normalized_symbol_name in seen_production_symbols:
            duplicate_production_symbols.append(normalized_symbol_name)
            continue
        seen_production_symbols.add(normalized_symbol_name)
        normalized_production_symbols.append(normalized_symbol_name)
    if duplicate_production_symbols:
        raise ValueError(
            "production symbol contract has duplicate symbols: "
            f"{duplicate_production_symbols[:10]}"
        )

    normalized_sector_tickers = _normalized_ticker_series(sector_frame)
    duplicate_sector_tickers = normalized_sector_tickers.loc[
        normalized_sector_tickers.duplicated()
    ].tolist()
    if duplicate_sector_tickers:
        raise ValueError(
            "production FF12 sector frame has duplicate tickers: "
            f"{duplicate_sector_tickers[:10]}"
        )

    expected_symbol_set = set(normalized_production_symbols)
    actual_symbol_set = set(normalized_sector_tickers.astype(str))
    missing_symbols = sorted(expected_symbol_set - actual_symbol_set)
    extra_symbols = sorted(actual_symbol_set - expected_symbol_set)
    if missing_symbols or extra_symbols:
        raise ValueError(
            "production FF12 sector frame does not match production symbols: "
            f"missing={missing_symbols[:10]}, extra={extra_symbols[:10]}"
        )

    ff12_values = pandas.to_numeric(sector_frame["ff12"], errors="coerce")
    invalid_ff12_mask = (
        ff12_values.isna()
        | ff12_values.lt(1)
        | ff12_values.gt(12)
        | (ff12_values % 1 != 0)
    )
    if invalid_ff12_mask.any():
        invalid_symbols = normalized_sector_tickers.loc[
            invalid_ff12_mask
        ].tolist()
        raise ValueError(
            "production FF12 sector frame has invalid FF12 values: "
            f"{invalid_symbols[:10]}"
        )

    confidence_series = (
        sector_frame["classification_confidence"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    low_confidence_symbols = normalized_sector_tickers.loc[
        confidence_series == LOW_CONFIDENCE_VALUE
    ].tolist()
    if low_confidence_symbols:
        raise ValueError(
            "production FF12 sector frame has low-confidence rows: "
            f"{low_confidence_symbols[:10]}"
        )

    source_series = (
        sector_frame["ff12_source"].astype(str).str.strip().str.lower()
    )
    fallback_symbols = normalized_sector_tickers.loc[
        source_series == MISSING_SIC_FALLBACK_SOURCE
    ].tolist()
    if fallback_symbols:
        raise ValueError(
            "production FF12 sector frame has missing-SIC fallback rows: "
            f"{fallback_symbols[:10]}"
        )


def build_promoted_production_ff12_sector_frame(
    *,
    production_symbols: list[str],
    production_sector_frame: pandas.DataFrame,
    research_sector_frame: pandas.DataFrame,
) -> ProductionFf12PromotionBuildResult:
    """Return production sector rows plus research rows for promoted symbols."""

    normalized_production_symbols: list[str] = []
    seen_production_symbols: set[str] = set()
    duplicate_production_symbols: list[str] = []
    for symbol_name in production_symbols:
        normalized_symbol_name = normalize_symbol_for_cache(symbol_name)
        if not normalized_symbol_name:
            continue
        if normalized_symbol_name in seen_production_symbols:
            duplicate_production_symbols.append(normalized_symbol_name)
            continue
        seen_production_symbols.add(normalized_symbol_name)
        normalized_production_symbols.append(normalized_symbol_name)
    if duplicate_production_symbols:
        raise ValueError(
            "production symbol contract has duplicate symbols: "
            f"{duplicate_production_symbols[:10]}"
        )

    _validate_sector_source_frame(
        production_sector_frame,
        source_label="production",
    )
    _validate_sector_source_frame(
        research_sector_frame,
        source_label="research",
    )

    production_schema_columns = list(production_sector_frame.columns)
    production_sector_row_by_ticker = _build_row_index_by_ticker(
        production_sector_frame
    )
    research_sector_row_by_ticker = _build_row_index_by_ticker(
        research_sector_frame
    )

    production_symbol_set = set(normalized_production_symbols)
    missing_production_symbols = [
        symbol_name
        for symbol_name in normalized_production_symbols
        if symbol_name not in production_sector_row_by_ticker
    ]
    missing_research_symbols = [
        symbol_name
        for symbol_name in missing_production_symbols
        if symbol_name not in research_sector_row_by_ticker
    ]
    if missing_research_symbols:
        raise ValueError(
            "promoted production symbols are missing research sector rows: "
            f"{missing_research_symbols[:20]}"
        )

    if missing_production_symbols:
        missing_research_columns = (
            set(production_schema_columns) - set(research_sector_frame.columns)
        )
        if missing_research_columns:
            raise ValueError(
                "research sector frame cannot supply production schema columns: "
                f"{sorted(missing_research_columns)}"
            )

    removed_sector_symbols = sorted(
        symbol_name
        for symbol_name in production_sector_row_by_ticker
        if symbol_name not in production_symbol_set
    )

    output_records: list[dict[str, object]] = []
    for symbol_name in normalized_production_symbols:
        if symbol_name in production_sector_row_by_ticker:
            row_index = production_sector_row_by_ticker[symbol_name]
            selected_row = production_sector_frame.iloc[
                row_index,
            ][production_schema_columns].copy()
        else:
            row_index = research_sector_row_by_ticker[symbol_name]
            selected_row = research_sector_frame.iloc[
                row_index,
            ][production_schema_columns].copy()
        selected_row["ticker"] = symbol_name
        output_records.append(selected_row.to_dict())

    output_sector_frame = pandas.DataFrame(
        output_records,
        columns=production_schema_columns,
    )
    validate_production_ff12_sector_contract(
        output_sector_frame,
        normalized_production_symbols,
    )
    return ProductionFf12PromotionBuildResult(
        sector_frame=output_sector_frame,
        appended_symbols=missing_production_symbols,
        removed_sector_symbols=removed_sector_symbols,
    )


def _write_sector_outputs_to_temporary_directory(
    *,
    temporary_directory: Path,
    sector_frame: pandas.DataFrame,
) -> dict[Path, str]:
    """Write production sector outputs under a temporary directory."""

    temporary_parquet_path = temporary_directory / PRODUCTION_SECTOR_PARQUET_FILE_NAME
    temporary_csv_path = temporary_directory / PRODUCTION_SECTOR_CSV_FILE_NAME
    sector_frame.to_parquet(temporary_parquet_path, index=False)
    sector_frame.to_csv(temporary_csv_path, index=False)
    return {
        temporary_parquet_path: PRODUCTION_SECTOR_PARQUET_FILE_NAME,
        temporary_csv_path: PRODUCTION_SECTOR_CSV_FILE_NAME,
    }


def _replace_outputs_atomically(
    *,
    data_directory: Path,
    replacement_paths: dict[Path, str],
) -> None:
    """Replace production sector outputs using same-filesystem atomic swaps."""

    for temporary_path, final_file_name in replacement_paths.items():
        final_path = data_directory / final_file_name
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_path, final_path)


def sync_production_ff12_sector(
    *,
    data_directory: Path = DATA_DIRECTORY,
    publish_outputs: bool = True,
) -> ProductionFf12PromotionReport:
    """Append promoted research FF12 rows and optionally publish outputs."""

    resolved_data_directory = Path(data_directory)
    paths = ProductionFf12PromotionPaths(resolved_data_directory)
    production_symbols = load_production_symbols(paths.production_symbols_path)
    production_sector_frame, production_sector_source_path = (
        _read_sector_frame_from_pair(
            parquet_path=paths.production_sector_parquet_path,
            csv_path=paths.production_sector_csv_path,
            source_label="production",
        )
    )
    research_sector_frame, research_sector_source_path = _read_sector_frame_from_pair(
        parquet_path=paths.research_sector_parquet_path,
        csv_path=paths.research_sector_csv_path,
        source_label="research",
    )

    build_result = build_promoted_production_ff12_sector_frame(
        production_symbols=production_symbols,
        production_sector_frame=production_sector_frame,
        research_sector_frame=research_sector_frame,
    )

    if publish_outputs:
        resolved_data_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=resolved_data_directory,
            prefix=".production_ff12_promotion_",
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            replacement_paths = _write_sector_outputs_to_temporary_directory(
                temporary_directory=temporary_directory,
                sector_frame=build_result.sector_frame,
            )
            _replace_outputs_atomically(
                data_directory=resolved_data_directory,
                replacement_paths=replacement_paths,
            )

    LOGGER.info(
        "Production FF12 promotion sync %s: %s sector rows from %s and %s",
        "published" if publish_outputs else "validated",
        len(build_result.sector_frame),
        production_sector_source_path,
        research_sector_source_path,
    )
    return ProductionFf12PromotionReport(
        published=publish_outputs,
        production_symbol_count=len(production_symbols),
        original_sector_row_count=len(production_sector_frame),
        final_sector_row_count=len(build_result.sector_frame),
        appended_symbols=build_result.appended_symbols,
        removed_sector_symbols=build_result.removed_sector_symbols,
        ff12_source_counts=_count_values(build_result.sector_frame, "ff12_source"),
        output_paths={
            "production_symbols": str(paths.production_symbols_path),
            "production_sector_parquet": str(paths.production_sector_parquet_path),
            "production_sector_csv": str(paths.production_sector_csv_path),
            "research_sector_parquet": str(paths.research_sector_parquet_path),
            "research_sector_csv": str(paths.research_sector_csv_path),
        },
    )
