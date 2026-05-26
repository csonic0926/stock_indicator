"""Atomic research-universe and FF12 sector contract builder.

This module owns the research universe refresh: fetch the SEC ticker list,
rebuild the auditable universe decisions, apply sticky policy/quarantine
layers, rebuild FF12 coverage, validate the contract, and atomically publish
``research_new_symbols.txt`` plus ``research_new_symbols_with_sector`` outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

import pandas

from stock_indicator.sector_pipeline.config import (
    DATA_DIRECTORY,
    LAST_RUN_CONFIG_PATH,
    SIC_TO_FAMA_FRENCH_MAPPING_PATH,
)
from stock_indicator.sector_pipeline.ff_mapping import (
    attach_fama_french_groups,
    build_classification_lookup,
    load_fama_french_mapping,
)
from stock_indicator.sector_pipeline.sec_api import (
    fetch_company_ticker_exchange_table,
    map_tickers_to_central_index_and_classification,
)
from stock_indicator.sector_pipeline.secondary_classification import (
    apply_secondary_classifications,
)
from stock_indicator.sector_pipeline.utils import (
    normalize_ticker_symbol,
    save_json_file,
)
from stock_indicator.symbols import (
    build_symbol_hard_filter_audit_frame,
    normalize_security_title,
    normalize_symbol_for_cache,
)

LOGGER = logging.getLogger(__name__)

SYMBOL_HARD_FILTER_AUDIT_FILE_NAME = "symbol_hard_filter_audit.csv"
SYMBOLS_HARD_FILTERED_FILE_NAME = "symbols_hard_filtered_from_sec.txt"
SYMBOL_SECOND_LAYER_CANDIDATE_FILE_NAME = "symbol_second_layer_candidate_audit.csv"
SYMBOL_LLM_CLASSIFICATION_FILE_NAME = "symbol_universe_llm_classification.csv"
SYMBOL_POLICY_OVERRIDE_FILE_NAME = "symbol_universe_policy_overrides.csv"
RUNTIME_GUARDRAIL_LLM_FILE_NAME = "runtime_guardrail_conflict_llm_classification.csv"
SYMBOL_PRICE_QUARANTINE_FILE_NAME = "symbols_price_source_quarantine.csv"
SYMBOL_TRADABILITY_GATE_AUDIT_FILE_NAME = "symbol_tradability_gate_audit.csv"
PRICE_HISTORY_PREPARE_AUDIT_FILE_NAME = "stock_data_2010_yf_clean_prepare_audit.csv"
PRICE_HISTORY_DIRECTORY_NAME = "stock_data_2010_yf_clean"
SYMBOL_HARD_PLUS_LLM_AUDIT_FILE_NAME = "symbol_universe_hard_plus_llm_audit.csv"
SYMBOL_HARD_PLUS_LLM_FILE_NAME = "symbols_hard_plus_llm_from_sec.txt"
SYMBOL_FINAL_AUDIT_FILE_NAME = "symbol_universe_final_audit.csv"
SYMBOL_CONTRACT_FILE_NAME = "research_new_symbols.txt"
SYMBOL_PRICE_SOURCE_USABLE_FILE_NAME = "symbols_price_source_usable.txt"
SECTOR_PARQUET_FILE_NAME = "research_new_symbols_with_sector.parquet"
SECTOR_CSV_FILE_NAME = "research_new_symbols_with_sector.csv"

FINAL_AUDIT_COLUMNS = [
    "symbol",
    "sec_title",
    "sec_exchange",
    "final_decision",
    "decision_source",
    "hard_filter_reason",
    "llm_decision",
    "semantic_type",
    "confidence",
    "reason",
    "price_source_status",
    "tradability_status",
    "tradability_reason",
    "price_history_rows",
    "price_history_first_date",
    "price_history_last_date",
]

TRUSTED_SEC_EXCHANGES = {
    "nasdaq",
    "nyse",
    "nyse american",
    "nyse arca",
    "cboe bzx",
}
TRADABILITY_MINIMUM_PRICE_HISTORY_ROWS = 252
TRADABILITY_MINIMUM_SYMBOL_AGE_DAYS = 365
MANUAL_INCLUDE_DECISION_SOURCE = "policy_override"

SECOND_LAYER_REASON_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "spac_acquisition_title",
        re.compile(
            r"\b(?:ACQUISITION|SPAC)\b",
            flags=re.IGNORECASE,
        ),
    ),
    ("bdc_vehicle", re.compile(r"\b(?:BDC|BUSINESS DEVELOPMENT COMPANY)\b")),
    ("fund_word", re.compile(r"\bFUND\b", flags=re.IGNORECASE)),
    ("trust_word", re.compile(r"\bTRUST\b", flags=re.IGNORECASE)),
    (
        "municipal_or_clo_vehicle",
        re.compile(
            r"\b(?:MUNICIPAL|MUNI|CLO|COLLATERALIZED LOAN)\b",
            flags=re.IGNORECASE,
        ),
    ),
    ("royalty_trust", re.compile(r"\bROYALTY\s+TRUST\b", flags=re.IGNORECASE)),
    (
        "note_or_debt_abbreviation",
        re.compile(r"\b(?:DEBT|NOTE|NOTES|BOND|BONDS|DEBENTURE)\b", flags=re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class UniversePipelinePaths:
    """Filesystem paths used by the universe pipeline."""

    data_directory: Path

    @property
    def hard_filter_audit_path(self) -> Path:
        """Return the hard-filter audit CSV path."""

        return self.data_directory / SYMBOL_HARD_FILTER_AUDIT_FILE_NAME

    @property
    def hard_filtered_symbols_path(self) -> Path:
        """Return the hard-filtered symbol list path."""

        return self.data_directory / SYMBOLS_HARD_FILTERED_FILE_NAME

    @property
    def second_layer_candidate_path(self) -> Path:
        """Return the second-layer candidate audit path."""

        return self.data_directory / SYMBOL_SECOND_LAYER_CANDIDATE_FILE_NAME

    @property
    def llm_classification_path(self) -> Path:
        """Return the second-layer LLM classification cache path."""

        return self.data_directory / SYMBOL_LLM_CLASSIFICATION_FILE_NAME

    @property
    def policy_override_path(self) -> Path:
        """Return the policy override CSV path."""

        return self.data_directory / SYMBOL_POLICY_OVERRIDE_FILE_NAME

    @property
    def runtime_guardrail_llm_path(self) -> Path:
        """Return the one-shot runtime guardrail LLM classification path."""

        return self.data_directory / RUNTIME_GUARDRAIL_LLM_FILE_NAME

    @property
    def price_quarantine_path(self) -> Path:
        """Return the sticky Yahoo price-source quarantine CSV path."""

        return self.data_directory / SYMBOL_PRICE_QUARANTINE_FILE_NAME

    @property
    def tradability_gate_audit_path(self) -> Path:
        """Return the tradability and maturity gate audit CSV path."""

        return self.data_directory / SYMBOL_TRADABILITY_GATE_AUDIT_FILE_NAME

    @property
    def price_history_prepare_audit_path(self) -> Path:
        """Return the local long-history preparation audit CSV path."""

        return self.data_directory / PRICE_HISTORY_PREPARE_AUDIT_FILE_NAME

    @property
    def price_history_directory_path(self) -> Path:
        """Return the local long-history price directory used for maturity checks."""

        return self.data_directory / PRICE_HISTORY_DIRECTORY_NAME

    @property
    def hard_plus_llm_audit_path(self) -> Path:
        """Return the hard-filter plus LLM audit path."""

        return self.data_directory / SYMBOL_HARD_PLUS_LLM_AUDIT_FILE_NAME

    @property
    def hard_plus_llm_symbols_path(self) -> Path:
        """Return the symbol list after hard-filter, LLM, policy, and guardrail layers."""

        return self.data_directory / SYMBOL_HARD_PLUS_LLM_FILE_NAME

    @property
    def final_audit_path(self) -> Path:
        """Return the final universe audit path."""

        return self.data_directory / SYMBOL_FINAL_AUDIT_FILE_NAME

    @property
    def symbols_path(self) -> Path:
        """Return the final tradable universe contract path."""

        return self.data_directory / SYMBOL_CONTRACT_FILE_NAME

    @property
    def price_source_usable_symbols_path(self) -> Path:
        """Return the price-source usable symbol mirror path."""

        return self.data_directory / SYMBOL_PRICE_SOURCE_USABLE_FILE_NAME

    @property
    def sector_parquet_path(self) -> Path:
        """Return the final sector parquet path."""

        return self.data_directory / SECTOR_PARQUET_FILE_NAME

    @property
    def sector_csv_path(self) -> Path:
        """Return the final sector CSV path."""

        return self.data_directory / SECTOR_CSV_FILE_NAME


@dataclass(frozen=True)
class UniversePipelineReport:
    """Summary of a completed universe pipeline run."""

    published: bool
    final_symbol_count: int
    current_symbol_count: int
    added_symbols: list[str]
    removed_symbols: list[str]
    sector_row_count: int
    hard_filter_decision_counts: dict[str, int]
    hard_plus_decision_counts: dict[str, int]
    final_decision_counts: dict[str, int]
    decision_source_counts: dict[str, int]
    ff12_source_counts: dict[str, int]
    price_quarantine_count: int
    missing_llm_classification_count: int
    title_changed_llm_classification_count: int
    output_paths: dict[str, str]
    tradability_gate_exclusion_count: int = 0
    tradability_missing_price_history_count: int = 0
    tradability_immature_price_history_count: int = 0
    tradability_untrusted_exchange_count: int = 0

    def to_lines(self) -> list[str]:
        """Return a compact human-readable report."""

        action_label = "completed" if self.published else "dry run completed"
        added_sample = ", ".join(self.added_symbols[:20]) or "none"
        removed_sample = ", ".join(self.removed_symbols[:20]) or "none"
        return [
            f"Universe pipeline {action_label}",
            f"symbols: {self.final_symbol_count}",
            f"current symbols: {self.current_symbol_count}",
            f"added symbols: {len(self.added_symbols)} ({added_sample})",
            f"removed symbols: {len(self.removed_symbols)} ({removed_sample})",
            f"sector rows: {self.sector_row_count}",
            f"ff12 sources: {_format_count_mapping(self.ff12_source_counts)}",
            f"final decisions: {_format_count_mapping(self.final_decision_counts)}",
            f"decision sources: {_format_count_mapping(self.decision_source_counts)}",
            f"price quarantined: {self.price_quarantine_count}",
            f"tradability gate excluded: {self.tradability_gate_exclusion_count}",
            "tradability missing price history: "
            f"{self.tradability_missing_price_history_count}",
            "tradability immature price history: "
            f"{self.tradability_immature_price_history_count}",
            "tradability untrusted exchange: "
            f"{self.tradability_untrusted_exchange_count}",
            f"missing LLM classifications quarantined: {self.missing_llm_classification_count}",
            "title-changed LLM classifications quarantined: "
            f"{self.title_changed_llm_classification_count}",
        ]


SectorFrameBuilder = Callable[
    [list[str], pandas.DataFrame, str | Path],
    pandas.DataFrame,
]


@dataclass(frozen=True)
class PriceHistoryMetadata:
    """Minimum local price-history facts needed by the tradability gate."""

    row_count: int
    first_date: str
    last_date: str


def _format_count_mapping(count_mapping: dict[str, int]) -> str:
    """Return a deterministic compact count mapping."""

    if not count_mapping:
        return "{}"
    count_parts = [
        f"{key}={count_mapping[key]}"
        for key in sorted(count_mapping)
    ]
    return ", ".join(count_parts)


def _load_csv_frame(
    csv_path: Path,
    required_columns: set[str] | None = None,
) -> pandas.DataFrame:
    """Load a CSV audit file while preserving real ticker symbols such as ``NA``."""

    if not csv_path.exists():
        return pandas.DataFrame(columns=sorted(required_columns or set()))
    try:
        loaded_frame = pandas.read_csv(csv_path, keep_default_na=False).fillna("")
    except (OSError, pandas.errors.ParserError) as read_error:
        raise ValueError(f"Could not read {csv_path}: {read_error}") from read_error
    if required_columns is None:
        return loaded_frame
    missing_columns = required_columns - set(loaded_frame.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing_columns)}")
    return loaded_frame


def _load_existing_symbols(symbols_path: Path) -> list[str]:
    """Return symbols currently published in ``symbols_path``."""

    if not symbols_path.exists():
        return []
    return sorted(
        {
            normalize_symbol_for_cache(symbol_text)
            for symbol_text in symbols_path.read_text(encoding="utf-8").splitlines()
            if normalize_symbol_for_cache(symbol_text)
        }
    )


def _normalize_company_ticker_table(
    company_ticker_table: pandas.DataFrame,
) -> pandas.DataFrame:
    """Return SEC ticker rows with normalized ticker, CIK, and title columns."""

    if "ticker" not in company_ticker_table.columns:
        raise ValueError("SEC company ticker table must contain a 'ticker' column")
    normalized_table = company_ticker_table.copy()
    normalized_table["ticker"] = normalized_table["ticker"].map(
        normalize_symbol_for_cache
    )
    if "title" not in normalized_table.columns:
        normalized_table["title"] = ""
    normalized_table["title"] = normalized_table["title"].map(normalize_security_title)
    if "cik" in normalized_table.columns:
        normalized_table["cik"] = pandas.to_numeric(
            normalized_table["cik"],
            errors="coerce",
        )
    else:
        normalized_table["cik"] = pandas.NA
    if "exchange" not in normalized_table.columns:
        normalized_table["exchange"] = ""
    normalized_table["exchange"] = normalized_table["exchange"].map(
        _normalize_exchange_name
    )
    normalized_table = normalized_table.loc[normalized_table["ticker"] != ""].copy()
    return normalized_table.drop_duplicates(subset=["ticker"], keep="first")


def _normalize_exchange_name(exchange_name: Any) -> str:
    """Return a normalized exchange name for deterministic tradability checks."""

    if exchange_name is None:
        return ""
    try:
        if pandas.isna(exchange_name):
            return ""
    except TypeError:
        return ""
    return re.sub(r"\s+", " ", str(exchange_name).strip()).lower()


def _build_exchange_by_symbol(
    company_ticker_table: pandas.DataFrame,
) -> dict[str, str]:
    """Return normalized SEC exchange names keyed by symbol."""

    if "ticker" not in company_ticker_table.columns:
        return {}
    exchange_by_symbol: dict[str, str] = {}
    for company_ticker_record in company_ticker_table.to_dict("records"):
        symbol_name = normalize_symbol_for_cache(company_ticker_record.get("ticker"))
        if not symbol_name:
            continue
        exchange_by_symbol[symbol_name] = _normalize_exchange_name(
            company_ticker_record.get("exchange", "")
        )
    return exchange_by_symbol


def _find_second_layer_reasons(symbol_name: str, security_title: str) -> list[str]:
    """Return semantic-review reasons for an otherwise hard-included symbol."""

    return [
        reason_name
        for reason_name, reason_pattern in SECOND_LAYER_REASON_PATTERNS
        if reason_pattern.search(security_title)
    ]


def build_second_layer_candidate_audit_frame(
    hard_filter_audit_frame: pandas.DataFrame,
) -> pandas.DataFrame:
    """Return hard-included symbols that require sticky LLM semantic review."""

    required_columns = {"symbol", "sec_title", "hard_filter_decision"}
    missing_columns = required_columns - set(hard_filter_audit_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Hard-filter audit missing columns: {sorted(missing_columns)}"
        )

    candidate_records: list[dict[str, str]] = []
    for audit_record in hard_filter_audit_frame.to_dict("records"):
        hard_filter_decision = str(audit_record["hard_filter_decision"])
        if hard_filter_decision != "include":
            continue
        symbol_name = normalize_symbol_for_cache(audit_record.get("symbol"))
        security_title = normalize_security_title(audit_record.get("sec_title"))
        second_layer_reasons = _find_second_layer_reasons(symbol_name, security_title)
        if not second_layer_reasons:
            continue
        candidate_records.append(
            {
                "symbol": symbol_name,
                "sec_title": security_title,
                "hard_filter_decision": hard_filter_decision,
                "second_layer_reasons": "|".join(second_layer_reasons),
            }
        )
    return pandas.DataFrame(
        candidate_records,
        columns=[
            "symbol",
            "sec_title",
            "hard_filter_decision",
            "second_layer_reasons",
        ],
    )


def _load_policy_overrides(policy_override_path: Path) -> list[dict[str, str]]:
    """Load design-time universe policy overrides."""

    required_columns = {
        "match_field",
        "match_value",
        "override_decision",
        "override_semantic_type",
        "override_reason",
    }
    policy_override_frame = _load_csv_frame(policy_override_path, required_columns)
    return [
        {column_name: str(value) for column_name, value in policy_record.items()}
        for policy_record in policy_override_frame.to_dict("records")
    ]


def _load_runtime_guardrail_classifications(
    runtime_guardrail_llm_path: Path,
) -> dict[str, dict[str, str]]:
    """Load reviewed one-shot runtime guardrail conflict decisions."""

    required_columns = {"symbol", "decision", "semantic_type", "confidence", "reason"}
    classification_frame = _load_csv_frame(runtime_guardrail_llm_path, required_columns)
    classification_by_symbol: dict[str, dict[str, str]] = {}
    for classification_record in classification_frame.to_dict("records"):
        symbol_name = normalize_symbol_for_cache(classification_record.get("symbol"))
        if not symbol_name:
            continue
        classification_by_symbol[symbol_name] = {
            "decision": str(classification_record["decision"]).strip().lower(),
            "semantic_type": str(classification_record["semantic_type"]),
            "confidence": str(classification_record["confidence"]),
            "reason": str(classification_record["reason"]),
        }
    return classification_by_symbol


def _load_price_quarantine_symbols(
    price_quarantine_path: Path,
) -> dict[str, str]:
    """Load sticky Yahoo price-source quarantine statuses by symbol."""

    if not price_quarantine_path.exists():
        return {}
    quarantine_frame = _load_csv_frame(price_quarantine_path, {"symbol"})
    quarantine_by_symbol: dict[str, str] = {}
    for quarantine_record in quarantine_frame.to_dict("records"):
        symbol_name = normalize_symbol_for_cache(quarantine_record.get("symbol"))
        if not symbol_name:
            continue
        quarantine_status = str(quarantine_record.get("status", "")).strip()
        quarantine_by_symbol[symbol_name] = quarantine_status or "price_quarantine"
    return quarantine_by_symbol


def _parse_integer_value(raw_value: Any) -> int | None:
    """Return ``raw_value`` as an integer when possible."""

    try:
        if raw_value is None or pandas.isna(raw_value):
            return None
    except TypeError:
        return None
    try:
        return int(float(str(raw_value).strip()))
    except (TypeError, ValueError):
        return None


def _normalize_iso_date(raw_value: Any) -> str:
    """Return an ISO date string or an empty string."""

    try:
        if raw_value is None or pandas.isna(raw_value):
            return ""
    except TypeError:
        return ""
    timestamp_value = pandas.to_datetime(raw_value, errors="coerce")
    if pandas.isna(timestamp_value):
        return ""
    return pandas.Timestamp(timestamp_value).date().isoformat()


def _read_price_history_metadata_from_csv(
    price_history_path: Path,
) -> PriceHistoryMetadata | None:
    """Read one local price CSV and return compact history metadata."""

    if not price_history_path.exists():
        return None
    try:
        price_date_frame = pandas.read_csv(
            price_history_path,
            usecols=[0],
            parse_dates=[0],
        )
    except (
        OSError,
        pandas.errors.EmptyDataError,
        pandas.errors.ParserError,
        ValueError,
    ):
        return None
    if price_date_frame.empty:
        return None
    date_series = pandas.to_datetime(price_date_frame.iloc[:, 0], errors="coerce")
    date_series = date_series.dropna()
    if date_series.empty:
        return None
    return PriceHistoryMetadata(
        row_count=int(len(date_series)),
        first_date=pandas.Timestamp(date_series.min()).date().isoformat(),
        last_date=pandas.Timestamp(date_series.max()).date().isoformat(),
    )


def _load_price_history_metadata_from_prepare_audit(
    price_history_prepare_audit_path: Path,
) -> dict[str, PriceHistoryMetadata]:
    """Load price-history metadata from the long-history preparation audit."""

    required_columns = {
        "symbol",
        "row_count_final",
        "first_date_final",
        "last_date_final",
    }
    if not price_history_prepare_audit_path.exists():
        return {}
    audit_frame = _load_csv_frame(
        price_history_prepare_audit_path,
        required_columns,
    )
    metadata_by_symbol: dict[str, PriceHistoryMetadata] = {}
    for audit_record in audit_frame.to_dict("records"):
        symbol_name = normalize_symbol_for_cache(audit_record.get("symbol"))
        row_count = _parse_integer_value(audit_record.get("row_count_final"))
        if not symbol_name or row_count is None:
            continue
        metadata_by_symbol[symbol_name] = PriceHistoryMetadata(
            row_count=row_count,
            first_date=_normalize_iso_date(audit_record.get("first_date_final")),
            last_date=_normalize_iso_date(audit_record.get("last_date_final")),
        )
    return metadata_by_symbol


def load_price_history_metadata_for_symbols(
    symbol_names: list[str],
    *,
    price_history_prepare_audit_path: Path,
    price_history_directory_path: Path,
) -> dict[str, PriceHistoryMetadata]:
    """Return local price-history metadata for the requested symbols.

    The preparation audit is preferred because it is cheap and auditable.  CSVs
    are inspected only for symbols missing from that audit so the cron pipeline
    can admit a symbol after an operator backfills a local price file.
    """

    metadata_by_symbol = _load_price_history_metadata_from_prepare_audit(
        price_history_prepare_audit_path
    )
    for symbol_name in symbol_names:
        normalized_symbol = normalize_symbol_for_cache(symbol_name)
        if not normalized_symbol or normalized_symbol in metadata_by_symbol:
            continue
        price_history_path = price_history_directory_path / f"{normalized_symbol}.csv"
        price_metadata = _read_price_history_metadata_from_csv(price_history_path)
        if price_metadata is not None:
            metadata_by_symbol[normalized_symbol] = price_metadata
    return metadata_by_symbol


def _load_llm_classifications(
    llm_classification_path: Path,
) -> dict[str, dict[str, str]]:
    """Load sticky second-layer LLM classifications by symbol."""

    required_columns = {
        "symbol",
        "sec_title",
        "decision",
        "semantic_type",
        "confidence",
        "reason",
    }
    classification_frame = _load_csv_frame(llm_classification_path, required_columns)
    classification_by_symbol: dict[str, dict[str, str]] = {}
    for classification_record in classification_frame.to_dict("records"):
        symbol_name = normalize_symbol_for_cache(classification_record.get("symbol"))
        if not symbol_name:
            continue
        classification_by_symbol[symbol_name] = {
            "sec_title": normalize_security_title(classification_record.get("sec_title")),
            "decision": str(classification_record["decision"]).strip().lower(),
            "semantic_type": str(classification_record["semantic_type"]),
            "confidence": str(classification_record["confidence"]),
            "reason": str(classification_record["reason"]),
        }
    return classification_by_symbol


def _apply_policy_override(
    audit_record: dict[str, Any],
    policy_overrides: list[dict[str, str]],
) -> dict[str, Any]:
    """Apply the first matching policy override to an audit record."""

    updated_record = audit_record.copy()
    for policy_override in policy_overrides:
        match_field = policy_override["match_field"]
        match_value = policy_override["match_value"]
        if str(updated_record.get(match_field, "")) != match_value:
            continue
        updated_record["final_decision"] = policy_override["override_decision"]
        updated_record["decision_source"] = "policy_override"
        updated_record["semantic_type"] = policy_override["override_semantic_type"]
        updated_record["confidence"] = "policy"
        updated_record["reason"] = policy_override["override_reason"]
        return updated_record
    return updated_record


def _apply_runtime_guardrail_classification(
    audit_record: dict[str, Any],
    runtime_guardrail_classifications: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Apply one-shot runtime guardrail LLM decisions to an audit record."""

    updated_record = audit_record.copy()
    symbol_name = normalize_symbol_for_cache(updated_record.get("symbol"))
    classification = runtime_guardrail_classifications.get(symbol_name)
    if classification is None:
        return updated_record
    updated_record["final_decision"] = classification["decision"]
    updated_record["decision_source"] = "runtime_guardrail_llm_review"
    updated_record["semantic_type"] = classification["semantic_type"]
    updated_record["confidence"] = classification["confidence"]
    updated_record["reason"] = classification["reason"]
    return updated_record


def build_hard_plus_llm_audit_frame(
    hard_filter_audit_frame: pandas.DataFrame,
    second_layer_candidate_frame: pandas.DataFrame,
    llm_classifications: dict[str, dict[str, str]],
    policy_overrides: list[dict[str, str]],
    runtime_guardrail_classifications: dict[str, dict[str, str]],
) -> pandas.DataFrame:
    """Return universe decisions before the sticky price-source quarantine layer."""

    second_layer_symbols = {
        normalize_symbol_for_cache(symbol_name)
        for symbol_name in second_layer_candidate_frame.get("symbol", pandas.Series(dtype=str))
    }

    final_records: list[dict[str, Any]] = []
    for hard_filter_record in hard_filter_audit_frame.to_dict("records"):
        symbol_name = normalize_symbol_for_cache(hard_filter_record.get("symbol"))
        security_title = normalize_security_title(hard_filter_record.get("sec_title"))
        hard_filter_decision = str(hard_filter_record.get("hard_filter_decision", ""))
        hard_filter_reason = str(hard_filter_record.get("hard_filter_reason", ""))

        if hard_filter_decision == "exclude":
            final_record: dict[str, Any] = {
                "symbol": symbol_name,
                "sec_title": security_title,
                "final_decision": "exclude",
                "decision_source": "hard_filter",
                "hard_filter_reason": hard_filter_reason,
                "llm_decision": "",
                "semantic_type": "",
                "confidence": "",
                "reason": hard_filter_reason,
                "price_source_status": "",
            }
        elif symbol_name in second_layer_symbols:
            llm_record = llm_classifications.get(symbol_name)
            if llm_record is None:
                final_record = {
                    "symbol": symbol_name,
                    "sec_title": security_title,
                    "final_decision": "quarantine",
                    "decision_source": "llm_second_layer_missing_cache",
                    "hard_filter_reason": "",
                    "llm_decision": "quarantine",
                    "semantic_type": "missing_llm_result",
                    "confidence": "low",
                    "reason": "Missing sticky LLM classification for second-layer candidate.",
                    "price_source_status": "",
                }
            elif llm_record.get("sec_title") and llm_record["sec_title"] != security_title:
                final_record = {
                    "symbol": symbol_name,
                    "sec_title": security_title,
                    "final_decision": "quarantine",
                    "decision_source": "llm_second_layer_title_changed",
                    "hard_filter_reason": "",
                    "llm_decision": "quarantine",
                    "semantic_type": "stale_llm_result",
                    "confidence": "low",
                    "reason": (
                        "SEC title changed since sticky LLM classification; "
                        "requires refreshed semantic review."
                    ),
                    "price_source_status": "",
                }
            else:
                llm_decision = str(llm_record.get("decision", "quarantine"))
                final_record = {
                    "symbol": symbol_name,
                    "sec_title": security_title,
                    "final_decision": llm_decision,
                    "decision_source": "llm_second_layer",
                    "hard_filter_reason": "",
                    "llm_decision": llm_decision,
                    "semantic_type": str(llm_record.get("semantic_type", "")),
                    "confidence": str(llm_record.get("confidence", "low")),
                    "reason": str(llm_record.get("reason", "")),
                    "price_source_status": "",
                }
        else:
            final_record = {
                "symbol": symbol_name,
                "sec_title": security_title,
                "final_decision": "include",
                "decision_source": "hard_filter_clean_pass",
                "hard_filter_reason": "",
                "llm_decision": "",
                "semantic_type": "not_second_layer_candidate",
                "confidence": "rule_pass",
                "reason": "Passed deterministic hard filter and no second-layer flag.",
                "price_source_status": "",
            }

        final_record = _apply_runtime_guardrail_classification(
            final_record,
            runtime_guardrail_classifications,
        )
        final_record = _apply_policy_override(final_record, policy_overrides)
        final_records.append(final_record)

    return pandas.DataFrame(final_records, columns=FINAL_AUDIT_COLUMNS)


def apply_price_source_quarantine(
    hard_plus_llm_audit_frame: pandas.DataFrame,
    price_quarantine_by_symbol: dict[str, str],
) -> pandas.DataFrame:
    """Return final decisions after removing sticky Yahoo-unpriceable symbols."""

    final_audit_frame = hard_plus_llm_audit_frame.copy()
    if "price_source_status" not in final_audit_frame.columns:
        final_audit_frame["price_source_status"] = ""

    for row_index, audit_record in final_audit_frame.iterrows():
        if str(audit_record.get("final_decision", "")) != "include":
            continue
        symbol_name = normalize_symbol_for_cache(audit_record.get("symbol"))
        quarantine_status = price_quarantine_by_symbol.get(symbol_name)
        if quarantine_status is None:
            continue
        final_audit_frame.loc[row_index, "final_decision"] = "exclude"
        final_audit_frame.loc[row_index, "decision_source"] = "price_source_quarantine"
        final_audit_frame.loc[row_index, "confidence"] = "sticky_price_audit"
        final_audit_frame.loc[row_index, "price_source_status"] = quarantine_status
        final_audit_frame.loc[
            row_index,
            "reason",
        ] = f"Sticky Yahoo price-source quarantine: {quarantine_status}."

    return final_audit_frame[FINAL_AUDIT_COLUMNS]


def _price_history_is_mature(
    price_metadata: PriceHistoryMetadata,
    *,
    minimum_price_history_rows: int,
    minimum_symbol_age_days: int,
) -> tuple[bool, str]:
    """Return whether local price history is mature enough for the universe."""

    if price_metadata.row_count < minimum_price_history_rows:
        return (
            False,
            f"price_history_rows_below_{minimum_price_history_rows}",
        )

    first_timestamp = pandas.to_datetime(price_metadata.first_date, errors="coerce")
    last_timestamp = pandas.to_datetime(price_metadata.last_date, errors="coerce")
    if pandas.isna(first_timestamp) or pandas.isna(last_timestamp):
        return False, "price_history_dates_missing"
    symbol_age_days = int((last_timestamp - first_timestamp).days)
    if symbol_age_days < minimum_symbol_age_days:
        return False, f"price_history_age_below_{minimum_symbol_age_days}_days"
    return True, "mature_price_history"


def apply_tradability_gate(
    price_source_audit_frame: pandas.DataFrame,
    *,
    company_ticker_table: pandas.DataFrame,
    price_history_metadata_by_symbol: dict[str, PriceHistoryMetadata],
    trusted_sec_exchanges: set[str] = TRUSTED_SEC_EXCHANGES,
    minimum_price_history_rows: int = TRADABILITY_MINIMUM_PRICE_HISTORY_ROWS,
    minimum_symbol_age_days: int = TRADABILITY_MINIMUM_SYMBOL_AGE_DAYS,
) -> pandas.DataFrame:
    """Exclude included symbols without trusted exchange and mature price history.

    This is the persistent version of the legacy ``2010_safe`` provenance gate:
    a ticker must be a reviewed include, listed on a trusted SEC exchange, and
    have enough local daily history before it enters the research universe.
    Manual policy includes bypass the gate because those are explicit research
    choices that remain auditable.
    """

    final_audit_frame = price_source_audit_frame.copy()
    for column_name in FINAL_AUDIT_COLUMNS:
        if column_name not in final_audit_frame.columns:
            final_audit_frame[column_name] = ""
    audit_text_column_names = (
        "sec_exchange",
        "tradability_status",
        "tradability_reason",
        "price_history_first_date",
        "price_history_last_date",
    )
    for audit_text_column_name in audit_text_column_names:
        audit_text_column = final_audit_frame[audit_text_column_name]
        final_audit_frame[audit_text_column_name] = (
            audit_text_column.where(audit_text_column.notna(), "").astype(object)
        )
    price_history_rows_column = final_audit_frame["price_history_rows"]
    final_audit_frame["price_history_rows"] = (
        price_history_rows_column.where(price_history_rows_column.notna(), "")
        .astype(object)
    )

    exchange_by_symbol = _build_exchange_by_symbol(company_ticker_table)
    trusted_exchange_set = {
        _normalize_exchange_name(exchange_name)
        for exchange_name in trusted_sec_exchanges
    }

    for row_index, audit_record in final_audit_frame.iterrows():
        symbol_name = normalize_symbol_for_cache(audit_record.get("symbol"))
        exchange_name = exchange_by_symbol.get(symbol_name, "")
        price_metadata = price_history_metadata_by_symbol.get(symbol_name)
        final_audit_frame.loc[row_index, "sec_exchange"] = exchange_name
        if price_metadata is not None:
            final_audit_frame.loc[row_index, "price_history_rows"] = (
                price_metadata.row_count
            )
            final_audit_frame.loc[row_index, "price_history_first_date"] = (
                price_metadata.first_date
            )
            final_audit_frame.loc[row_index, "price_history_last_date"] = (
                price_metadata.last_date
            )

        if str(audit_record.get("final_decision", "")) != "include":
            final_audit_frame.loc[row_index, "tradability_status"] = "not_applicable"
            continue

        if (
            str(audit_record.get("decision_source", ""))
            == MANUAL_INCLUDE_DECISION_SOURCE
        ):
            final_audit_frame.loc[row_index, "tradability_status"] = "manual_include"
            final_audit_frame.loc[
                row_index,
                "tradability_reason",
            ] = "Manual policy include bypasses automated tradability gate."
            continue

        if exchange_name not in trusted_exchange_set:
            final_audit_frame.loc[row_index, "final_decision"] = "exclude"
            final_audit_frame.loc[row_index, "decision_source"] = "tradability_gate"
            final_audit_frame.loc[row_index, "confidence"] = "tradability_gate"
            final_audit_frame.loc[row_index, "tradability_status"] = "exclude"
            final_audit_frame.loc[row_index, "tradability_reason"] = (
                "untrusted_or_missing_sec_exchange"
            )
            final_audit_frame.loc[row_index, "reason"] = (
                "Tradability gate: SEC exchange is missing or not a trusted "
                "primary exchange."
            )
            continue

        if price_metadata is None:
            final_audit_frame.loc[row_index, "final_decision"] = "exclude"
            final_audit_frame.loc[row_index, "decision_source"] = "tradability_gate"
            final_audit_frame.loc[row_index, "confidence"] = "tradability_gate"
            final_audit_frame.loc[row_index, "tradability_status"] = "exclude"
            final_audit_frame.loc[row_index, "tradability_reason"] = (
                "missing_price_history"
            )
            final_audit_frame.loc[row_index, "reason"] = (
                "Tradability gate: no local long-history price metadata."
            )
            continue

        price_history_is_mature, maturity_reason = _price_history_is_mature(
            price_metadata,
            minimum_price_history_rows=minimum_price_history_rows,
            minimum_symbol_age_days=minimum_symbol_age_days,
        )
        if not price_history_is_mature:
            final_audit_frame.loc[row_index, "final_decision"] = "exclude"
            final_audit_frame.loc[row_index, "decision_source"] = "tradability_gate"
            final_audit_frame.loc[row_index, "confidence"] = "tradability_gate"
            final_audit_frame.loc[row_index, "tradability_status"] = "exclude"
            final_audit_frame.loc[row_index, "tradability_reason"] = maturity_reason
            final_audit_frame.loc[row_index, "reason"] = (
                f"Tradability gate: {maturity_reason}."
            )
            continue

        final_audit_frame.loc[row_index, "tradability_status"] = "include"
        final_audit_frame.loc[row_index, "tradability_reason"] = (
            "trusted_exchange_and_mature_price_history"
        )

    return final_audit_frame[FINAL_AUDIT_COLUMNS]


def extract_included_symbols(audit_frame: pandas.DataFrame) -> list[str]:
    """Return sorted included symbols from a universe audit frame."""

    included_symbol_series = audit_frame.loc[
        audit_frame["final_decision"] == "include",
        "symbol",
    ]
    included_symbols = {
        normalize_symbol_for_cache(symbol_name)
        for symbol_name in included_symbol_series
        if normalize_symbol_for_cache(symbol_name)
    }
    return sorted(included_symbols)


def _write_symbol_list(symbols_to_write: list[str], output_path: Path) -> None:
    """Write newline-separated symbols to ``output_path``."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(symbols_to_write) + "\n", encoding="utf-8")


def _build_universe_frame_for_sector(
    final_symbols: list[str],
    company_ticker_table: pandas.DataFrame,
) -> pandas.DataFrame:
    """Return a ticker/CIK universe frame for sector classification."""

    normalized_company_table = _normalize_company_ticker_table(company_ticker_table)
    sec_mapping_frame = normalized_company_table[["ticker", "cik"]].copy()
    symbol_frame = pandas.DataFrame({"ticker": final_symbols})
    symbol_frame["ticker"] = symbol_frame["ticker"].map(normalize_ticker_symbol)
    universe_frame = symbol_frame.merge(sec_mapping_frame, on="ticker", how="left")
    return universe_frame.drop_duplicates(subset=["ticker"], keep="first")


def build_sector_classification_frame_for_symbols(
    final_symbols: list[str],
    company_ticker_table: pandas.DataFrame,
    mapping_source: str | Path = SIC_TO_FAMA_FRENCH_MAPPING_PATH,
) -> pandas.DataFrame:
    """Build sector classification rows for the final symbol contract."""

    universe_frame = _build_universe_frame_for_sector(final_symbols, company_ticker_table)
    ticker_mapping_frame = map_tickers_to_central_index_and_classification(
        universe_frame
    )
    mapping_frame = load_fama_french_mapping(mapping_source)
    lookup_frame = build_classification_lookup(mapping_frame)
    classified_frame = attach_fama_french_groups(ticker_mapping_frame, lookup_frame)
    classified_frame = apply_secondary_classifications(classified_frame)
    classified_frame["sic_desc"] = ""
    for optional_column_name in (
        "secondary_sector",
        "secondary_industry",
        "secondary_source",
        "secondary_reason",
    ):
        if optional_column_name not in classified_frame.columns:
            classified_frame[optional_column_name] = ""
    return classified_frame


def validate_sector_contract(
    sector_frame: pandas.DataFrame,
    final_symbols: list[str],
) -> None:
    """Raise when FF12 sector output does not cover the final symbol contract."""

    required_columns = {
        "ticker",
        "ff12",
        "ff12_source",
        "classification_confidence",
    }
    missing_columns = required_columns - set(sector_frame.columns)
    if missing_columns:
        raise ValueError(f"Sector frame missing columns: {sorted(missing_columns)}")

    normalized_sector_frame = sector_frame.copy()
    normalized_sector_frame["ticker"] = normalized_sector_frame["ticker"].map(
        normalize_ticker_symbol
    )
    duplicate_tickers = normalized_sector_frame.loc[
        normalized_sector_frame["ticker"].duplicated(),
        "ticker",
    ].tolist()
    if duplicate_tickers:
        raise ValueError(f"Sector frame has duplicate tickers: {duplicate_tickers[:10]}")

    expected_symbol_set = set(final_symbols)
    actual_symbol_set = set(normalized_sector_frame["ticker"].astype(str))
    missing_symbols = sorted(expected_symbol_set - actual_symbol_set)
    extra_symbols = sorted(actual_symbol_set - expected_symbol_set)
    if missing_symbols or extra_symbols:
        raise ValueError(
            "Sector frame does not match final universe: "
            f"missing={missing_symbols[:10]}, extra={extra_symbols[:10]}"
        )

    ff12_values = pandas.to_numeric(normalized_sector_frame["ff12"], errors="coerce")
    invalid_ff12_mask = ff12_values.isna() | ff12_values.lt(1) | ff12_values.gt(12)
    if invalid_ff12_mask.any():
        invalid_symbols = normalized_sector_frame.loc[
            invalid_ff12_mask,
            "ticker",
        ].tolist()
        raise ValueError(f"Sector frame has invalid FF12 values: {invalid_symbols[:10]}")

    confidence_series = (
        normalized_sector_frame["classification_confidence"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    low_confidence_symbols = normalized_sector_frame.loc[
        confidence_series == "low",
        "ticker",
    ].tolist()
    if low_confidence_symbols:
        raise ValueError(
            "Sector frame has low-confidence FF12 fallbacks: "
            f"{low_confidence_symbols[:10]}"
        )

    source_series = (
        normalized_sector_frame["ff12_source"].astype(str).str.strip().str.lower()
    )
    fallback_symbols = normalized_sector_frame.loc[
        source_series == "missing_sic_fallback",
        "ticker",
    ].tolist()
    if fallback_symbols:
        raise ValueError(
            "Sector frame has missing-SIC fallback rows: "
            f"{fallback_symbols[:10]}"
        )


def validate_symbol_count_drift(
    final_symbols: list[str],
    current_symbols: list[str],
    maximum_symbol_drop_ratio: float,
) -> None:
    """Raise when the proposed universe drops too far versus the current contract."""

    if not current_symbols:
        return
    if maximum_symbol_drop_ratio < 0 or maximum_symbol_drop_ratio >= 1:
        raise ValueError("maximum_symbol_drop_ratio must be >= 0 and < 1")

    current_symbol_count = len(current_symbols)
    final_symbol_count = len(final_symbols)
    allowed_drop_count = max(1, int(current_symbol_count * maximum_symbol_drop_ratio))
    actual_drop_count = current_symbol_count - final_symbol_count
    if actual_drop_count <= allowed_drop_count:
        return
    raise ValueError(
        "Proposed universe is too small versus current contract: "
        f"current={current_symbol_count}, proposed={final_symbol_count}, "
        f"drop={actual_drop_count}, allowed_drop={allowed_drop_count}"
    )


def _write_pipeline_outputs_to_directory(
    *,
    temporary_directory: Path,
    hard_filter_audit_frame: pandas.DataFrame,
    hard_filtered_symbols: list[str],
    second_layer_candidate_frame: pandas.DataFrame,
    hard_plus_llm_audit_frame: pandas.DataFrame,
    hard_plus_llm_symbols: list[str],
    tradability_gate_audit_frame: pandas.DataFrame,
    final_audit_frame: pandas.DataFrame,
    final_symbols: list[str],
    sector_frame: pandas.DataFrame,
) -> dict[Path, Path]:
    """Write all outputs under ``temporary_directory`` and return replacements."""

    temporary_paths = UniversePipelinePaths(temporary_directory)
    hard_filter_audit_frame.to_csv(
        temporary_paths.hard_filter_audit_path,
        index=False,
    )
    _write_symbol_list(hard_filtered_symbols, temporary_paths.hard_filtered_symbols_path)
    second_layer_candidate_frame.to_csv(
        temporary_paths.second_layer_candidate_path,
        index=False,
    )
    hard_plus_llm_audit_frame.to_csv(
        temporary_paths.hard_plus_llm_audit_path,
        index=False,
    )
    _write_symbol_list(
        hard_plus_llm_symbols,
        temporary_paths.hard_plus_llm_symbols_path,
    )
    tradability_gate_audit_frame.to_csv(
        temporary_paths.tradability_gate_audit_path,
        index=False,
    )
    final_audit_frame.to_csv(temporary_paths.final_audit_path, index=False)
    _write_symbol_list(final_symbols, temporary_paths.symbols_path)
    _write_symbol_list(final_symbols, temporary_paths.price_source_usable_symbols_path)
    sector_frame.to_parquet(temporary_paths.sector_parquet_path, index=False)
    sector_frame.to_csv(temporary_paths.sector_csv_path, index=False)

    final_paths = UniversePipelinePaths(DATA_DIRECTORY)
    return {
        temporary_paths.hard_filter_audit_path: final_paths.hard_filter_audit_path,
        temporary_paths.hard_filtered_symbols_path: final_paths.hard_filtered_symbols_path,
        temporary_paths.second_layer_candidate_path: final_paths.second_layer_candidate_path,
        temporary_paths.hard_plus_llm_audit_path: final_paths.hard_plus_llm_audit_path,
        temporary_paths.hard_plus_llm_symbols_path: final_paths.hard_plus_llm_symbols_path,
        temporary_paths.tradability_gate_audit_path: (
            final_paths.tradability_gate_audit_path
        ),
        temporary_paths.final_audit_path: final_paths.final_audit_path,
        temporary_paths.sector_parquet_path: final_paths.sector_parquet_path,
        temporary_paths.sector_csv_path: final_paths.sector_csv_path,
        temporary_paths.price_source_usable_symbols_path: (
            final_paths.price_source_usable_symbols_path
        ),
        temporary_paths.symbols_path: final_paths.symbols_path,
    }


def _replace_outputs_atomically(replacement_paths: dict[Path, Path]) -> None:
    """Replace output files one by one using atomic same-filesystem swaps."""

    for temporary_path, final_path in replacement_paths.items():
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_path, final_path)


def _count_values(frame: pandas.DataFrame, column_name: str) -> dict[str, int]:
    """Return value counts for ``column_name`` as a plain dictionary."""

    if column_name not in frame.columns:
        return {}
    count_series = frame[column_name].astype(str).value_counts()
    return {str(value): int(count) for value, count in count_series.items()}


def _write_sector_last_run_configuration(
    mapping_source: str | Path,
    paths: UniversePipelinePaths,
) -> None:
    """Keep the legacy sector update configuration pointed at research output."""

    save_json_file(
        {
            "mapping_source": str(mapping_source),
            "output": str(paths.sector_parquet_path),
            "universe_source": str(paths.symbols_path),
        },
        LAST_RUN_CONFIG_PATH,
    )


def run_universe_pipeline(
    *,
    data_directory: Path = DATA_DIRECTORY,
    mapping_source: str | Path = SIC_TO_FAMA_FRENCH_MAPPING_PATH,
    company_ticker_table: pandas.DataFrame | None = None,
    sector_frame_builder: SectorFrameBuilder | None = None,
    publish_outputs: bool = True,
    maximum_symbol_drop_ratio: float = 0.05,
) -> UniversePipelineReport:
    """Run the daily universe pipeline and optionally publish validated outputs."""

    resolved_data_directory = Path(data_directory)
    paths = UniversePipelinePaths(resolved_data_directory)
    sector_frame_builder = (
        sector_frame_builder or build_sector_classification_frame_for_symbols
    )
    resolved_data_directory.mkdir(parents=True, exist_ok=True)

    if company_ticker_table is None:
        company_ticker_table = fetch_company_ticker_exchange_table()
    normalized_company_table = _normalize_company_ticker_table(company_ticker_table)

    hard_filter_audit_frame = build_symbol_hard_filter_audit_frame(
        normalized_company_table
    )
    hard_filtered_symbols = sorted(
        hard_filter_audit_frame.loc[
            hard_filter_audit_frame["hard_filter_decision"] == "include",
            "symbol",
        ]
        .astype(str)
        .tolist()
    )
    second_layer_candidate_frame = build_second_layer_candidate_audit_frame(
        hard_filter_audit_frame
    )
    llm_classifications = _load_llm_classifications(paths.llm_classification_path)
    policy_overrides = _load_policy_overrides(paths.policy_override_path)
    runtime_guardrail_classifications = _load_runtime_guardrail_classifications(
        paths.runtime_guardrail_llm_path
    )
    price_quarantine_by_symbol = _load_price_quarantine_symbols(
        paths.price_quarantine_path
    )

    hard_plus_llm_audit_frame = build_hard_plus_llm_audit_frame(
        hard_filter_audit_frame,
        second_layer_candidate_frame,
        llm_classifications,
        policy_overrides,
        runtime_guardrail_classifications,
    )
    hard_plus_llm_symbols = extract_included_symbols(hard_plus_llm_audit_frame)
    price_source_audit_frame = apply_price_source_quarantine(
        hard_plus_llm_audit_frame,
        price_quarantine_by_symbol,
    )
    price_history_metadata_by_symbol = load_price_history_metadata_for_symbols(
        extract_included_symbols(price_source_audit_frame),
        price_history_prepare_audit_path=paths.price_history_prepare_audit_path,
        price_history_directory_path=paths.price_history_directory_path,
    )
    final_audit_frame = apply_tradability_gate(
        price_source_audit_frame,
        company_ticker_table=normalized_company_table,
        price_history_metadata_by_symbol=price_history_metadata_by_symbol,
    )
    tradability_gate_audit_frame = final_audit_frame.loc[
        final_audit_frame["tradability_status"].astype(str) != ""
    ].copy()
    final_symbols = extract_included_symbols(final_audit_frame)
    current_symbols = _load_existing_symbols(paths.symbols_path)
    validate_symbol_count_drift(
        final_symbols,
        current_symbols,
        maximum_symbol_drop_ratio,
    )
    sector_frame = sector_frame_builder(
        final_symbols,
        normalized_company_table,
        mapping_source,
    )
    validate_sector_contract(sector_frame, final_symbols)

    if publish_outputs:
        with tempfile.TemporaryDirectory(
            dir=resolved_data_directory,
            prefix=".universe_pipeline_",
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            replacement_paths = _write_pipeline_outputs_to_directory(
                temporary_directory=temporary_directory,
                hard_filter_audit_frame=hard_filter_audit_frame,
                hard_filtered_symbols=hard_filtered_symbols,
                second_layer_candidate_frame=second_layer_candidate_frame,
                hard_plus_llm_audit_frame=hard_plus_llm_audit_frame,
                hard_plus_llm_symbols=hard_plus_llm_symbols,
                tradability_gate_audit_frame=tradability_gate_audit_frame,
                final_audit_frame=final_audit_frame,
                final_symbols=final_symbols,
                sector_frame=sector_frame,
            )
            replacement_paths = {
                temporary_path: resolved_data_directory / final_path.name
                for temporary_path, final_path in replacement_paths.items()
            }
            _replace_outputs_atomically(replacement_paths)

    if publish_outputs and resolved_data_directory == DATA_DIRECTORY:
        _write_sector_last_run_configuration(mapping_source, paths)

    final_symbol_set = set(final_symbols)
    current_symbol_set = set(current_symbols)
    report = UniversePipelineReport(
        published=publish_outputs,
        final_symbol_count=len(final_symbols),
        current_symbol_count=len(current_symbols),
        added_symbols=sorted(final_symbol_set - current_symbol_set),
        removed_symbols=sorted(current_symbol_set - final_symbol_set),
        sector_row_count=len(sector_frame),
        hard_filter_decision_counts=_count_values(
            hard_filter_audit_frame,
            "hard_filter_decision",
        ),
        hard_plus_decision_counts=_count_values(
            hard_plus_llm_audit_frame,
            "final_decision",
        ),
        final_decision_counts=_count_values(final_audit_frame, "final_decision"),
        decision_source_counts=_count_values(final_audit_frame, "decision_source"),
        ff12_source_counts=_count_values(sector_frame, "ff12_source"),
        price_quarantine_count=int(
            (
                final_audit_frame["decision_source"].astype(str)
                == "price_source_quarantine"
            ).sum()
        ),
        missing_llm_classification_count=int(
            (
                hard_plus_llm_audit_frame["decision_source"].astype(str)
                == "llm_second_layer_missing_cache"
            ).sum()
        ),
        title_changed_llm_classification_count=int(
            (
                hard_plus_llm_audit_frame["decision_source"].astype(str)
                == "llm_second_layer_title_changed"
            ).sum()
        ),
        output_paths={
            "symbols": str(paths.symbols_path),
            "sector_parquet": str(paths.sector_parquet_path),
            "sector_csv": str(paths.sector_csv_path),
            "final_audit": str(paths.final_audit_path),
            "tradability_gate_audit": str(paths.tradability_gate_audit_path),
        },
        tradability_gate_exclusion_count=int(
            (
                final_audit_frame["decision_source"].astype(str)
                == "tradability_gate"
            ).sum()
        ),
        tradability_missing_price_history_count=int(
            (
                final_audit_frame["tradability_reason"].astype(str)
                == "missing_price_history"
            ).sum()
        ),
        tradability_immature_price_history_count=int(
            final_audit_frame["tradability_reason"]
            .astype(str)
            .str.startswith("price_history_")
            .sum()
        ),
        tradability_untrusted_exchange_count=int(
            (
                final_audit_frame["tradability_reason"].astype(str)
                == "untrusted_or_missing_sec_exchange"
            ).sum()
        ),
    )
    LOGGER.info(
        "Universe pipeline %s: %s symbols",
        "published" if publish_outputs else "dry-run validated",
        report.final_symbol_count,
    )
    return report
