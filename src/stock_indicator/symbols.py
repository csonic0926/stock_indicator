"""Utilities for maintaining a local cache of common-stock symbols.

The local cache is stored as a newline-separated text file at
``data/symbols.txt``. The source universe comes from the SEC company tickers
feed, then obvious non-common-stock instruments are filtered out before the
cache is written.
"""
# TODO: review

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas

LOGGER = logging.getLogger(__name__)

SYMBOL_CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "symbols.txt"
)

# Symbol representing the S&P 500 index.
SP500_SYMBOL = "^GSPC"

SEPARATED_NON_COMMON_STOCK_SYMBOL_PATTERN = re.compile(
    r"[.-](?:PR[A-Z]?|P[A-Z]|WT[A-Z]?|WS[A-Z]?|WTS?|RT|RIGHT|U|UN|R)$",
    flags=re.IGNORECASE,
)
NASDAQ_NON_COMMON_STOCK_SUFFIX_PATTERN = re.compile(
    r"^[A-Z]{4,5}(?:U|W|WS|WT|R|RT|P)$",
    flags=re.IGNORECASE,
)
OBVIOUS_NON_COMMON_STOCK_TITLE_PATTERN = re.compile(
    r"\b("
    r"ETF|ETN|EXCHANGE[- ]TRADED|"
    r"WARRANTS?|UNITS?|RIGHTS?|"
    r"PREFERRED|PREFERENCE|PFD|PRF|PREF|"
    r"NOTES?|BONDS?|DEBENTURES?|TREASURY"
    r")\b",
    flags=re.IGNORECASE,
)
OBVIOUS_ETF_LIKE_TRUST_TITLE_PATTERN = re.compile(
    r"(?:"
    r"\b(?:SPDR|ISHARES|PROSHARES|DIREXION|WISDOMTREE|VANECK|"
    r"GLOBAL X|ARK|BITWISE|GRAYSCALE)\b.*\b(?:TRUST|ETF|ETN)\b"
    r"|\bQQQ TRUST\b"
    r"|\b(?:PHYSICAL|GOLD|SILVER|BITCOIN|ETHER|ETHEREUM|COMMODITY|"
    r"CURRENCY)\b.*\bTRUST\b"
    r"|\bTRUST\b.*\b(?:PHYSICAL|GOLD|SILVER|BITCOIN|ETHER|ETHEREUM|"
    r"COMMODITY|CURRENCY)\b"
    r")",
    flags=re.IGNORECASE,
)


def normalize_symbol_for_cache(symbol_value: Any) -> str:
    """Return the local symbol format used by cached price files."""

    if symbol_value is None:
        return ""
    try:
        if pandas.isna(symbol_value):
            return ""
    except TypeError:
        return ""
    symbol_text = str(symbol_value).strip().upper()
    if not symbol_text or symbol_text == "NAN":
        return ""
    return re.sub(r"(?<=\w)-(?!$)", ".", symbol_text)


def normalize_security_title(security_title: Any) -> str:
    """Return a normalized SEC title string for rule-based inspection."""

    if security_title is None:
        return ""
    try:
        if pandas.isna(security_title):
            return ""
    except TypeError:
        return ""
    return re.sub(r"\s+", " ", str(security_title).strip().upper())


def identify_non_common_stock_reason(
    symbol_name: str,
    security_title: str,
) -> str | None:
    """Return the rule reason when a SEC ticker is not common-stock-like.

    The SEC company tickers feed is a good issuer/ticker source, but it is not
    a common-stock whitelist. These rules intentionally avoid FF12 or SIC so
    sector classification cannot remove valid operating companies.
    """

    normalized_symbol = normalize_symbol_for_cache(symbol_name)
    normalized_title = normalize_security_title(security_title)
    if not normalized_symbol:
        return "missing_symbol"
    if SEPARATED_NON_COMMON_STOCK_SYMBOL_PATTERN.search(normalized_symbol):
        return "non_common_symbol_suffix"
    if NASDAQ_NON_COMMON_STOCK_SUFFIX_PATTERN.fullmatch(normalized_symbol):
        return "nasdaq_non_common_suffix"
    if normalized_title and OBVIOUS_NON_COMMON_STOCK_TITLE_PATTERN.search(
        normalized_title
    ):
        return "obvious_non_common_title"
    if normalized_title and OBVIOUS_ETF_LIKE_TRUST_TITLE_PATTERN.search(
        normalized_title
    ):
        return "obvious_etf_like_trust_title"
    return None


def build_symbol_hard_filter_audit_frame(
    company_ticker_table: pandas.DataFrame,
) -> pandas.DataFrame:
    """Return per-symbol hard-filter decisions for SEC ticker records."""

    if "ticker" not in company_ticker_table.columns:
        raise ValueError("SEC company ticker table must contain a 'ticker' column")

    audit_records: list[dict[str, str]] = []
    for company_ticker_record in company_ticker_table.to_dict("records"):
        normalized_symbol = normalize_symbol_for_cache(
            company_ticker_record.get("ticker")
        )
        security_title = normalize_security_title(company_ticker_record.get("title"))
        rejection_reason = identify_non_common_stock_reason(
            normalized_symbol,
            security_title,
        )
        audit_records.append(
            {
                "symbol": normalized_symbol,
                "sec_title": security_title,
                "hard_filter_decision": (
                    "include" if rejection_reason is None else "exclude"
                ),
                "hard_filter_reason": rejection_reason or "",
            }
        )
    return pandas.DataFrame(audit_records)


def build_hard_filtered_symbols_from_company_tickers(
    company_ticker_table: pandas.DataFrame,
) -> list[str]:
    """Build sorted symbols that pass the deterministic hard filter."""

    if "ticker" not in company_ticker_table.columns:
        raise ValueError("SEC company ticker table must contain a 'ticker' column")

    audit_frame = build_symbol_hard_filter_audit_frame(company_ticker_table)
    accepted_symbol_series = audit_frame.loc[
        audit_frame["hard_filter_decision"] == "include", "symbol"
    ]
    accepted_symbol_set = {
        symbol_name for symbol_name in accepted_symbol_series.astype(str) if symbol_name
    }

    rejection_counts_by_reason = (
        audit_frame.loc[
            audit_frame["hard_filter_decision"] == "exclude",
            "hard_filter_reason",
        ]
        .value_counts()
        .to_dict()
    )
    if rejection_counts_by_reason:
        LOGGER.info(
            "Rejected %d SEC ticker records from symbol cache: %s",
            sum(rejection_counts_by_reason.values()),
            rejection_counts_by_reason,
        )
    return sorted(accepted_symbol_set)


def build_common_stock_symbols_from_company_tickers(
    company_ticker_table: pandas.DataFrame,
) -> list[str]:
    """Build the legacy symbol cache candidate list from SEC ticker records."""

    return build_hard_filtered_symbols_from_company_tickers(company_ticker_table)


def update_symbol_cache() -> None:
    """Build the local common-stock symbol cache from SEC company tickers.

    The SEC feed is the source of ticker/issuer coverage, including new IPOs.
    ``symbols.txt`` should contain tradeable common-stock candidates only, so
    deterministic hard filters remove warrant, unit, right, preferred, obvious
    ETF, ETN, bond and note securities before writing the cache.
    """

    from stock_indicator.sector_pipeline.sec_api import fetch_company_ticker_table

    company_ticker_table = fetch_company_ticker_table()
    symbol_list = build_hard_filtered_symbols_from_company_tickers(
        company_ticker_table
    )
    SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYMBOL_CACHE_PATH.write_text("\n".join(symbol_list) + "\n", encoding="utf-8")
    LOGGER.info(
        "Symbol cache written to %s (%d common-stock candidates from SEC company_tickers.json)",
        SYMBOL_CACHE_PATH,
        len(symbol_list),
    )


def load_symbols() -> list[str]:
    """Return the list of symbols from the local cache.

    The cache file may contain one ticker per line or a JSON encoded list of
    strings representing ticker symbols.
    """

    if not SYMBOL_CACHE_PATH.exists():
        update_symbol_cache()
    file_content = SYMBOL_CACHE_PATH.read_text(encoding="utf-8")
    try:
        parsed_symbols = json.loads(file_content)
    except json.JSONDecodeError:
        symbol_list = [
            line.strip()
            for line in file_content.splitlines()
            if line.strip()
        ]
    else:
        if not isinstance(parsed_symbols, list) or not all(
            isinstance(symbol, str) for symbol in parsed_symbols
        ):
            raise ValueError("Symbol cache JSON must be a list of strings.")
        symbol_list = parsed_symbols
    return symbol_list
