"""Tests for the atomic universe pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas
import pytest

from stock_indicator import universe_pipeline


def _write_csv(data_directory: Path, file_name: str, rows: list[dict[str, object]]) -> None:
    """Write a small CSV fixture under ``data_directory``."""

    data_directory.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(rows).to_csv(data_directory / file_name, index=False)


def _write_mature_price_audit(
    data_directory: Path,
    symbol_names: list[str],
) -> None:
    """Write mature local price-history metadata for fixture symbols."""

    _write_csv(
        data_directory,
        universe_pipeline.PRICE_HISTORY_PREPARE_AUDIT_FILE_NAME,
        [
            {
                "symbol": symbol_name,
                "row_count_final": 300,
                "first_date_final": "2024-01-02",
                "last_date_final": "2026-01-05",
            }
            for symbol_name in symbol_names
        ],
    )


def _build_sector_frame(
    symbol_names: list[str],
    company_ticker_table: pandas.DataFrame,
    mapping_source: str | Path,
) -> pandas.DataFrame:
    """Return trusted sector rows for the supplied symbols."""

    del company_ticker_table, mapping_source
    sector_records: list[dict[str, object]] = []
    for symbol_position, symbol_name in enumerate(symbol_names, start=1):
        sector_records.append(
            {
                "ticker": symbol_name,
                "cik": 1000 + symbol_position,
                "sic": 3571,
                "ff12": 6,
                "ff48": -1,
                "ff49": -1,
                "ff_label": "BusEq",
                "ff12_source": "sic_mapping",
                "classification_confidence": "high",
                "classification_issue": "",
                "sic_desc": "",
                "secondary_sector": "",
                "secondary_industry": "",
                "secondary_source": "",
                "secondary_reason": "",
            }
        )
    return pandas.DataFrame(sector_records)


def test_universe_pipeline_applies_llm_policy_price_and_preserves_na_symbol(
    tmp_path: Path,
) -> None:
    """The final contract should combine all universe layers atomically."""

    data_directory = tmp_path / "data"
    company_ticker_table = pandas.DataFrame(
        [
            {"ticker": "AAA", "cik": 1, "title": "Alpha Analytics Inc."},
            {
                "ticker": "DLR",
                "cik": 2,
                "title": "Digital Realty Trust, Inc.",
                "exchange": "NYSE",
            },
            {
                "ticker": "BBDC",
                "cik": 3,
                "title": "Barings BDC, Inc.",
                "exchange": "NYSE",
            },
            {
                "ticker": "BADF",
                "cik": 4,
                "title": "Bad Income Fund",
                "exchange": "NYSE",
            },
            {
                "ticker": "OLD",
                "cik": 5,
                "title": "Old Operating Inc.",
                "exchange": "NYSE",
            },
            {
                "ticker": "NA",
                "cik": 6,
                "title": "Nano Labs Ltd",
                "exchange": "Nasdaq",
            },
            {
                "ticker": "WRT.WS",
                "cik": 7,
                "title": "Warrant Example",
                "exchange": "NYSE",
            },
        ]
    )
    company_ticker_table.loc[
        company_ticker_table["ticker"] == "AAA",
        "exchange",
    ] = "Nasdaq"
    _write_mature_price_audit(data_directory, ["AAA", "DLR", "NA"])
    _write_csv(
        data_directory,
        universe_pipeline.SYMBOL_LLM_CLASSIFICATION_FILE_NAME,
        [
            {
                "symbol": "DLR",
                "sec_title": "DIGITAL REALTY TRUST, INC.",
                "decision": "include",
                "semantic_type": "reit_common_equity",
                "confidence": "high",
                "reason": "REIT common equity.",
            },
            {
                "symbol": "BBDC",
                "sec_title": "BARINGS BDC, INC.",
                "decision": "include",
                "semantic_type": "bdc_common_equity",
                "confidence": "high",
                "reason": "BDC equity.",
            },
            {
                "symbol": "BADF",
                "sec_title": "BAD INCOME FUND",
                "decision": "exclude",
                "semantic_type": "closed_end_fund",
                "confidence": "high",
                "reason": "Closed-end fund.",
            },
        ],
    )
    _write_csv(
        data_directory,
        universe_pipeline.SYMBOL_POLICY_OVERRIDE_FILE_NAME,
        [
            {
                "match_field": "semantic_type",
                "match_value": "bdc_common_equity",
                "override_decision": "exclude",
                "override_semantic_type": "private_credit_or_bdc_vehicle",
                "override_reason": "BDC policy exclusion.",
            }
        ],
    )
    _write_csv(
        data_directory,
        universe_pipeline.SYMBOL_PRICE_QUARANTINE_FILE_NAME,
        [{"symbol": "OLD", "status": "empty_csv"}],
    )

    report = universe_pipeline.run_universe_pipeline(
        data_directory=data_directory,
        company_ticker_table=company_ticker_table,
        sector_frame_builder=_build_sector_frame,
    )

    final_symbols = (data_directory / universe_pipeline.SYMBOL_CONTRACT_FILE_NAME).read_text(encoding="utf-8").split()
    assert final_symbols == ["AAA", "DLR", "NA"]
    assert report.final_symbol_count == 3
    assert report.price_quarantine_count == 1
    final_audit_frame = pandas.read_csv(
        data_directory / universe_pipeline.SYMBOL_FINAL_AUDIT_FILE_NAME,
        keep_default_na=False,
    )
    assert final_audit_frame.set_index("symbol").loc["BBDC", "decision_source"] == "policy_override"
    old_symbol_record = final_audit_frame.set_index("symbol").loc["OLD"]
    assert old_symbol_record["decision_source"] == "price_source_quarantine"
    assert (data_directory / universe_pipeline.SECTOR_PARQUET_FILE_NAME).exists()
    assert (data_directory / universe_pipeline.SECTOR_CSV_FILE_NAME).exists()


def test_universe_pipeline_quarantines_new_second_layer_candidate_without_llm(
    tmp_path: Path,
) -> None:
    """Ambiguous new symbols must fail closed when no sticky LLM row exists."""

    data_directory = tmp_path / "data"
    company_ticker_table = pandas.DataFrame(
        [
            {
                "ticker": "AAA",
                "cik": 1,
                "title": "Alpha Analytics Inc.",
                "exchange": "Nasdaq",
            },
            {
                "ticker": "NEWF",
                "cik": 2,
                "title": "New Income Fund",
                "exchange": "NYSE",
            },
        ]
    )
    _write_mature_price_audit(data_directory, ["AAA", "NEWF"])

    report = universe_pipeline.run_universe_pipeline(
        data_directory=data_directory,
        company_ticker_table=company_ticker_table,
        sector_frame_builder=_build_sector_frame,
    )

    final_symbols = (data_directory / universe_pipeline.SYMBOL_CONTRACT_FILE_NAME).read_text(encoding="utf-8").split()
    assert final_symbols == ["AAA"]
    assert report.missing_llm_classification_count == 1
    final_audit_frame = pandas.read_csv(
        data_directory / universe_pipeline.SYMBOL_FINAL_AUDIT_FILE_NAME,
        keep_default_na=False,
    )
    new_fund_record = final_audit_frame.set_index("symbol").loc["NEWF"]
    assert new_fund_record["final_decision"] == "quarantine"
    assert new_fund_record["decision_source"] == "llm_second_layer_missing_cache"


def test_universe_pipeline_excludes_untrusted_exchange_and_immature_symbols(
    tmp_path: Path,
) -> None:
    """The tradability gate should block OTC and brand-new symbols."""

    data_directory = tmp_path / "data"
    company_ticker_table = pandas.DataFrame(
        [
            {
                "ticker": "MATURE",
                "cik": 1,
                "title": "Mature Exchange Inc.",
                "exchange": "Nasdaq",
            },
            {
                "ticker": "OTCF",
                "cik": 2,
                "title": "OTC Filing Corp.",
                "exchange": "OTC",
            },
            {
                "ticker": "IPOX",
                "cik": 3,
                "title": "Brand New IPO Inc.",
                "exchange": "NYSE",
            },
            {
                "ticker": "MISSING",
                "cik": 4,
                "title": "Missing Price Inc.",
                "exchange": "NYSE",
            },
        ]
    )
    _write_csv(
        data_directory,
        universe_pipeline.PRICE_HISTORY_PREPARE_AUDIT_FILE_NAME,
        [
            {
                "symbol": "MATURE",
                "row_count_final": 300,
                "first_date_final": "2024-01-02",
                "last_date_final": "2026-01-05",
            },
            {
                "symbol": "OTCF",
                "row_count_final": 300,
                "first_date_final": "2024-01-02",
                "last_date_final": "2026-01-05",
            },
            {
                "symbol": "IPOX",
                "row_count_final": 40,
                "first_date_final": "2025-11-03",
                "last_date_final": "2026-01-05",
            },
        ],
    )

    report = universe_pipeline.run_universe_pipeline(
        data_directory=data_directory,
        company_ticker_table=company_ticker_table,
        sector_frame_builder=_build_sector_frame,
    )

    final_symbols = (data_directory / universe_pipeline.SYMBOL_CONTRACT_FILE_NAME).read_text(encoding="utf-8").split()
    assert final_symbols == ["MATURE"]
    assert report.tradability_gate_exclusion_count == 3
    final_audit_frame = pandas.read_csv(
        data_directory / universe_pipeline.SYMBOL_FINAL_AUDIT_FILE_NAME,
        keep_default_na=False,
    ).set_index("symbol")
    assert final_audit_frame.loc["OTCF", "tradability_reason"] == (
        "untrusted_or_missing_sec_exchange"
    )
    assert final_audit_frame.loc["IPOX", "tradability_reason"] == (
        "price_history_rows_below_252"
    )
    assert final_audit_frame.loc["MISSING", "tradability_reason"] == (
        "missing_price_history"
    )


def test_universe_pipeline_policy_include_bypasses_tradability_gate(
    tmp_path: Path,
) -> None:
    """Explicit policy includes should not be removed by automated maturity checks."""

    data_directory = tmp_path / "data"
    company_ticker_table = pandas.DataFrame(
        [
            {
                "ticker": "GLD",
                "cik": 1,
                "title": "SPDR Gold Trust",
                "exchange": "NYSE Arca",
            }
        ]
    )
    _write_csv(
        data_directory,
        universe_pipeline.SYMBOL_POLICY_OVERRIDE_FILE_NAME,
        [
            {
                "match_field": "symbol",
                "match_value": "GLD",
                "override_decision": "include",
                "override_semantic_type": "manual_policy_include",
                "override_reason": "Manual policy include.",
            }
        ],
    )

    report = universe_pipeline.run_universe_pipeline(
        data_directory=data_directory,
        company_ticker_table=company_ticker_table,
        sector_frame_builder=_build_sector_frame,
    )

    final_symbols = (data_directory / universe_pipeline.SYMBOL_CONTRACT_FILE_NAME).read_text(encoding="utf-8").split()
    assert final_symbols == ["GLD"]
    assert report.tradability_gate_exclusion_count == 0
    final_audit_frame = pandas.read_csv(
        data_directory / universe_pipeline.SYMBOL_FINAL_AUDIT_FILE_NAME,
        keep_default_na=False,
    ).set_index("symbol")
    assert final_audit_frame.loc["GLD", "tradability_status"] == "manual_include"


def test_universe_pipeline_keeps_existing_contract_when_sector_validation_fails(
    tmp_path: Path,
) -> None:
    """No output swap should happen if sector coverage validation fails."""

    data_directory = tmp_path / "data"
    data_directory.mkdir(parents=True)
    existing_symbols_path = data_directory / universe_pipeline.SYMBOL_CONTRACT_FILE_NAME
    existing_symbols_path.write_text("OLD\n", encoding="utf-8")
    existing_sector_csv_path = data_directory / universe_pipeline.SECTOR_CSV_FILE_NAME
    existing_sector_csv_path.write_text("ticker,ff12\nOLD,6\n", encoding="utf-8")
    company_ticker_table = pandas.DataFrame(
        [
            {
                "ticker": "AAA",
                "cik": 1,
                "title": "Alpha Analytics Inc.",
                "exchange": "Nasdaq",
            }
        ]
    )
    _write_mature_price_audit(data_directory, ["AAA"])

    def build_invalid_sector_frame(
        symbol_names: list[str],
        company_ticker_table: pandas.DataFrame,
        mapping_source: str | Path,
    ) -> pandas.DataFrame:
        del symbol_names, company_ticker_table, mapping_source
        return pandas.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "ff12": 12,
                    "ff12_source": "missing_sic_fallback",
                    "classification_confidence": "low",
                }
            ]
        )

    with pytest.raises(ValueError, match="low-confidence"):
        universe_pipeline.run_universe_pipeline(
            data_directory=data_directory,
            company_ticker_table=company_ticker_table,
            sector_frame_builder=build_invalid_sector_frame,
        )

    assert existing_symbols_path.read_text(encoding="utf-8") == "OLD\n"
    assert existing_sector_csv_path.read_text(encoding="utf-8") == "ticker,ff12\nOLD,6\n"


def test_validate_sector_contract_rejects_custom_ff_group() -> None:
    """Universe FF12 output is constrained to the standard groups 1-12."""

    sector_frame = pandas.DataFrame(
        [
            {
                "ticker": "GLD",
                "ff12": 13,
                "ff12_source": "manual_policy_include",
                "classification_confidence": "high",
            }
        ]
    )

    with pytest.raises(ValueError, match="invalid FF12"):
        universe_pipeline.validate_sector_contract(sector_frame, ["GLD"])


def test_universe_pipeline_rejects_large_symbol_count_drop(tmp_path: Path) -> None:
    """A partial SEC response should not replace a much larger current universe."""

    data_directory = tmp_path / "data"
    data_directory.mkdir(parents=True)
    current_symbols = [f"OLD{symbol_index}" for symbol_index in range(100)]
    (data_directory / universe_pipeline.SYMBOL_CONTRACT_FILE_NAME).write_text(
        "\n".join(current_symbols) + "\n",
        encoding="utf-8",
    )
    company_ticker_table = pandas.DataFrame(
        [
            {
                "ticker": "AAA",
                "cik": 1,
                "title": "Alpha Analytics Inc.",
                "exchange": "Nasdaq",
            },
            {
                "ticker": "BBB",
                "cik": 2,
                "title": "Beta Builders Inc.",
                "exchange": "NYSE",
            },
        ]
    )

    with pytest.raises(ValueError, match="Proposed universe is too small"):
        universe_pipeline.run_universe_pipeline(
            data_directory=data_directory,
            company_ticker_table=company_ticker_table,
            sector_frame_builder=_build_sector_frame,
        )

    assert (data_directory / universe_pipeline.SYMBOL_CONTRACT_FILE_NAME).read_text(encoding="utf-8").startswith(
        "OLD0\n"
    )
