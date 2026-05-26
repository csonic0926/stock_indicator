import pandas as pd

from stock_indicator.sector_pipeline.utils import normalize_ticker_symbol
from stock_indicator.sector_pipeline.ff_mapping import (
    build_classification_lookup,
    attach_fama_french_groups,
)
from stock_indicator.sector_pipeline.secondary_classification import (
    apply_secondary_classifications,
    build_secondary_classifications_from_yfinance_metadata,
    load_secondary_classifications,
)


def test_normalize_ticker_symbol_converts_dash_to_dot():
    assert normalize_ticker_symbol("brk-b") == "BRK.B"


def test_build_classification_lookup_expands_ranges():
    mapping_data_frame = pd.DataFrame(
        {
            "sic_start": [1000],
            "sic_end": [1002],
            "ff12": [1],
            "ff48": [2],
            "ff49": [3],
            "label": ["Test"],
        }
    )
    lookup_data_frame = build_classification_lookup(mapping_data_frame)
    assert len(lookup_data_frame) == 3
    assert (
        lookup_data_frame.loc[lookup_data_frame["sic"] == 1001, "ff48"].iloc[0] == 2
    )


def test_attach_fama_french_groups_merges_lookup():
    data_frame = pd.DataFrame({"sic": [1001]})
    lookup_data_frame = pd.DataFrame(
        {
            "sic": [1001],
            "ff12": [1],
            "ff48": [2],
            "ff49": [3],
            "ff_label": ["Label"],
        }
    )
    merged_data_frame = attach_fama_french_groups(data_frame, lookup_data_frame)
    assert merged_data_frame["ff48"].iloc[0] == 2
    assert merged_data_frame["ff_label"].iloc[0] == "Label"


def test_attach_fama_french_groups_marks_classification_provenance():
    """FF12=12 should distinguish valid Other from missing-SIC fallback."""

    data_frame = pd.DataFrame({"sic": [1001, 7011, None]})
    lookup_data_frame = pd.DataFrame(
        {
            "sic": [1001],
            "ff12": [1],
            "ff48": [2],
            "ff49": [3],
            "ff_label": ["Label"],
        }
    )

    merged_data_frame = attach_fama_french_groups(data_frame, lookup_data_frame)

    assert list(merged_data_frame["ff12"]) == [1, 12, 12]
    assert list(merged_data_frame["ff12_source"]) == [
        "sic_mapping",
        "sic_unmapped_other",
        "missing_sic_fallback",
    ]
    assert list(merged_data_frame["classification_confidence"]) == [
        "high",
        "medium",
        "low",
    ]
    assert list(merged_data_frame["classification_issue"]) == [
        "",
        "unmapped_sic",
        "missing_sic",
    ]


def test_build_secondary_classifications_from_yfinance_metadata():
    """Yahoo metadata should produce deterministic secondary FF12 records."""

    metadata_frame = pd.DataFrame(
        {
            "symbol": ["ARCC", "BABA", "SPY"],
            "yf_quote_type": ["EQUITY", "EQUITY", "ETF"],
            "yf_sector": [
                "Financial Services",
                "Consumer Cyclical",
                "",
            ],
            "yf_industry": ["Asset Management", "Internet Retail", ""],
        }
    )

    secondary_frame = build_secondary_classifications_from_yfinance_metadata(
        metadata_frame
    )

    assert list(secondary_frame["ticker"]) == ["ARCC", "BABA"]
    assert list(secondary_frame["ff12"]) == [11, 9]
    assert set(secondary_frame["ff12_source"]) == {"secondary_yfinance"}


def test_apply_secondary_classifications_only_upgrades_missing_sic_rows():
    """Secondary data should not overwrite existing SIC-based classifications."""

    classified_data_frame = pd.DataFrame(
        {
            "ticker": ["ARCC", "AAPL"],
            "sic": [None, 3571],
            "ff12": [12, 6],
            "ff_label": ["Other", "BusEq"],
            "ff12_source": ["missing_sic_fallback", "sic_mapping"],
            "classification_confidence": ["low", "high"],
            "classification_issue": ["missing_sic", ""],
        }
    )
    secondary_frame = pd.DataFrame(
        {
            "ticker": ["ARCC", "AAPL"],
            "ff12": [11, 11],
            "ff_label": ["Money", "Money"],
            "ff12_source": ["secondary_yfinance", "secondary_yfinance"],
            "classification_confidence": ["high", "high"],
            "classification_issue": ["resolved_missing_sic", "resolved"],
        }
    )

    upgraded_frame = apply_secondary_classifications(
        classified_data_frame,
        secondary_frame,
    )

    assert list(upgraded_frame["ff12"]) == [11, 6]
    assert list(upgraded_frame["ff12_source"]) == [
        "secondary_yfinance",
        "sic_mapping",
    ]


def test_load_secondary_classifications_includes_only_llm_reviewed_symbols(tmp_path):
    """LLM review rows should classify includes and leave exclusions for audit."""

    yfinance_classification_path = tmp_path / "sector_secondary_classifications.csv"
    yfinance_classification_path.write_text(
        "ticker,ff12,ff_label,ff12_source,classification_confidence,"
        "classification_issue\n"
        "BABA,9,Shops,secondary_yfinance,high,resolved_missing_sic\n",
        encoding="utf-8",
    )
    llm_classification_path = tmp_path / "sector_missing_sic_llm_classification.csv"
    llm_classification_path.write_text(
        "ticker,llm_decision,ff12,ff_label,ff12_source,"
        "classification_confidence,classification_issue,reason\n"
        "ACMIF,include,12,Other,secondary_llm,medium,"
        "resolved_missing_sic_llm,Metals exploration maps to FF12 Other.\n"
        "SHELL,exclude,,,,low,shell_company,Shell company has no sector.\n",
        encoding="utf-8",
    )

    classification_frame = load_secondary_classifications(
        yfinance_classification_path,
        llm_classification_path,
    )

    assert set(classification_frame["ticker"]) == {"BABA", "ACMIF"}
    assert "SHELL" not in set(classification_frame["ticker"])
    acmif_row = classification_frame.loc[
        classification_frame["ticker"] == "ACMIF"
    ].iloc[0]
    assert acmif_row["ff12"] == 12
    assert acmif_row["ff12_source"] == "secondary_llm"
