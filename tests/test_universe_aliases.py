"""Tests for production-old and research-new universe file separation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas

from stock_indicator import manage

DATA_DIRECTORY = Path("data")
PRODUCTION_SYMBOLS_PATH = DATA_DIRECTORY / "production_old_symbols.txt"
RESEARCH_SYMBOLS_PATH = DATA_DIRECTORY / "research_new_symbols.txt"
PRODUCTION_SECTOR_PATH = DATA_DIRECTORY / "production_old_symbols_with_sector.parquet"
RESEARCH_SECTOR_PATH = DATA_DIRECTORY / "research_new_symbols_with_sector.parquet"


def _load_symbols(symbols_path: Path) -> list[str]:
    """Return normalized non-empty symbols from a text file."""

    return [
        line_text.strip().upper()
        for line_text in symbols_path.read_text(encoding="utf-8").splitlines()
        if line_text.strip()
    ]


def _assert_sector_file_covers_symbols(
    sector_path: Path,
    symbol_path: Path,
) -> pandas.DataFrame:
    """Assert that an FF12 file exactly covers the supplied symbol list."""

    symbol_set = set(_load_symbols(symbol_path))
    sector_frame = pandas.read_parquet(sector_path)
    ticker_set = set(sector_frame["ticker"].astype(str).str.upper())
    assert ticker_set == symbol_set
    ff12_values = pandas.to_numeric(sector_frame["ff12"], errors="raise")
    assert ff12_values.between(1, 12).all()
    return sector_frame


def test_symbol_list_aliases_point_to_separated_universes() -> None:
    """Named symbol-list aliases should separate production from research."""

    assert (
        manage.SYMBOL_LIST_PATHS["production_old"]
        == PRODUCTION_SYMBOLS_PATH.resolve()
    )
    assert (
        manage.SYMBOL_LIST_PATHS["research_new"]
        == RESEARCH_SYMBOLS_PATH.resolve()
    )
    assert (
        manage.SYMBOL_LIST_PATHS["2010_safe"]
        == PRODUCTION_SYMBOLS_PATH.resolve()
    )


def test_committed_multi_bucket_configs_select_expected_universes() -> None:
    """Production and research configs should resolve to their intended universes."""

    expected_config_values = {
        DATA_DIRECTORY / "multi_bucket_production.json": {
            "data_source": "daily",
            "symbol_list": "production_old",
            "ff12_data_path": "data/production_old_symbols_with_sector.parquet",
        },
        DATA_DIRECTORY / "multi_bucket_triple_explore.json": {
            "data_source": "2010",
            "symbol_list": "research_new",
            "ff12_data_path": "data/research_new_symbols_with_sector.parquet",
        },
        DATA_DIRECTORY / "multi_bucket_triple_explore_old_universe.json": {
            "data_source": "2010",
            "symbol_list": "production_old",
            "ff12_data_path": "data/production_old_symbols_with_sector.parquet",
        },
    }
    for config_path, expected_values in expected_config_values.items():
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
        for config_key, expected_value in expected_values.items():
            assert config_document[config_key] == expected_value


def test_production_config_uses_old_universe_risk_priority_path() -> None:
    """Production cron config should promote the tested old-universe path."""

    config_document = json.loads(
        (DATA_DIRECTORY / "multi_bucket_production.json").read_text(
            encoding="utf-8"
        )
    )
    bucket_by_label = {
        bucket_document["label"]: bucket_document
        for bucket_document in config_document["buckets"]
    }

    assert bucket_by_label["fish_tail_explore"]["max_hold"] == 7
    assert config_document["risk_score_priority_overrides"] == {
        "scores": [25, 50],
        "priorities": {
            "fish_head_production": 1,
            "fish_tail_explore": 2,
            "fish_head_b30_35": 3,
        },
    }


def test_production_default_symbol_file_is_old_universe_alias() -> None:
    """Default symbol file should remain a byte-for-byte production alias."""

    assert (DATA_DIRECTORY / "symbols.txt").read_text(encoding="utf-8") == (
        PRODUCTION_SYMBOLS_PATH.read_text(encoding="utf-8")
    )


def test_sector_files_cover_expected_universes_with_standard_ff12_groups() -> None:
    """Production and research sector files should cover all symbols with FF12 1-12."""

    production_sector_frame = _assert_sector_file_covers_symbols(
        PRODUCTION_SECTOR_PATH,
        PRODUCTION_SYMBOLS_PATH,
    )
    research_sector_frame = _assert_sector_file_covers_symbols(
        RESEARCH_SECTOR_PATH,
        RESEARCH_SYMBOLS_PATH,
    )
    production_source_values = {
        source_text.strip().lower()
        for source_text in production_sector_frame["ff12_source"].astype(str)
    }
    production_confidence_values = {
        confidence_text.strip().lower()
        for confidence_text in production_sector_frame[
            "classification_confidence"
        ].astype(str)
    }
    assert "legacy_backtest" in production_source_values
    assert "missing_sic_fallback" not in production_source_values
    assert "low" not in production_confidence_values
    assert len(production_sector_frame) == len(_load_symbols(PRODUCTION_SYMBOLS_PATH))
    assert len(research_sector_frame) == len(_load_symbols(RESEARCH_SYMBOLS_PATH))


def test_no_config_uses_custom_etf_ff12_group() -> None:
    """Configs should not refer to the abandoned ETF FF12=13 path."""

    text_paths = list(DATA_DIRECTORY.glob("multi_bucket*.json"))
    policy_override_path = DATA_DIRECTORY / "symbol_universe_policy_overrides.csv"
    if policy_override_path.exists():
        text_paths.append(policy_override_path)

    for text_path in text_paths:
        file_text = text_path.read_text(encoding="utf-8")
        lowered_text = file_text.lower()
        assert "ff12=13" not in lowered_text
        assert "manual_anti_crash_etf" not in lowered_text
        assert "institutional_anti_crash_etf" not in lowered_text

        if text_path.suffix != ".json":
            continue
        config_document: dict[str, Any] = json.loads(file_text)
        for bucket_document in config_document.get("buckets", []):
            assert 13 not in bucket_document.get("skip_ff12_groups", [])
