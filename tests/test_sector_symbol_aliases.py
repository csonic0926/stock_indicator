"""Tests for dot/dash symbol aliases in sector lookups."""

from pathlib import Path

import pytest

import stock_indicator.strategy as strategy


def test_load_ff12_groups_by_symbol_supports_dot_dash_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sector lookup should match YF dash symbols and SEC dot symbols."""

    sector_csv_path = tmp_path / "symbols_with_sector.csv"
    sector_csv_path.write_text(
        "ticker,ff12\nBRK.B,11\nBF.B,1\nDIS,12\n",
        encoding="utf-8",
    )

    from stock_indicator.sector_pipeline import config as sector_config

    monkeypatch.setattr(
        sector_config,
        "DEFAULT_OUTPUT_PARQUET_PATH",
        tmp_path / "missing_symbols_with_sector.parquet",
    )
    monkeypatch.setattr(
        sector_config,
        "DEFAULT_OUTPUT_CSV_PATH",
        sector_csv_path,
    )

    symbol_to_group = strategy.load_ff12_groups_by_symbol()

    assert symbol_to_group["BRK.B"] == 11
    assert symbol_to_group["BRK-B"] == 11
    assert symbol_to_group["BF.B"] == 1
    assert symbol_to_group["BF-B"] == 1
    assert symbol_to_group["DIS"] == 12


def test_load_ff12_groups_by_symbol_skips_missing_sic_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-confidence FF12=12 fallback should not be used as real sector data."""

    sector_csv_path = tmp_path / "symbols_with_sector.csv"
    sector_csv_path.write_text(
        "ticker,ff12,ff12_source,classification_confidence\n"
        "AAPL,6,sic_mapping,high\n"
        "AAL,12,sic_unmapped_other,medium\n"
        "ADR,12,missing_sic_fallback,low\n",
        encoding="utf-8",
    )

    from stock_indicator.sector_pipeline import config as sector_config

    monkeypatch.setattr(
        sector_config,
        "DEFAULT_OUTPUT_PARQUET_PATH",
        tmp_path / "missing_symbols_with_sector.parquet",
    )
    monkeypatch.setattr(
        sector_config,
        "DEFAULT_OUTPUT_CSV_PATH",
        sector_csv_path,
    )

    symbol_to_group = strategy.load_ff12_groups_by_symbol()

    assert symbol_to_group["AAPL"] == 6
    assert symbol_to_group["AAL"] == 12
    assert "ADR" not in symbol_to_group


def test_load_symbols_excluded_by_industry_is_legacy_no_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Industry exclusions should not override upstream symbol decisions."""

    sector_csv_path = tmp_path / "symbols_with_sector.csv"
    sector_csv_path.write_text(
        "ticker,ff12,sic\n"
        "BRK.B,12,6331\n"
        "NFLX,12,7841\n"
        "AAC.WS,12,6770\n",
        encoding="utf-8",
    )

    from stock_indicator.sector_pipeline import config as sector_config
    from stock_indicator.sector_pipeline import overrides as sector_overrides

    monkeypatch.setattr(
        sector_config,
        "DEFAULT_OUTPUT_PARQUET_PATH",
        tmp_path / "missing_symbols_with_sector.parquet",
    )
    monkeypatch.setattr(
        sector_config,
        "DEFAULT_OUTPUT_CSV_PATH",
        sector_csv_path,
    )
    monkeypatch.setattr(
        sector_overrides,
        "SECTOR_OVERRIDES_CSV_PATH",
        tmp_path / "missing_sector_overrides.csv",
    )

    excluded_symbols = strategy.load_symbols_excluded_by_industry()

    assert "BRK.B" not in excluded_symbols
    assert "BRK-B" not in excluded_symbols
    assert "NFLX" not in excluded_symbols
    assert "AAC.WS" not in excluded_symbols
    assert "AAC-WS" not in excluded_symbols
    assert excluded_symbols == set()
