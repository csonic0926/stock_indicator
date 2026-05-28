"""Tests for appending promoted candidate FF12 rows into production."""

from __future__ import annotations

import io
from pathlib import Path

import pandas
import pytest

from stock_indicator import manage
from stock_indicator import production_ff12_promotion
from stock_indicator import strategy


SECTOR_COLUMNS = [
    "ticker",
    "cik",
    "sic",
    "ff12",
    "ff48",
    "ff49",
    "ff_label",
    "ff12_source",
    "classification_confidence",
    "classification_issue",
    "secondary_sector",
    "secondary_industry",
    "secondary_source",
    "secondary_reason",
    "sic_desc",
]


def _sector_record(
    symbol_name: str,
    *,
    ff12_group: object = 6,
    ff12_source: str = "legacy_backtest",
    classification_confidence: str = "high",
) -> dict[str, object]:
    """Return a complete sector fixture row."""

    return {
        "ticker": symbol_name,
        "cik": "",
        "sic": "",
        "ff12": ff12_group,
        "ff48": -1,
        "ff49": -1,
        "ff_label": "BusEq",
        "ff12_source": ff12_source,
        "classification_confidence": classification_confidence,
        "classification_issue": "",
        "secondary_sector": "",
        "secondary_industry": "",
        "secondary_source": "",
        "secondary_reason": "",
        "sic_desc": "",
    }


def _write_sector_pair(
    *,
    data_directory: Path,
    parquet_file_name: str,
    csv_file_name: str,
    sector_records: list[dict[str, object]],
) -> None:
    """Write matching parquet and CSV sector fixtures."""

    sector_frame = pandas.DataFrame(sector_records, columns=SECTOR_COLUMNS)
    sector_frame.to_parquet(data_directory / parquet_file_name, index=False)
    sector_frame.to_csv(data_directory / csv_file_name, index=False)


def _write_promotion_inputs(
    *,
    data_directory: Path,
    production_symbols: list[str],
    production_sector_records: list[dict[str, object]],
    candidate_sector_records: list[dict[str, object]],
) -> None:
    """Write a complete promotion fixture under ``data_directory``."""

    data_directory.mkdir(parents=True, exist_ok=True)
    (data_directory / "production_symbols.txt").write_text(
        "\n".join(production_symbols) + "\n",
        encoding="utf-8",
    )
    _write_sector_pair(
        data_directory=data_directory,
        parquet_file_name="production_symbols_with_sector.parquet",
        csv_file_name="production_symbols_with_sector.csv",
        sector_records=production_sector_records,
    )
    _write_sector_pair(
        data_directory=data_directory,
        parquet_file_name="production_candidate_symbols_with_sector.parquet",
        csv_file_name="production_candidate_symbols_with_sector.csv",
        sector_records=candidate_sector_records,
    )


def test_sync_appends_promoted_symbol_and_preserves_legacy_rows(
    tmp_path: Path,
) -> None:
    """Existing production rows stay frozen while missing symbols use candidates."""

    data_directory = tmp_path / "data"
    _write_promotion_inputs(
        data_directory=data_directory,
        production_symbols=["OLD", "NEW"],
        production_sector_records=[
            _sector_record("OLD", ff12_group=3, ff12_source="legacy_backtest"),
        ],
        candidate_sector_records=[
            _sector_record("OLD", ff12_group=9, ff12_source="sic_mapping"),
            _sector_record(
                "NEW",
                ff12_group=12,
                ff12_source="secondary_yfinance",
                classification_confidence="medium",
            ),
        ],
    )

    report = production_ff12_promotion.sync_production_ff12_sector(
        data_directory=data_directory,
    )

    parquet_frame = pandas.read_parquet(
        data_directory / "production_symbols_with_sector.parquet"
    )
    runtime_parquet_frame = pandas.read_parquet(
        data_directory / "symbols_with_sector.parquet"
    )
    csv_frame = pandas.read_csv(
        data_directory / "production_symbols_with_sector.csv",
        keep_default_na=False,
    )
    sector_frame_by_ticker = parquet_frame.set_index("ticker")

    assert report.appended_symbols == ["NEW"]
    assert list(parquet_frame["ticker"]) == ["OLD", "NEW"]
    assert data_directory.joinpath("symbols.txt").read_text(
        encoding="utf-8"
    ) == "OLD\nNEW\n"
    pandas.testing.assert_frame_equal(parquet_frame, runtime_parquet_frame)
    assert sector_frame_by_ticker.loc["OLD", "ff12"] == 3
    assert sector_frame_by_ticker.loc["OLD", "ff12_source"] == "legacy_backtest"
    assert sector_frame_by_ticker.loc["NEW", "ff12"] == 12
    assert sector_frame_by_ticker.loc["NEW", "ff12_source"] == "secondary_yfinance"
    assert sector_frame_by_ticker.loc["NEW", "classification_confidence"] == "medium"
    pandas.testing.assert_series_equal(
        parquet_frame[["ticker", "ff12", "ff12_source", "classification_confidence"]]
        .astype(str)
        .reset_index(drop=True)
        .stack(),
        csv_frame[["ticker", "ff12", "ff12_source", "classification_confidence"]]
        .astype(str)
        .reset_index(drop=True)
        .stack(),
        check_names=False,
    )


def test_sync_fails_when_promoted_symbol_has_no_candidate_sector_row(
    tmp_path: Path,
) -> None:
    """A production symbol missing from both sector sources must fail closed."""

    data_directory = tmp_path / "data"
    _write_promotion_inputs(
        data_directory=data_directory,
        production_symbols=["OLD", "MISSING"],
        production_sector_records=[
            _sector_record("OLD", ff12_group=3, ff12_source="legacy_backtest"),
        ],
        candidate_sector_records=[
            _sector_record("OTHER", ff12_group=4, ff12_source="sic_mapping"),
        ],
    )

    with pytest.raises(ValueError, match="missing candidate sector rows"):
        production_ff12_promotion.sync_production_ff12_sector(
            data_directory=data_directory,
        )

    production_sector_frame = pandas.read_parquet(
        data_directory / "production_symbols_with_sector.parquet"
    )
    assert list(production_sector_frame["ticker"]) == ["OLD"]


def test_sync_fails_on_duplicate_sector_rows(tmp_path: Path) -> None:
    """Duplicate sector rows are ambiguous and must not be published."""

    data_directory = tmp_path / "data"
    _write_promotion_inputs(
        data_directory=data_directory,
        production_symbols=["OLD", "NEW"],
        production_sector_records=[
            _sector_record("OLD", ff12_group=3, ff12_source="legacy_backtest"),
        ],
        candidate_sector_records=[
            _sector_record("NEW", ff12_group=6, ff12_source="sic_mapping"),
            _sector_record("NEW", ff12_group=7, ff12_source="sic_mapping"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate tickers"):
        production_ff12_promotion.sync_production_ff12_sector(
            data_directory=data_directory,
        )


@pytest.mark.parametrize(
    "invalid_sector_record, expected_message",
    [
        (
            _sector_record("NEW", ff12_group=13, ff12_source="sic_mapping"),
            "invalid FF12 values",
        ),
        (
            _sector_record(
                "NEW",
                ff12_group=12,
                ff12_source="sic_unmapped_other",
                classification_confidence="low",
            ),
            "low-confidence rows",
        ),
        (
            _sector_record(
                "NEW",
                ff12_group=12,
                ff12_source="missing_sic_fallback",
                classification_confidence="medium",
            ),
            "missing-SIC fallback rows",
        ),
    ],
)
def test_sync_fails_on_invalid_promoted_sector_contract(
    tmp_path: Path,
    invalid_sector_record: dict[str, object],
    expected_message: str,
) -> None:
    """Promoted rows with invalid FF12 contract values must fail closed."""

    data_directory = tmp_path / "data"
    _write_promotion_inputs(
        data_directory=data_directory,
        production_symbols=["OLD", "NEW"],
        production_sector_records=[
            _sector_record("OLD", ff12_group=3, ff12_source="legacy_backtest"),
        ],
        candidate_sector_records=[invalid_sector_record],
    )

    with pytest.raises(ValueError, match=expected_message):
        production_ff12_promotion.sync_production_ff12_sector(
            data_directory=data_directory,
        )


def test_sync_drops_inactive_sector_rows_to_match_symbol_contract(
    tmp_path: Path,
) -> None:
    """Inactive sector rows should not remain in the active production contract."""

    data_directory = tmp_path / "data"
    _write_promotion_inputs(
        data_directory=data_directory,
        production_symbols=["OLD"],
        production_sector_records=[
            _sector_record("OLD", ff12_group=3, ff12_source="legacy_backtest"),
            _sector_record("REMOVED", ff12_group=4, ff12_source="legacy_backtest"),
        ],
        candidate_sector_records=[
            _sector_record("OLD", ff12_group=3, ff12_source="sic_mapping"),
        ],
    )

    report = production_ff12_promotion.sync_production_ff12_sector(
        data_directory=data_directory,
    )

    production_sector_frame = pandas.read_parquet(
        data_directory / "production_symbols_with_sector.parquet"
    )
    assert report.removed_sector_symbols == ["REMOVED"]
    assert list(production_sector_frame["ticker"]) == ["OLD"]


def test_mixed_source_sector_is_loaded_by_existing_ff12_loader(
    tmp_path: Path,
) -> None:
    """The existing FF12 loader should accept mixed audited production sources."""

    data_directory = tmp_path / "data"
    _write_promotion_inputs(
        data_directory=data_directory,
        production_symbols=["OLD", "NEW"],
        production_sector_records=[
            _sector_record("OLD", ff12_group=3, ff12_source="legacy_backtest"),
        ],
        candidate_sector_records=[
            _sector_record(
                "NEW",
                ff12_group=12,
                ff12_source="sic_unmapped_other",
                classification_confidence="medium",
            ),
        ],
    )
    production_ff12_promotion.sync_production_ff12_sector(
        data_directory=data_directory,
    )

    symbol_to_ff12_group = strategy.load_ff12_groups_by_symbol(
        data_directory / "production_symbols_with_sector.parquet"
    )

    assert symbol_to_ff12_group["OLD"] == 3
    assert symbol_to_ff12_group["NEW"] == 12


def test_manage_sync_command_runs_helper_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Management shell command should expose the promotion helper."""

    recorded_publish_values: list[bool] = []

    def fake_sync_production_ff12_sector(
        *,
        publish_outputs: bool = True,
    ) -> production_ff12_promotion.ProductionFf12PromotionReport:
        """Record helper invocation and return a small report."""

        recorded_publish_values.append(publish_outputs)
        return production_ff12_promotion.ProductionFf12PromotionReport(
            published=publish_outputs,
            production_symbol_count=1,
            original_sector_row_count=1,
            final_sector_row_count=1,
            appended_symbols=[],
            removed_sector_symbols=[],
            ff12_source_counts={"legacy_backtest": 1},
            output_paths={},
        )

    monkeypatch.setattr(
        manage.production_ff12_promotion,
        "sync_production_ff12_sector",
        fake_sync_production_ff12_sector,
    )

    output_buffer = io.StringIO()
    shell = manage.StockShell(stdout=output_buffer)
    shell.onecmd("sync_production_ff12_sector --dry-run")

    assert recorded_publish_values == [False]
    assert "dry run completed" in output_buffer.getvalue()
