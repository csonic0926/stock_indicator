"""Tests for building and updating the sector classification dataset."""

# TODO: review

import pandas as pd


def test_build_and_update_dataset(
    monkeypatch,
    tmp_path,
    submissions_payloads,
    ff_mapping_file,
) -> None:
    """The pipeline should build and refresh the curated symbol universe."""
    import stock_indicator.sector_pipeline.sec_api as sec_api_module
    import stock_indicator.sector_pipeline.pipeline as pipeline_module

    def fake_load_symbols() -> list[str]:
        return ["AAA", "BBB"]

    def fake_fetch_company_ticker_table() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ticker": "AAA", "cik": 1},
                {"ticker": "BBB", "cik": 2},
                {"ticker": "REMOVED", "cik": 3},
            ]
        )

    def fake_fetch_submissions_json(
        central_index_key: int,
        use_cache: bool = True,
    ) -> dict:
        return submissions_payloads[central_index_key]

    monkeypatch.setattr(pipeline_module, "load_symbols", fake_load_symbols)
    monkeypatch.setattr(
        sec_api_module,
        "fetch_company_ticker_table",
        fake_fetch_company_ticker_table,
    )
    monkeypatch.setattr(
        sec_api_module,
        "fetch_submissions_json",
        fake_fetch_submissions_json,
    )
    monkeypatch.setattr(
        sec_api_module,
        "SUBMISSIONS_DIRECTORY",
        tmp_path / "submissions",
    )

    monkeypatch.setattr(
        pipeline_module,
        "LAST_RUN_CONFIG_PATH",
        tmp_path / "last_run.json",
    )
    monkeypatch.setattr(
        pipeline_module,
        "DEFAULT_OUTPUT_PARQUET_PATH",
        tmp_path / "output.parquet",
    )
    monkeypatch.setattr(
        pipeline_module,
        "DEFAULT_OUTPUT_CSV_PATH",
        tmp_path / "output.csv",
    )
    monkeypatch.setattr(
        pipeline_module,
        "SUBMISSIONS_DIRECTORY",
        tmp_path / "submissions",
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, *positional_arguments, **keyword_arguments: None,
    )

    data_frame = pipeline_module.build_sector_classification_dataset(
        ff_mapping_file,
        tmp_path / "output.parquet",
        tmp_path / "output.csv",
    )
    assert set(data_frame["ticker"]) == {"AAA", "BBB"}
    assert set(data_frame["ff48"]) == {10, 20}
    assert (tmp_path / "last_run.json").exists()

    updated_data_frame = pipeline_module.update_latest_dataset()
    assert len(updated_data_frame) == 2
