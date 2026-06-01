"""Tests for historical risk score CSV and report generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType


def load_historical_risk_scores_module() -> ModuleType:
    """Load the standalone historical_risk_scores script as a module."""
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "historical_risk_scores.py"
    )
    module_specification = importlib.util.spec_from_file_location(
        "historical_risk_scores",
        script_path,
    )
    if module_specification is None or module_specification.loader is None:
        raise RuntimeError(f"Unable to load historical risk score script: {script_path}")

    historical_risk_scores_module = importlib.util.module_from_spec(
        module_specification
    )
    module_specification.loader.exec_module(historical_risk_scores_module)
    return historical_risk_scores_module


def test_build_risk_report_content_includes_score_and_gate_impact() -> None:
    """Monthly risk report should expose score, recommendation, and gate status."""
    historical_risk_scores_module = load_historical_risk_scores_module()
    report_content = historical_risk_scores_module.build_risk_report_content(
        (
            "2026-06",
            25,
            25,
            "Known Apr CPI/PCE inflation; FSB private-credit and AI-valuation risk",
            "H",
        ),
        generated_date_text="2026-06-01",
    )

    assert "# Risk Report: 2026-06" in report_content
    assert "- Risk score: 50" in report_content
    assert "- Recommendation: reduce" in report_content
    assert "Duration remains 25" in report_content
    assert "- Gate status: open (stop threshold: 75)" in report_content


def test_write_latest_risk_report_writes_month_named_file() -> None:
    """Latest risk report should be written as a month-named Markdown file."""
    historical_risk_scores_module = load_historical_risk_scores_module()
    with TemporaryDirectory() as temporary_directory_name:
        report_directory = Path(temporary_directory_name)
        report_path = historical_risk_scores_module.write_latest_risk_report(
            report_directory,
            generated_date_text="2026-06-01",
        )

        assert report_path == report_directory / "2026-06.md"
        assert report_path.exists()
        assert "Known Apr CPI/PCE inflation" in report_path.read_text(
            encoding="utf-8"
        )
