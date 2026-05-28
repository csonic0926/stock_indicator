"""Build final symbol-universe audit from hard filter, LLM, and overrides."""
# TODO: review

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas

DEFAULT_HARD_FILTER_AUDIT_PATH = Path("data/symbol_hard_filter_audit.csv")
DEFAULT_SECOND_LAYER_CANDIDATE_PATH = Path(
    "data/symbol_second_layer_candidate_audit.csv"
)
DEFAULT_LLM_CLASSIFICATION_PATH = Path("data/symbol_universe_llm_classification.csv")
DEFAULT_POLICY_OVERRIDE_PATH = Path("data/symbol_universe_policy_overrides.csv")
DEFAULT_RUNTIME_GUARDRAIL_LLM_PATH = Path(
    "data/runtime_guardrail_conflict_llm_classification.csv"
)
DEFAULT_FINAL_AUDIT_PATH = Path("data/symbol_universe_hard_plus_llm_audit.csv")
DEFAULT_FINAL_SYMBOLS_PATH = Path("data/symbols_hard_plus_llm_from_sec.txt")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for final universe construction."""

    parser = argparse.ArgumentParser(
        description="Build final symbol universe from hard filter and LLM audit."
    )
    parser.add_argument(
        "--hard-filter-audit-path",
        type=Path,
        default=DEFAULT_HARD_FILTER_AUDIT_PATH,
    )
    parser.add_argument(
        "--second-layer-candidate-path",
        type=Path,
        default=DEFAULT_SECOND_LAYER_CANDIDATE_PATH,
    )
    parser.add_argument(
        "--llm-classification-path",
        type=Path,
        default=DEFAULT_LLM_CLASSIFICATION_PATH,
    )
    parser.add_argument(
        "--policy-override-path",
        type=Path,
        default=DEFAULT_POLICY_OVERRIDE_PATH,
    )
    parser.add_argument(
        "--runtime-guardrail-llm-path",
        type=Path,
        default=DEFAULT_RUNTIME_GUARDRAIL_LLM_PATH,
    )
    parser.add_argument(
        "--final-audit-path",
        type=Path,
        default=DEFAULT_FINAL_AUDIT_PATH,
    )
    parser.add_argument(
        "--final-symbols-path",
        type=Path,
        default=DEFAULT_FINAL_SYMBOLS_PATH,
    )
    return parser.parse_args()


def load_frame(csv_path: Path) -> pandas.DataFrame:
    """Load a CSV file and replace missing values with empty strings."""

    return pandas.read_csv(csv_path, keep_default_na=False).fillna("")


def load_policy_overrides(policy_override_path: Path) -> list[dict[str, str]]:
    """Load policy override rows from CSV."""

    if not policy_override_path.exists():
        return []
    override_frame = load_frame(policy_override_path)
    required_columns = {
        "match_field",
        "match_value",
        "override_decision",
        "override_semantic_type",
        "override_reason",
    }
    missing_columns = required_columns - set(override_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Policy override file missing columns: {sorted(missing_columns)}"
        )
    return [
        {column_name: str(value) for column_name, value in override_row.items()}
        for override_row in override_frame.to_dict("records")
    ]


def load_runtime_guardrail_classifications(
    runtime_guardrail_llm_path: Path,
) -> dict[str, dict[str, str]]:
    """Load symbol decisions from runtime guardrail conflict review."""

    if not runtime_guardrail_llm_path.exists():
        return {}
    runtime_guardrail_frame = load_frame(runtime_guardrail_llm_path)
    required_columns = {"symbol", "decision", "semantic_type", "confidence", "reason"}
    missing_columns = required_columns - set(runtime_guardrail_frame.columns)
    if missing_columns:
        raise ValueError(
            "Runtime guardrail LLM file missing columns: "
            f"{sorted(missing_columns)}"
        )
    classifications_by_symbol: dict[str, dict[str, str]] = {}
    for classification_record in runtime_guardrail_frame.to_dict("records"):
        symbol_name = str(classification_record["symbol"]).strip().upper()
        if not symbol_name:
            continue
        classifications_by_symbol[symbol_name] = {
            "decision": str(classification_record["decision"]),
            "semantic_type": str(classification_record["semantic_type"]),
            "confidence": str(classification_record["confidence"]),
            "reason": str(classification_record["reason"]),
        }
    return classifications_by_symbol


def apply_policy_override(
    audit_record: dict[str, Any],
    policy_overrides: list[dict[str, str]],
) -> dict[str, Any]:
    """Apply the first matching policy override to an audit record."""

    for policy_override in policy_overrides:
        match_field = policy_override["match_field"]
        match_value = policy_override["match_value"]
        if str(audit_record.get(match_field, "")) != match_value:
            continue
        audit_record["final_decision"] = policy_override["override_decision"]
        audit_record["decision_source"] = "policy_override"
        audit_record["semantic_type"] = policy_override["override_semantic_type"]
        audit_record["confidence"] = "policy"
        audit_record["reason"] = policy_override["override_reason"]
        return audit_record
    return audit_record


def apply_runtime_guardrail_classification(
    audit_record: dict[str, Any],
    runtime_guardrail_classifications: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Apply reviewed runtime-conflict decisions to a final audit record."""

    symbol_name = str(audit_record.get("symbol", "")).strip().upper()
    classification = runtime_guardrail_classifications.get(symbol_name)
    if classification is None:
        return audit_record
    audit_record["final_decision"] = classification["decision"]
    audit_record["decision_source"] = "runtime_guardrail_llm_review"
    audit_record["semantic_type"] = classification["semantic_type"]
    audit_record["confidence"] = classification["confidence"]
    audit_record["reason"] = classification["reason"]
    return audit_record


def build_final_audit_frame(
    hard_filter_audit_frame: pandas.DataFrame,
    second_layer_candidate_frame: pandas.DataFrame,
    llm_classification_frame: pandas.DataFrame,
    policy_overrides: list[dict[str, str]],
    runtime_guardrail_classifications: dict[str, dict[str, str]] | None = None,
) -> pandas.DataFrame:
    """Return final symbol decisions from all evidence layers."""

    runtime_guardrail_classifications = runtime_guardrail_classifications or {}
    second_layer_symbols = set(second_layer_candidate_frame["symbol"].astype(str))
    llm_decision_by_symbol = llm_classification_frame.set_index("symbol").to_dict(
        "index"
    )

    final_records: list[dict[str, Any]] = []
    for hard_filter_record in hard_filter_audit_frame.to_dict("records"):
        symbol_name = str(hard_filter_record["symbol"])
        security_title = str(hard_filter_record["sec_title"])
        hard_decision = str(hard_filter_record["hard_filter_decision"])
        hard_reason = str(hard_filter_record["hard_filter_reason"])

        if hard_decision == "exclude":
            final_record: dict[str, Any] = {
                "symbol": symbol_name,
                "sec_title": security_title,
                "final_decision": "exclude",
                "decision_source": "hard_filter",
                "hard_filter_reason": hard_reason,
                "llm_decision": "",
                "semantic_type": "",
                "confidence": "",
                "reason": hard_reason,
            }
        elif symbol_name in second_layer_symbols:
            llm_record = llm_decision_by_symbol.get(symbol_name, {})
            llm_decision = str(llm_record.get("decision", "quarantine"))
            final_record = {
                "symbol": symbol_name,
                "sec_title": security_title,
                "final_decision": llm_decision,
                "decision_source": "llm_second_layer",
                "hard_filter_reason": "",
                "llm_decision": llm_decision,
                "semantic_type": str(
                    llm_record.get("semantic_type", "missing_llm_result")
                ),
                "confidence": str(llm_record.get("confidence", "low")),
                "reason": str(
                    llm_record.get("reason", "Missing LLM classification")
                ),
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
            }
        final_record = apply_runtime_guardrail_classification(
            final_record,
            runtime_guardrail_classifications,
        )
        final_record = apply_policy_override(final_record, policy_overrides)
        final_records.append(final_record)
    return pandas.DataFrame(final_records)


def write_final_symbols(
    final_audit_frame: pandas.DataFrame,
    final_symbols_path: Path,
) -> None:
    """Write included symbols to a newline-separated text file."""

    final_symbols = sorted(
        symbol_name
        for symbol_name in final_audit_frame.loc[
            final_audit_frame["final_decision"] == "include", "symbol"
        ].astype(str)
        if symbol_name
    )
    final_symbols_path.parent.mkdir(parents=True, exist_ok=True)
    final_symbols_path.write_text("\n".join(final_symbols) + "\n", encoding="utf-8")


def main() -> None:
    """Build and write final universe audit files."""

    arguments = parse_arguments()
    hard_filter_audit_frame = load_frame(arguments.hard_filter_audit_path)
    second_layer_candidate_frame = load_frame(arguments.second_layer_candidate_path)
    llm_classification_frame = load_frame(arguments.llm_classification_path)
    policy_overrides = load_policy_overrides(arguments.policy_override_path)
    runtime_guardrail_classifications = load_runtime_guardrail_classifications(
        arguments.runtime_guardrail_llm_path,
    )

    final_audit_frame = build_final_audit_frame(
        hard_filter_audit_frame,
        second_layer_candidate_frame,
        llm_classification_frame,
        policy_overrides,
        runtime_guardrail_classifications,
    )
    arguments.final_audit_path.parent.mkdir(parents=True, exist_ok=True)
    final_audit_frame.to_csv(arguments.final_audit_path, index=False)
    write_final_symbols(final_audit_frame, arguments.final_symbols_path)

    print(final_audit_frame["final_decision"].value_counts().to_string())
    print(final_audit_frame["decision_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
