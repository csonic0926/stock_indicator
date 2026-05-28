"""Classify ambiguous symbol-universe candidates with a local LLM proxy."""
# TODO: review

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas
import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_CANDIDATE_AUDIT_PATH = Path("data/symbol_second_layer_candidate_audit.csv")
DEFAULT_PROMPT_PATH = Path("data/symbol_universe_llm_prompt.md")
DEFAULT_SCHEMA_PATH = Path("data/symbol_universe_llm_schema.json")
DEFAULT_OUTPUT_CSV_PATH = Path("data/symbol_universe_llm_classification.csv")
DEFAULT_OUTPUT_JSONL_PATH = Path("data/symbol_universe_llm_classification_batches.jsonl")
DEFAULT_PROXY_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_MODEL_NAME = "personal/gpt-5.4-mini"
DEFAULT_BATCH_SIZE = 100
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_RETRY_COUNT = 2
DEFAULT_SLEEP_BETWEEN_BATCHES_SECONDS = 1.0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the classifier."""

    parser = argparse.ArgumentParser(
        description="Classify second-layer symbol candidates with local CLI Proxy."
    )
    parser.add_argument(
        "--candidate-audit-path",
        type=Path,
        default=DEFAULT_CANDIDATE_AUDIT_PATH,
    )
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output-csv-path", type=Path, default=DEFAULT_OUTPUT_CSV_PATH)
    parser.add_argument(
        "--output-jsonl-path",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL_PATH,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--model-name",
        default=os.environ.get("CLI_PROXY_MODEL", DEFAULT_MODEL_NAME),
    )
    parser.add_argument(
        "--proxy-base-url",
        default=os.environ.get("CLI_PROXY_BASE_URL", DEFAULT_PROXY_BASE_URL),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


def load_candidate_records(candidate_audit_path: Path) -> list[dict[str, str]]:
    """Load symbol and SEC title records for LLM classification."""

    candidate_frame = pandas.read_csv(candidate_audit_path).fillna("")
    required_columns = {"symbol", "sec_title"}
    missing_columns = required_columns - set(candidate_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Candidate audit missing required columns: {sorted(missing_columns)}"
        )
    return [
        {
            "symbol": str(candidate_record["symbol"]),
            "sec_title": str(candidate_record["sec_title"]),
        }
        for candidate_record in candidate_frame.to_dict("records")
    ]


def chunk_records(
    records: list[dict[str, str]], batch_size: int
) -> list[list[dict[str, str]]]:
    """Split records into stable batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        records[batch_start : batch_start + batch_size]
        for batch_start in range(0, len(records), batch_size)
    ]


def build_batch_prompt(
    prompt_text: str,
    batch_records: list[dict[str, str]],
) -> str:
    """Return the full prompt for a single classification batch."""

    return (
        prompt_text
        + "\n\nInput records:\n"
        + json.dumps(batch_records, ensure_ascii=False, indent=2)
    )


def extract_response_output_text(response_payload: dict[str, Any]) -> str:
    """Extract output text from a Responses API payload."""

    output_text_fragments: list[str] = []
    for output_item in response_payload.get("output", []):
        if output_item.get("type") != "message":
            continue
        for content_item in output_item.get("content", []):
            if content_item.get("type") == "output_text":
                output_text_fragments.append(str(content_item.get("text", "")))
    output_text = "".join(output_text_fragments).strip()
    if not output_text:
        raise ValueError("LLM response did not contain output_text")
    return output_text


def call_llm_proxy(
    *,
    proxy_base_url: str,
    model_name: str,
    api_key: str,
    prompt_text: str,
    response_schema: dict[str, Any],
    request_timeout_seconds: int,
) -> dict[str, Any]:
    """Call the local CLI Proxy Responses endpoint once."""

    request_body = {
        "model": model_name,
        "input": prompt_text,
        "reasoning": {"effort": "low"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "symbol_classification",
                "schema": response_schema,
                "strict": True,
            }
        },
    }
    response = requests.post(
        f"{proxy_base_url.rstrip('/')}/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_body,
        timeout=request_timeout_seconds,
    )
    response.raise_for_status()
    response_payload = response.json()
    if response_payload.get("status") != "completed":
        raise ValueError(f"LLM response did not complete: {response_payload}")
    return response_payload


def classify_batch_with_retries(
    *,
    batch_records: list[dict[str, str]],
    batch_number: int,
    prompt_text: str,
    response_schema: dict[str, Any],
    proxy_base_url: str,
    model_name: str,
    api_key: str,
    request_timeout_seconds: int,
) -> dict[str, Any]:
    """Classify one batch and retry transient failures."""

    batch_prompt = build_batch_prompt(prompt_text, batch_records)
    last_error: Exception | None = None
    for attempt_number in range(1, DEFAULT_RETRY_COUNT + 2):
        try:
            response_payload = call_llm_proxy(
                proxy_base_url=proxy_base_url,
                model_name=model_name,
                api_key=api_key,
                prompt_text=batch_prompt,
                response_schema=response_schema,
                request_timeout_seconds=request_timeout_seconds,
            )
            output_text = extract_response_output_text(response_payload)
            parsed_output = json.loads(output_text)
            output_records = parsed_output.get("records", [])
            validate_batch_output(batch_records, output_records, batch_number)
            return {
                "batch_number": batch_number,
                "response_id": response_payload.get("id", ""),
                "model": response_payload.get("model", model_name),
                "records": output_records,
            }
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            last_error = error
            LOGGER.warning(
                "Batch %d attempt %d failed: %s",
                batch_number,
                attempt_number,
                error,
            )
            time.sleep(float(attempt_number))
    raise RuntimeError(f"Batch {batch_number} failed") from last_error


def validate_batch_output(
    batch_records: list[dict[str, str]],
    output_records: list[dict[str, str]],
    batch_number: int,
) -> None:
    """Validate that LLM output covers exactly the input symbols."""

    input_symbols = [record["symbol"] for record in batch_records]
    output_symbols = [str(record.get("symbol", "")) for record in output_records]
    if len(output_records) != len(batch_records):
        raise ValueError(
            f"Batch {batch_number} output count {len(output_records)} "
            f"does not match input count {len(batch_records)}"
        )
    if sorted(output_symbols) != sorted(input_symbols):
        raise ValueError(
            f"Batch {batch_number} output symbols do not match input symbols"
        )


def write_batch_jsonl(
    output_jsonl_path: Path,
    batch_result: dict[str, Any],
) -> None:
    """Append a parsed batch result to a JSONL file."""

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(batch_result, ensure_ascii=False) + "\n")


def merge_classifications_with_candidates(
    candidate_records: list[dict[str, str]],
    classification_records: list[dict[str, str]],
) -> pandas.DataFrame:
    """Return a CSV-ready frame with input evidence and LLM decisions."""

    candidate_frame = pandas.DataFrame(candidate_records)
    classification_frame = pandas.DataFrame(classification_records)
    merged_frame = candidate_frame.merge(
        classification_frame,
        on="symbol",
        how="left",
        validate="one_to_one",
    )
    return merged_frame[
        [
            "symbol",
            "sec_title",
            "decision",
            "semantic_type",
            "confidence",
            "reason",
        ]
    ]


def main() -> None:
    """Run LLM classification for all second-layer symbol candidates."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_arguments()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the local CLI Proxy")

    prompt_text = arguments.prompt_path.read_text(encoding="utf-8")
    response_schema = json.loads(arguments.schema_path.read_text(encoding="utf-8"))
    candidate_records = load_candidate_records(arguments.candidate_audit_path)
    record_batches = chunk_records(candidate_records, arguments.batch_size)

    if arguments.output_jsonl_path.exists():
        arguments.output_jsonl_path.unlink()

    classification_records: list[dict[str, str]] = []
    LOGGER.info(
        "Classifying %d records in %d batches using %s",
        len(candidate_records),
        len(record_batches),
        arguments.model_name,
    )
    for batch_index, batch_records in enumerate(record_batches, start=1):
        LOGGER.info(
            "Classifying batch %d/%d (%d records)",
            batch_index,
            len(record_batches),
            len(batch_records),
        )
        batch_result = classify_batch_with_retries(
            batch_records=batch_records,
            batch_number=batch_index,
            prompt_text=prompt_text,
            response_schema=response_schema,
            proxy_base_url=arguments.proxy_base_url,
            model_name=arguments.model_name,
            api_key=api_key,
            request_timeout_seconds=arguments.request_timeout_seconds,
        )
        write_batch_jsonl(arguments.output_jsonl_path, batch_result)
        classification_records.extend(batch_result["records"])
        time.sleep(DEFAULT_SLEEP_BETWEEN_BATCHES_SECONDS)

    output_frame = merge_classifications_with_candidates(
        candidate_records,
        classification_records,
    )
    arguments.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(arguments.output_csv_path, index=False)
    LOGGER.info("Wrote %s", arguments.output_csv_path)


if __name__ == "__main__":
    main()
