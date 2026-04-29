from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


TASK_NAMES = {
    "single_label_classification",
    "multi_label_classification",
    "information_extraction",
    "rule_based_transformation",
    "extractive_qa",
}


@dataclass
class ParseResult:
    """Structured result of parsing a model output for one benchmark instance."""

    success: bool
    task_name: str
    parsed_output: Optional[Dict[str, Any]]
    error_code: Optional[str]
    error_detail: Optional[str]
    raw_output: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_name": self.task_name,
            "parsed_output": self.parsed_output,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "raw_output": self.raw_output,
        }


def _make_success(task_name: str, raw_output: str, parsed_output: Dict[str, Any]) -> ParseResult:
    return ParseResult(
        success=True,
        task_name=task_name,
        parsed_output=parsed_output,
        error_code=None,
        error_detail=None,
        raw_output=raw_output,
    )


def _make_failure(task_name: str, raw_output: str, error_code: str, error_detail: str) -> ParseResult:
    return ParseResult(
        success=False,
        task_name=task_name,
        parsed_output=None,
        error_code=error_code,
        error_detail=error_detail,
        raw_output=raw_output,
    )


def _parse_json_object(raw_output: str, task_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[ParseResult]]:
    text = raw_output.strip()
    if not text:
        return None, _make_failure(task_name, raw_output, "empty_output", "Model returned empty output.")

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, _make_failure(task_name, raw_output, "invalid_json", f"JSON decode error: {exc.msg} at line {exc.lineno} column {exc.colno}.")

    if not isinstance(value, dict):
        return None, _make_failure(task_name, raw_output, "wrong_schema", "Top-level JSON value must be an object.")

    return value, None


def _require_exact_keys(
    obj: Dict[str, Any],
    expected_keys: List[str],
    task_name: str,
    raw_output: str,
) -> Optional[ParseResult]:
    actual_keys = set(obj.keys())
    expected = set(expected_keys)

    missing = sorted(expected - actual_keys)
    extra = sorted(actual_keys - expected)

    if missing:
        return _make_failure(task_name, raw_output, "missing_key", f"Missing required key(s): {missing}")

    if extra:
        return _make_failure(task_name, raw_output, "extra_key", f"Unexpected key(s): {extra}")

    return None


def _normalize_string_value(value: Any, field_name: str, task_name: str, raw_output: str) -> Tuple[Optional[str], Optional[ParseResult]]:
    if not isinstance(value, str):
        return None, _make_failure(task_name, raw_output, "wrong_type", f"Field '{field_name}' must be a string.")
    return value.strip(), None


def parse_single_label(raw_output: str, allowed_labels: Optional[List[str]] = None) -> ParseResult:
    task_name = "single_label_classification"
    obj, failure = _parse_json_object(raw_output, task_name)
    if failure:
        return failure

    key_failure = _require_exact_keys(obj, ["label"], task_name, raw_output)
    if key_failure:
        return key_failure

    label, value_failure = _normalize_string_value(obj.get("label"), "label", task_name, raw_output)
    if value_failure:
        return value_failure

    if allowed_labels is not None and label not in allowed_labels:
        return _make_failure(task_name, raw_output, "invalid_label", f"Label '{label}' is not in the allowed set: {allowed_labels}")

    return _make_success(task_name, raw_output, {"label": label})


def parse_multi_label(raw_output: str, allowed_labels: Optional[List[str]] = None) -> ParseResult:
    task_name = "multi_label_classification"
    obj, failure = _parse_json_object(raw_output, task_name)
    if failure:
        return failure

    key_failure = _require_exact_keys(obj, ["labels"], task_name, raw_output)
    if key_failure:
        return key_failure

    labels = obj.get("labels")
    if not isinstance(labels, list):
        return _make_failure(task_name, raw_output, "wrong_type", "Field 'labels' must be a list.")

    normalized: List[str] = []
    for idx, label in enumerate(labels):
        if not isinstance(label, str):
            return _make_failure(task_name, raw_output, "wrong_type", f"Label at index {idx} must be a string.")
        normalized.append(label.strip())

    if allowed_labels is not None:
        invalid = [label for label in normalized if label not in allowed_labels]
        if invalid:
            return _make_failure(task_name, raw_output, "invalid_label", f"Invalid labels not in allowed set: {invalid}")

    canonical = sorted(set(normalized))
    return _make_success(task_name, raw_output, {"labels": canonical})


def parse_information_extraction(raw_output: str) -> ParseResult:
    task_name = "information_extraction"
    obj, failure = _parse_json_object(raw_output, task_name)
    if failure:
        return failure

    expected_keys = ["person", "date", "location"]
    key_failure = _require_exact_keys(obj, expected_keys, task_name, raw_output)
    if key_failure:
        return key_failure

    parsed: Dict[str, str] = {}
    for key in expected_keys:
        value, value_failure = _normalize_string_value(obj.get(key), key, task_name, raw_output)
        if value_failure:
            return value_failure
        parsed[key] = value

    return _make_success(task_name, raw_output, parsed)


def parse_rule_based_transformation(raw_output: str) -> ParseResult:
    task_name = "rule_based_transformation"
    obj, failure = _parse_json_object(raw_output, task_name)
    if failure:
        return failure

    key_failure = _require_exact_keys(obj, ["text"], task_name, raw_output)
    if key_failure:
        return key_failure

    text, value_failure = _normalize_string_value(obj.get("text"), "text", task_name, raw_output)
    if value_failure:
        return value_failure

    return _make_success(task_name, raw_output, {"text": text})


def parse_extractive_qa(raw_output: str) -> ParseResult:
    task_name = "extractive_qa"
    obj, failure = _parse_json_object(raw_output, task_name)
    if failure:
        return failure

    key_failure = _require_exact_keys(obj, ["answer"], task_name, raw_output)
    if key_failure:
        return key_failure

    answer, value_failure = _normalize_string_value(obj.get("answer"), "answer", task_name, raw_output)
    if value_failure:
        return value_failure

    return _make_success(task_name, raw_output, {"answer": answer})


def get_allowed_labels(instance_record: Dict[str, Any], task_name: str) -> Optional[List[str]]:
    metadata = instance_record.get("metadata") or {}
    gold_output = instance_record.get("gold_output") or {}

    if task_name == "single_label_classification":
        allowed = metadata.get("label_set") or metadata.get("allowed_labels")
        if isinstance(allowed, list):
            return allowed
        gold_label = gold_output.get("label")
        return [gold_label] if isinstance(gold_label, str) else None

    if task_name == "multi_label_classification":
        allowed = metadata.get("label_set") or metadata.get("allowed_labels")
        if isinstance(allowed, list):
            return allowed
        gold_labels = gold_output.get("labels")
        return gold_labels if isinstance(gold_labels, list) else None

    return None


def parse_prediction(raw_output: str, task_name: str, instance_record: Optional[Dict[str, Any]] = None) -> ParseResult:
    if task_name not in TASK_NAMES:
        return _make_failure(task_name, raw_output, "invalid_task_name", f"Unsupported task_name: {task_name}")

    if task_name == "single_label_classification":
        allowed = get_allowed_labels(instance_record or {}, task_name) if instance_record else None
        return parse_single_label(raw_output, allowed_labels=allowed)

    if task_name == "multi_label_classification":
        allowed = get_allowed_labels(instance_record or {}, task_name) if instance_record else None
        return parse_multi_label(raw_output, allowed_labels=allowed)

    if task_name == "information_extraction":
        return parse_information_extraction(raw_output)

    if task_name == "rule_based_transformation":
        return parse_rule_based_transformation(raw_output)

    if task_name == "extractive_qa":
        return parse_extractive_qa(raw_output)

    return _make_failure(task_name, raw_output, "invalid_task_name", f"Unsupported task_name: {task_name}")
