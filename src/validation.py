import re
from collections import Counter
from typing import List, Dict, Any

from src.templates import SINGLE_LABEL_SET, MULTI_LABEL_SET, IE_SCHEMA_KEYS, RULE_SET
from src.generation import apply_rule


EXPECTED_TASK_NAMES = {
    "single_label_classification",
    "multi_label_classification",
    "information_extraction",
    "rule_based_transformation",
    "extractive_qa",
}

# Check that the final base record contains all required top-level fields.
def validate_required_top_level_fields(record: Dict[str, Any]) -> List[str]:
    required_fields = {
        "example_id",
        "task_name",
        "template_name",
        "instruction",
        "input_data",
        "gold_output",
        "metadata",
    }

    issues = []
    missing = required_fields - set(record.keys())
    extra = set(record.keys()) - required_fields

    if missing:
        issues.append(f"missing_top_level_fields: {sorted(missing)}")
    if extra:
        issues.append(f"unexpected_top_level_fields: {sorted(extra)}")

    return issues


def validate_task_name(record: Dict[str, Any]) -> List[str]:
    issues = []
    if record["task_name"] not in EXPECTED_TASK_NAMES:
        issues.append(f"invalid_task_name: {record['task_name']}")
    return issues


def validate_single_label_record(record: Dict[str, Any]) -> List[str]:
    issues = []

    gold = record["gold_output"]
    input_data = record["input_data"]

    if set(gold.keys()) != {"label"}:
        issues.append("single_label_invalid_gold_schema")

    if "text" not in input_data:
        issues.append("single_label_missing_text_input")

    if "label" in gold and gold["label"] not in SINGLE_LABEL_SET:
        issues.append(f"single_label_invalid_label: {gold['label']}")

    return issues


def validate_multi_label_record(record: Dict[str, Any]) -> List[str]:
    issues = []

    gold = record["gold_output"]
    input_data = record["input_data"]

    if set(gold.keys()) != {"labels"}:
        issues.append("multi_label_invalid_gold_schema")

    if "text" not in input_data:
        issues.append("multi_label_missing_text_input")

    labels = gold.get("labels")
    if not isinstance(labels, list):
        issues.append("multi_label_labels_not_list")
        return issues

    if not all(isinstance(label, str) for label in labels):
        issues.append("multi_label_non_string_label")

    if any(label not in MULTI_LABEL_SET for label in labels):
        invalid = [label for label in labels if label not in MULTI_LABEL_SET]
        issues.append(f"multi_label_invalid_labels: {invalid}")

    if len(labels) != len(set(labels)):
        issues.append("multi_label_duplicate_labels")

    if labels != sorted(labels):
        issues.append("multi_label_labels_not_sorted")

    return issues


def validate_information_extraction_record(record: Dict[str, Any]) -> List[str]:
    issues = []

    gold = record["gold_output"]
    input_data = record["input_data"]

    if "text" not in input_data:
        issues.append("ie_missing_text_input")

    if set(gold.keys()) != set(IE_SCHEMA_KEYS):
        issues.append("ie_invalid_gold_schema")

    for key in IE_SCHEMA_KEYS:
        if key in gold and not isinstance(gold[key], str):
            issues.append(f"ie_non_string_value_for_{key}")

    return issues


def validate_rule_based_transformation_record(record: Dict[str, Any]) -> List[str]:
    issues = []

    gold = record["gold_output"]
    input_data = record["input_data"]
    metadata = record["metadata"]

    if "text" not in input_data:
        issues.append("transformation_missing_text_input")

    if set(gold.keys()) != {"text"}:
        issues.append("transformation_invalid_gold_schema")

    rule_name = metadata.get("rule_name")
    if rule_name not in RULE_SET:
        issues.append(f"transformation_invalid_rule_name: {rule_name}")
        return issues

    source_text = input_data.get("text", "")
    expected_output = apply_rule(source_text, rule_name)
    actual_output = gold.get("text")

    if actual_output != expected_output:
        issues.append("transformation_non_deterministic_or_incorrect_gold")

    return issues


def validate_extractive_qa_record(record: Dict[str, Any]) -> List[str]:
    issues = []

    gold = record["gold_output"]
    input_data = record["input_data"]

    if "passage" not in input_data:
        issues.append("qa_missing_passage_input")
    if "question" not in input_data:
        issues.append("qa_missing_question_input")

    if set(gold.keys()) != {"answer"}:
        issues.append("qa_invalid_gold_schema")

    passage = input_data.get("passage", "")
    answer = gold.get("answer", "")

    if not isinstance(answer, str):
        issues.append("qa_answer_not_string")
        return issues

    if answer not in passage:
        issues.append("qa_answer_not_in_passage")

    if passage.count(answer) != 1:
        issues.append("qa_answer_not_unique_span")

    return issues

#  Run all relevant validations for one record.
def validate_record(record: Dict[str, Any]) -> List[str]:
    issues = []
    issues.extend(validate_required_top_level_fields(record))
    issues.extend(validate_task_name(record))

    task_name = record.get("task_name")

    if task_name == "single_label_classification":
        issues.extend(validate_single_label_record(record))
    elif task_name == "multi_label_classification":
        issues.extend(validate_multi_label_record(record))
    elif task_name == "information_extraction":
        issues.extend(validate_information_extraction_record(record))
    elif task_name == "rule_based_transformation":
        issues.extend(validate_rule_based_transformation_record(record))
    elif task_name == "extractive_qa":
        issues.extend(validate_extractive_qa_record(record))

    return issues

# Check that all example IDs are unique.
def validate_unique_ids(records: List[Dict[str, Any]]) -> List[str]:
    ids = [record["example_id"] for record in records]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]

    if duplicates:
        return [f"duplicate_example_ids: {duplicates}"]
    return []

# Check that each task has exactly expected_per_task records.
def validate_task_counts(records: List[Dict[str, Any]], expected_per_task: int = 50) -> List[str]:
    counts = Counter(record["task_name"] for record in records)
    issues = []

    for task_name in EXPECTED_TASK_NAMES:
        observed = counts.get(task_name, 0)
        if observed != expected_per_task:
            issues.append(
                f"task_count_mismatch:{task_name}:expected={expected_per_task}:observed={observed}"
            )

    return issues

# Check whether instructions are standardized within each task family.
def validate_instruction_consistency(records: List[Dict[str, Any]]) -> List[str]:
    issues = []
    instructions_by_task: Dict[str, set] = {}

    for record in records:
        instructions_by_task.setdefault(record["task_name"], set()).add(record["instruction"])

    for task_name, instructions in instructions_by_task.items():
        if len(instructions) > 1:
            issues.append(
                f"instruction_inconsistency:{task_name}:num_unique_instructions={len(instructions)}"
            )

    return issues

# Full validation for the clean base dataset.
def validate_dataset(records: List[Dict[str, Any]], expected_total: int = 250) -> Dict[str, Any]:
    record_issues: Dict[str, List[str]] = {}
    all_issue_counts = Counter()

    for record in records:
        issues = validate_record(record)
        if issues:
            record_issues[record["example_id"]] = issues
            for issue in issues:
                all_issue_counts[issue] += 1

    dataset_level_issues = []
    dataset_level_issues.extend(validate_unique_ids(records))
    dataset_level_issues.extend(validate_task_counts(records, expected_per_task=50))
    dataset_level_issues.extend(validate_instruction_consistency(records))

    if len(records) != expected_total:
        dataset_level_issues.append(
            f"total_record_count_mismatch:expected={expected_total}:observed={len(records)}"
        )

    for issue in dataset_level_issues:
        all_issue_counts[issue] += 1

    is_valid = (len(record_issues) == 0) and (len(dataset_level_issues) == 0)

    return {
        "is_valid": is_valid,
        "total_records": len(records),
        "num_records_with_issues": len(record_issues),
        "record_issues": record_issues,
        "dataset_level_issues": dataset_level_issues,
        "issue_counts": dict(all_issue_counts),
    }