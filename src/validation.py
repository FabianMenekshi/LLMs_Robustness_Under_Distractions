'''
If generation.py creates candidate examples, then validation.py checks whether those examples are 
structurally correct, internally consistent, and balanced enough to be trusted as the base benchmark.

It does that at two levels:

    1.  Record-level validation
        Check each example individually:
        -   does it have the required fields?
        -   is its schema correct for its task?
        -   is the gold output valid?
        -   is the QA answer unique?
        -   is the transformation output correct?
    2.  Dataset-level validation
        Check the whole dataset:
        -   are IDs unique?
        -   do we have 50 examples per task?
        -   are there duplicate rendered inputs?
        -   do we have enough instruction diversity?
        -   do we have enough template diversity?
        -   is any template too dominant?
        -   does QA have enough answer-field diversity?

'''


from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

from src.templates import SINGLE_LABEL_SET, MULTI_LABEL_SET, IE_SCHEMA_KEYS, RULE_SET
from src.generation import apply_rule


EXPECTED_TASK_NAMES = {
    "single_label_classification",
    "multi_label_classification",
    "information_extraction",
    "rule_based_transformation",
    "extractive_qa",
}


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
        issues.append(f"missing_top_level_fields:{sorted(missing)}")
    if extra:
        issues.append(f"unexpected_top_level_fields:{sorted(extra)}")

    return issues


def validate_task_name(record: Dict[str, Any]) -> List[str]:
    issues = []
    task_name = record.get("task_name")
    if task_name not in EXPECTED_TASK_NAMES:
        issues.append(f"invalid_task_name:{task_name}")
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
        issues.append(f"single_label_invalid_label:{gold['label']}")

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
        issues.append(f"multi_label_invalid_labels:{invalid}")

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
        issues.append(f"transformation_invalid_rule_name:{rule_name}")
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


def validate_unique_ids(records: List[Dict[str, Any]]) -> List[str]:
    ids = [record["example_id"] for record in records]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]

    if duplicates:
        return [f"duplicate_example_ids:{duplicates}"]
    return []


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


def validate_no_exact_duplicate_inputs(records: List[Dict[str, Any]]) -> List[str]:
    seen = {}
    duplicates = []

    for record in records:
        task_name = record["task_name"]

        if task_name == "extractive_qa":
            rendered = (
                f"PASSAGE: {record['input_data'].get('passage', '')}\n"
                f"QUESTION: {record['input_data'].get('question', '')}"
            )
        else:
            rendered = record["input_data"].get("text", "")

        key = (task_name, rendered)

        if key in seen:
            duplicates.append((seen[key], record["example_id"]))
        else:
            seen[key] = record["example_id"]

    if duplicates:
        return [f"duplicate_rendered_inputs:{duplicates}"]
    return []


def summarize_instruction_diversity(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    instructions_by_task: Dict[str, set] = {}

    for record in records:
        instructions_by_task.setdefault(record["task_name"], set()).add(record["instruction"])

    return {
        task_name: {
            "num_unique_instructions": len(instructions),
            "instructions": sorted(instructions),
        }
        for task_name, instructions in instructions_by_task.items()
    }


def validate_minimum_instruction_diversity(
    records: List[Dict[str, Any]],
    min_unique_per_task: int = 4,
) -> List[str]:
    issues = []
    instructions_by_task: Dict[str, set] = {}

    for record in records:
        instructions_by_task.setdefault(record["task_name"], set()).add(record["instruction"])

    for task_name in EXPECTED_TASK_NAMES:
        observed = len(instructions_by_task.get(task_name, set()))
        if observed < min_unique_per_task:
            issues.append(
                f"instruction_diversity_too_low:{task_name}:minimum={min_unique_per_task}:observed={observed}"
            )

    return issues


def validate_instruction_non_empty(records: List[Dict[str, Any]]) -> List[str]:
    issues = []

    for record in records:
        instruction = record.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            issues.append(f"empty_instruction:{record.get('example_id', '<missing_id>')}")

    return issues


def summarize_template_distribution(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    templates_by_task: Dict[str, Counter] = defaultdict(Counter)

    for record in records:
        templates_by_task[record["task_name"]][record["template_name"]] += 1

    return {
        task_name: dict(sorted(counter.items(), key=lambda item: item[0]))
        for task_name, counter in sorted(templates_by_task.items(), key=lambda item: item[0])
    }


def validate_minimum_template_diversity(
    records: List[Dict[str, Any]],
    minimums_by_task: Dict[str, int] | None = None,
) -> List[str]:
    """
    Soft structural diversity check.

    Default thresholds reflect what the current generator is expected to support:
    - single_label_classification: at least 4 templates
    - multi_label_classification: at least 2 templates
    - information_extraction: at least 5 templates
    - rule_based_transformation: at least 8 template-rule combinations
    - extractive_qa: at least 6 templates
    """
    if minimums_by_task is None:
        minimums_by_task = {
            "single_label_classification": 4,
            "multi_label_classification": 2,
            "information_extraction": 5,
            "rule_based_transformation": 8,
            "extractive_qa": 6,
        }

    issues = []
    templates_by_task: Dict[str, set] = defaultdict(set)

    for record in records:
        templates_by_task[record["task_name"]].add(record["template_name"])

    for task_name in EXPECTED_TASK_NAMES:
        observed = len(templates_by_task.get(task_name, set()))
        minimum = minimums_by_task.get(task_name, 1)
        if observed < minimum:
            issues.append(
                f"template_diversity_too_low:{task_name}:minimum={minimum}:observed={observed}"
            )

    return issues


def validate_template_concentration(
    records: List[Dict[str, Any]],
    max_share_by_task: Dict[str, float] | None = None,
) -> List[str]:
    """
    Warn if a single template dominates too much of a task's final selected set.
    """
    if max_share_by_task is None:
        max_share_by_task = {
            "single_label_classification": 0.40,
            "multi_label_classification": 0.60,
            "information_extraction": 0.35,
            "rule_based_transformation": 0.20,
            "extractive_qa": 0.20,
        }

    issues = []
    templates_by_task: Dict[str, Counter] = defaultdict(Counter)
    task_counts = Counter(record["task_name"] for record in records)

    for record in records:
        templates_by_task[record["task_name"]][record["template_name"]] += 1

    for task_name in sorted(templates_by_task.keys()):
        total = task_counts[task_name]
        allowed_share = max_share_by_task.get(task_name, 1.0)

        for template_name, count in templates_by_task[task_name].items():
            share = count / total if total else 0.0
            if share > allowed_share:
                issues.append(
                    f"template_concentration_too_high:{task_name}:{template_name}:share={share:.3f}:max={allowed_share:.3f}"
                )

    return issues


def summarize_qa_answer_field_distribution(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    qa_records = [record for record in records if record["task_name"] == "extractive_qa"]

    overall = Counter()
    by_template: Dict[str, Counter] = defaultdict(Counter)

    for record in qa_records:
        answer_field = record.get("metadata", {}).get("answer_field", "<missing>")
        template_name = record["template_name"]
        overall[answer_field] += 1
        by_template[template_name][answer_field] += 1

    return {
        "overall": dict(sorted(overall.items(), key=lambda item: item[0])),
        "by_template": {
            template_name: dict(sorted(counter.items(), key=lambda item: item[0]))
            for template_name, counter in sorted(by_template.items(), key=lambda item: item[0])
        },
    }


def validate_qa_answer_field_diversity(
    records: List[Dict[str, Any]],
    min_unique_answer_fields: int = 4,
) -> List[str]:
    issues = []

    qa_records = [record for record in records if record["task_name"] == "extractive_qa"]
    if not qa_records:
        return issues

    answer_fields = {
        record.get("metadata", {}).get("answer_field", "<missing>")
        for record in qa_records
    }

    observed = len(answer_fields)
    if observed < min_unique_answer_fields:
        issues.append(
            f"qa_answer_field_diversity_too_low:minimum={min_unique_answer_fields}:observed={observed}"
        )

    return issues


def validate_qa_answer_field_concentration(
    records: List[Dict[str, Any]],
    max_share: float = 0.45,
) -> List[str]:
    issues = []

    qa_records = [record for record in records if record["task_name"] == "extractive_qa"]
    if not qa_records:
        return issues

    counts = Counter(
        record.get("metadata", {}).get("answer_field", "<missing>")
        for record in qa_records
    )
    total = len(qa_records)

    for answer_field, count in counts.items():
        share = count / total if total else 0.0
        if share > max_share:
            issues.append(
                f"qa_answer_field_concentration_too_high:{answer_field}:share={share:.3f}:max={max_share:.3f}"
            )

    return issues


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
    dataset_level_issues.extend(validate_no_exact_duplicate_inputs(records))
    dataset_level_issues.extend(validate_instruction_non_empty(records))
    dataset_level_issues.extend(
        validate_minimum_instruction_diversity(records, min_unique_per_task=4)
    )

    # New balance-aware dataset checks
    dataset_level_issues.extend(validate_minimum_template_diversity(records))
    dataset_level_issues.extend(validate_template_concentration(records))
    dataset_level_issues.extend(validate_qa_answer_field_diversity(records, min_unique_answer_fields=4))
    dataset_level_issues.extend(validate_qa_answer_field_concentration(records, max_share=0.45))

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
        "instruction_diversity_summary": summarize_instruction_diversity(records),
        "template_distribution_summary": summarize_template_distribution(records),
        "qa_answer_field_summary": summarize_qa_answer_field_distribution(records),
    }