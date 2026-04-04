from collections import Counter, defaultdict
from typing import List, Dict, Any


REQUIRED_FIELDS = {
    "prompt_id",
    "base_example_id",
    "task_name",
    "distraction_type",
    "regime",
    "is_clean",
    "prompt_text",
    "gold_output",
}

EXPECTED_TASKS = {
    "single_label_classification",
    "multi_label_classification",
    "information_extraction",
    "rule_based_transformation",
    "extractive_qa",
}

EXPECTED_REGIMES = {"bounded", "unbounded"}

EXPECTED_DISTRACTION_TYPES = {
    "clean",
    "irrelevant_prefix",
    "irrelevant_suffix",
    "instruction_in_the_middle",
    "conflicting_instruction",
    "negation_distraction",
    "style_distraction",
    "length_stress",
}


def validate_prompt_instances(
    prompt_records: List[Dict[str, Any]],
    expected_total: int = 4000,
    expected_base_examples: int = 250,
    expected_per_base_example: int = 16,
) -> Dict[str, Any]:
    record_issues: Dict[str, List[str]] = {}
    issue_counts = Counter()
    dataset_level_issues = []

    prompt_ids = set()
    base_example_ids = set()
    counts_by_base = Counter()
    counts_by_regime = Counter()
    counts_by_distraction = Counter()
    counts_by_task = Counter()
    clean_count = 0
    distracted_count = 0

    for record in prompt_records:
        prompt_id = record.get("prompt_id", "<missing_prompt_id>")
        issues = []

        missing_fields = REQUIRED_FIELDS - set(record.keys())
        if missing_fields:
            issues.append(f"missing_fields:{sorted(missing_fields)}")

        if "prompt_id" in record:
            if record["prompt_id"] in prompt_ids:
                issues.append("duplicate_prompt_id")
            else:
                prompt_ids.add(record["prompt_id"])

        if "base_example_id" in record:
            base_example_ids.add(record["base_example_id"])
            counts_by_base[record["base_example_id"]] += 1

        if record.get("task_name") not in EXPECTED_TASKS:
            issues.append(f"invalid_task_name:{record.get('task_name')}")

        if record.get("regime") not in EXPECTED_REGIMES:
            issues.append(f"invalid_regime:{record.get('regime')}")
        else:
            counts_by_regime[record["regime"]] += 1

        if record.get("distraction_type") not in EXPECTED_DISTRACTION_TYPES:
            issues.append(f"invalid_distraction_type:{record.get('distraction_type')}")
        else:
            counts_by_distraction[record["distraction_type"]] += 1

        if "task_name" in record:
            counts_by_task[record["task_name"]] += 1

        if record.get("is_clean") is True:
            clean_count += 1
            if record.get("distraction_type") != "clean":
                issues.append("clean_flag_mismatch")
        else:
            distracted_count += 1
            if record.get("distraction_type") == "clean":
                issues.append("clean_flag_mismatch")

        prompt_text = record.get("prompt_text", "")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            issues.append("empty_prompt_text")

        if issues:
            record_issues[prompt_id] = issues
            for issue in issues:
                issue_counts[issue] += 1

    if len(prompt_records) != expected_total:
        dataset_level_issues.append(
            f"unexpected_total_prompt_instances:{len(prompt_records)}"
        )

    if len(base_example_ids) != expected_base_examples:
        dataset_level_issues.append(
            f"unexpected_unique_base_example_count:{len(base_example_ids)}"
        )

    for base_example_id, count in counts_by_base.items():
        if count != expected_per_base_example:
            dataset_level_issues.append(
                f"unexpected_prompt_count_for_base:{base_example_id}:{count}"
            )

    if counts_by_regime.get("bounded", 0) != expected_total // 2:
        dataset_level_issues.append(
            f"unexpected_bounded_count:{counts_by_regime.get('bounded', 0)}"
        )

    if counts_by_regime.get("unbounded", 0) != expected_total // 2:
        dataset_level_issues.append(
            f"unexpected_unbounded_count:{counts_by_regime.get('unbounded', 0)}"
        )

    if clean_count != expected_base_examples * 2:
        dataset_level_issues.append(f"unexpected_clean_count:{clean_count}")

    if distracted_count != expected_total - (expected_base_examples * 2):
        dataset_level_issues.append(f"unexpected_distracted_count:{distracted_count}")

    expected_counts_by_distraction = {
        "clean": expected_base_examples * 2,
        "irrelevant_prefix": expected_base_examples * 2,
        "irrelevant_suffix": expected_base_examples * 2,
        "instruction_in_the_middle": expected_base_examples * 2,
        "conflicting_instruction": expected_base_examples * 2,
        "negation_distraction": expected_base_examples * 2,
        "style_distraction": expected_base_examples * 2,
        "length_stress": expected_base_examples * 2,
    }

    for distraction_type, expected_count in expected_counts_by_distraction.items():
        actual_count = counts_by_distraction.get(distraction_type, 0)
        if actual_count != expected_count:
            dataset_level_issues.append(
                f"unexpected_count_for_distraction:{distraction_type}:{actual_count}"
            )

    is_valid = (
        len(record_issues) == 0 and
        len(dataset_level_issues) == 0
    )

    return {
        "is_valid": is_valid,
        "total_prompt_instances": len(prompt_records),
        "unique_base_example_ids": len(base_example_ids),
        "num_records_with_issues": len(record_issues),
        "record_issues": record_issues,
        "dataset_level_issues": dataset_level_issues,
        "issue_counts": dict(issue_counts),
        "counts_by_task": dict(counts_by_task),
        "counts_by_regime": dict(counts_by_regime),
        "counts_by_distraction_type": dict(counts_by_distraction),
        "clean_count": clean_count,
        "distracted_count": distracted_count,
    }