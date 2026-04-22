'''
Previously, we created the 4000 prompt instances, now prompt_instance_validation.py checks whether those prompt instances are:
    - structurally well-formed
    - internally consistent
    - complete across all expected conditions
    - correctly annotated with surface and distraction metadata

So this file is the prompt-level equivalent of what validation.py was for the base dataset.
But it is even stricter in some ways, because prompt instances carry much richer metadata.
'''

from collections import Counter
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

# These are required in the redesigned prompt schema.
REQUIRED_METADATA_FIELDS = {
    "source_instruction",
    "source_template_name",
    "prompt_surface_type",
    "placement",
}

# These are optional-but-expected fields in the richer redesigned schema.
OPTIONAL_METADATA_FIELDS = {
    "distraction_subtype",
    "distraction_variant_id",
    "surface_id",
    "surface_name",
    "opener_id",
    "opener_text",
    "layout_id",
    "layout_name",
    "noise_block_id",
    "noise_category",
    "noise_length",
    "noise_block_id_2",
    "noise_category_2",
    "long_noise_block_id",
    "long_noise_category",
    "long_noise_length",
    "conflict_variant_id",
    "conflict_subtype",
    "negation_variant_id",
    "negation_subtype",
    "style_id",
    "style_family",
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


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_required_fields(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    missing_fields = REQUIRED_FIELDS - set(record.keys())
    if missing_fields:
        issues.append(f"missing_fields:{sorted(missing_fields)}")

    return issues


def _validate_required_metadata_fields(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    missing_fields = REQUIRED_METADATA_FIELDS - set(record.keys())
    if missing_fields:
        issues.append(f"missing_metadata_fields:{sorted(missing_fields)}")

    for field_name in REQUIRED_METADATA_FIELDS:
        if field_name not in record:
            continue

        value = record[field_name]

        if field_name in {"source_instruction", "prompt_surface_type"}:
            if not _is_non_empty_string(value):
                issues.append(f"invalid_required_metadata_field:{field_name}")

        elif field_name == "placement":
            # placement is None for clean prompts and must be a non-empty string for distracted prompts
            if record.get("distraction_type") == "clean":
                if value is not None:
                    issues.append("clean_prompt_should_not_have_placement")
            else:
                if not _is_non_empty_string(value):
                    issues.append("distracted_prompt_missing_placement")

        elif field_name == "source_template_name":
            # Allow None, but if present it must be a non-empty string.
            if value is not None and not _is_non_empty_string(value):
                issues.append(f"invalid_required_metadata_field:{field_name}")

    return issues


def _validate_basic_values(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    if record.get("task_name") not in EXPECTED_TASKS:
        issues.append(f"invalid_task_name:{record.get('task_name')}")

    if record.get("regime") not in EXPECTED_REGIMES:
        issues.append(f"invalid_regime:{record.get('regime')}")

    if record.get("distraction_type") not in EXPECTED_DISTRACTION_TYPES:
        issues.append(f"invalid_distraction_type:{record.get('distraction_type')}")

    if not _is_non_empty_string(record.get("prompt_id")):
        issues.append("invalid_prompt_id")

    if not _is_non_empty_string(record.get("base_example_id")):
        issues.append("invalid_base_example_id")

    prompt_text = record.get("prompt_text")
    if not _is_non_empty_string(prompt_text):
        issues.append("empty_prompt_text")

    gold_output = record.get("gold_output")
    if not isinstance(gold_output, dict) or not gold_output:
        issues.append("invalid_gold_output")

    return issues


def _validate_clean_flag(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    is_clean = record.get("is_clean")
    distraction_type = record.get("distraction_type")

    if not isinstance(is_clean, bool):
        issues.append("is_clean_not_boolean")
        return issues

    if is_clean and distraction_type != "clean":
        issues.append("clean_flag_mismatch")

    if not is_clean and distraction_type == "clean":
        issues.append("clean_flag_mismatch")

    return issues


def _validate_surface_metadata(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    regime = record.get("regime")
    prompt_surface_type = record.get("prompt_surface_type")

    if regime == "bounded":
        if prompt_surface_type != "bounded_tagged":
            issues.append("invalid_bounded_prompt_surface_type")

        if not _is_non_empty_string(record.get("opener_id")):
            issues.append("bounded_missing_opener_id")

        if not _is_non_empty_string(record.get("opener_text")):
            issues.append("bounded_missing_opener_text")

        if not _is_non_empty_string(record.get("layout_id")):
            issues.append("bounded_missing_layout_id")

        if not _is_non_empty_string(record.get("layout_name")):
            issues.append("bounded_missing_layout_name")

    elif regime == "unbounded":
        if not _is_non_empty_string(prompt_surface_type):
            issues.append("unbounded_missing_prompt_surface_type")

        if not _is_non_empty_string(record.get("surface_id")):
            issues.append("unbounded_missing_surface_id")

        if not _is_non_empty_string(record.get("surface_name")):
            issues.append("unbounded_missing_surface_name")

    return issues


def _validate_distraction_specific_metadata(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    distraction_type = record.get("distraction_type")
    is_clean = record.get("is_clean")

    noise_block_id = record.get("noise_block_id")
    noise_block_id_2 = record.get("noise_block_id_2")
    long_noise_block_id = record.get("long_noise_block_id")
    conflict_variant_id = record.get("conflict_variant_id")
    negation_variant_id = record.get("negation_variant_id")
    style_id = record.get("style_id")
    distraction_subtype = record.get("distraction_subtype")
    distraction_variant_id = record.get("distraction_variant_id")

    if is_clean:
        # Clean prompts should not have distraction-specific content.
        forbidden_non_null_fields = [
            "distraction_subtype",
            "distraction_variant_id",
            "noise_block_id",
            "noise_block_id_2",
            "long_noise_block_id",
            "conflict_variant_id",
            "conflict_subtype",
            "negation_variant_id",
            "negation_subtype",
            "style_id",
            "style_family",
        ]
        for field_name in forbidden_non_null_fields:
            if record.get(field_name) is not None:
                issues.append(f"clean_prompt_has_distraction_metadata:{field_name}")
        return issues

    if distraction_type == "irrelevant_prefix":
        if not _is_non_empty_string(noise_block_id):
            issues.append("irrelevant_prefix_missing_noise_block_id")
        if noise_block_id_2 is not None:
            issues.append("irrelevant_prefix_should_not_have_second_noise_block")
        if long_noise_block_id is not None:
            issues.append("irrelevant_prefix_should_not_have_long_noise_block")
        if distraction_subtype != "short_prefix_noise":
            issues.append("irrelevant_prefix_invalid_subtype")

    elif distraction_type == "irrelevant_suffix":
        if not _is_non_empty_string(noise_block_id):
            issues.append("irrelevant_suffix_missing_noise_block_id")
        if noise_block_id_2 is not None:
            issues.append("irrelevant_suffix_should_not_have_second_noise_block")
        if long_noise_block_id is not None:
            issues.append("irrelevant_suffix_should_not_have_long_noise_block")
        if distraction_subtype != "short_suffix_noise":
            issues.append("irrelevant_suffix_invalid_subtype")

    elif distraction_type == "instruction_in_the_middle":
        if not _is_non_empty_string(noise_block_id):
            issues.append("instruction_in_the_middle_missing_first_noise_block")
        if not _is_non_empty_string(noise_block_id_2):
            issues.append("instruction_in_the_middle_missing_second_noise_block")
        if long_noise_block_id is not None:
            issues.append("instruction_in_the_middle_should_not_have_long_noise_block")
        if distraction_subtype != "middle_burial":
            issues.append("instruction_in_the_middle_invalid_subtype")

    elif distraction_type == "conflicting_instruction":
        if not _is_non_empty_string(conflict_variant_id):
            issues.append("conflicting_instruction_missing_variant_id")
        if not _is_non_empty_string(record.get("conflict_subtype")):
            issues.append("conflicting_instruction_missing_subtype")
        if not _is_non_empty_string(distraction_variant_id):
            issues.append("conflicting_instruction_missing_distraction_variant_id")
        if not _is_non_empty_string(distraction_subtype):
            issues.append("conflicting_instruction_missing_distraction_subtype")

    elif distraction_type == "negation_distraction":
        if not _is_non_empty_string(negation_variant_id):
            issues.append("negation_distraction_missing_variant_id")
        if not _is_non_empty_string(record.get("negation_subtype")):
            issues.append("negation_distraction_missing_subtype")
        if not _is_non_empty_string(distraction_variant_id):
            issues.append("negation_distraction_missing_distraction_variant_id")
        if not _is_non_empty_string(distraction_subtype):
            issues.append("negation_distraction_missing_distraction_subtype")

    elif distraction_type == "style_distraction":
        if not _is_non_empty_string(style_id):
            issues.append("style_distraction_missing_style_id")
        if not _is_non_empty_string(record.get("style_family")):
            issues.append("style_distraction_missing_style_family")
        if not _is_non_empty_string(distraction_variant_id):
            issues.append("style_distraction_missing_distraction_variant_id")
        if not _is_non_empty_string(distraction_subtype):
            issues.append("style_distraction_missing_distraction_subtype")

    elif distraction_type == "length_stress":
        if not _is_non_empty_string(long_noise_block_id):
            issues.append("length_stress_missing_long_noise_block_id")
        if noise_block_id is not None:
            issues.append("length_stress_should_not_have_short_noise_block")
        if noise_block_id_2 is not None:
            issues.append("length_stress_should_not_have_second_short_noise_block")
        if not _is_non_empty_string(distraction_variant_id):
            issues.append("length_stress_missing_distraction_variant_id")
        if not _is_non_empty_string(distraction_subtype):
            issues.append("length_stress_missing_distraction_subtype")

    return issues


def _validate_prompt_id_consistency(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    prompt_id = record.get("prompt_id")
    base_example_id = record.get("base_example_id")
    regime = record.get("regime")
    distraction_type = record.get("distraction_type")

    if all(_is_non_empty_string(v) for v in [prompt_id, base_example_id, regime, distraction_type]):
        expected_prompt_id = f"{base_example_id}__{regime}__{distraction_type}"
        if prompt_id != expected_prompt_id:
            issues.append("prompt_id_structure_mismatch")

    return issues


def validate_prompt_record(record: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    issues.extend(_validate_required_fields(record))
    issues.extend(_validate_required_metadata_fields(record))
    issues.extend(_validate_basic_values(record))
    issues.extend(_validate_clean_flag(record))
    issues.extend(_validate_surface_metadata(record))
    issues.extend(_validate_distraction_specific_metadata(record))
    issues.extend(_validate_prompt_id_consistency(record))

    return issues


def validate_prompt_instances(
    prompt_records: List[Dict[str, Any]],
    expected_total: int = 4000,
    expected_base_examples: int = 250,
    expected_per_base_example: int = 16,
) -> Dict[str, Any]:
    record_issues: Dict[str, List[str]] = {}
    issue_counts = Counter()
    dataset_level_issues: List[str] = []

    prompt_ids = set()
    base_example_ids = set()
    counts_by_base = Counter()
    counts_by_regime = Counter()
    counts_by_distraction = Counter()
    counts_by_task = Counter()
    counts_by_subtype = Counter()
    counts_by_surface_type = Counter()
    clean_count = 0
    distracted_count = 0

    for record in prompt_records:
        prompt_id = record.get("prompt_id", "<missing_prompt_id>")
        issues = validate_prompt_record(record)

        if "prompt_id" in record:
            if record["prompt_id"] in prompt_ids:
                issues.append("duplicate_prompt_id")
            else:
                prompt_ids.add(record["prompt_id"])

        if "base_example_id" in record and _is_non_empty_string(record.get("base_example_id")):
            base_example_ids.add(record["base_example_id"])
            counts_by_base[record["base_example_id"]] += 1

        if record.get("task_name") in EXPECTED_TASKS:
            counts_by_task[record["task_name"]] += 1

        if record.get("regime") in EXPECTED_REGIMES:
            counts_by_regime[record["regime"]] += 1

        if record.get("distraction_type") in EXPECTED_DISTRACTION_TYPES:
            counts_by_distraction[record["distraction_type"]] += 1

        if _is_non_empty_string(record.get("distraction_subtype")):
            counts_by_subtype[record["distraction_subtype"]] += 1

        if _is_non_empty_string(record.get("prompt_surface_type")):
            counts_by_surface_type[record["prompt_surface_type"]] += 1

        if record.get("is_clean") is True:
            clean_count += 1
        elif record.get("is_clean") is False:
            distracted_count += 1

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

    expected_half = expected_total // 2
    if counts_by_regime.get("bounded", 0) != expected_half:
        dataset_level_issues.append(
            f"unexpected_bounded_count:{counts_by_regime.get('bounded', 0)}"
        )

    if counts_by_regime.get("unbounded", 0) != expected_half:
        dataset_level_issues.append(
            f"unexpected_unbounded_count:{counts_by_regime.get('unbounded', 0)}"
        )

    expected_clean_count = expected_base_examples * 2
    if clean_count != expected_clean_count:
        dataset_level_issues.append(f"unexpected_clean_count:{clean_count}")

    if distracted_count != expected_total - expected_clean_count:
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

    if counts_by_surface_type.get("bounded_tagged", 0) != counts_by_regime.get("bounded", 0):
        dataset_level_issues.append(
            f"unexpected_bounded_surface_count:{counts_by_surface_type.get('bounded_tagged', 0)}"
        )

    if counts_by_regime.get("unbounded", 0) > 0:
        unbounded_surface_count = sum(
            count for surface_type, count in counts_by_surface_type.items()
            if surface_type != "bounded_tagged"
        )
        if unbounded_surface_count != counts_by_regime.get("unbounded", 0):
            dataset_level_issues.append(
                f"unexpected_unbounded_surface_count:{unbounded_surface_count}"
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
        "counts_by_distraction_subtype": dict(counts_by_subtype),
        "counts_by_prompt_surface_type": dict(counts_by_surface_type),
        "clean_count": clean_count,
        "distracted_count": distracted_count,
    }