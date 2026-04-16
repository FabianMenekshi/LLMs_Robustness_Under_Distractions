import json
import random
import re
from dataclasses import asdict
from typing import List, Dict, Any, Callable, Tuple
import os

from src.templates import (
    CandidateExample,
    SINGLE_LABEL_SET,
    MULTI_LABEL_SET,
    IE_SCHEMA_KEYS,
    RULE_SET,
    people,
    locations,
    dates,
    products,
    companies,
    other_locations,
    other_people,
    other_companies,
    other_dates,
    single_label_templates,
    single_label_value_map,
    multi_label_templates,
    topic_pool,
    multi_label_combinations,
    ie_templates,
    transformation_input_templates,
    rule_instructions,
    qa_templates,
)

random.seed(42)

TARGET_CANDIDATES_PER_TASK = 50


def _choose_instruction(instruction_pool: List[str], index: int) -> str:
    if not instruction_pool:
        raise ValueError("instruction_pool must not be empty")
    return instruction_pool[index % len(instruction_pool)]


def _get_single_label_instruction(template: Dict[str, Any], index: int) -> str:
    return _choose_instruction(template["instruction_pool"], index)


def _get_multi_label_instruction(template: Dict[str, Any], index: int) -> str:
    return _choose_instruction(template["instruction_pool"], index)


def _get_ie_instruction(template: Dict[str, Any], index: int) -> str:
    return _choose_instruction(template["instruction_pool"], index)


def _get_rule_instruction(rule_name: str, index: int) -> str:
    pool = rule_instructions[rule_name]
    return _choose_instruction(pool, index)


def _get_qa_instruction(template: Dict[str, Any], index: int) -> str:
    return _choose_instruction(template["instruction_pool"], index)


def _reindex_examples(
    examples: List[CandidateExample],
    prefix: str,
    start_id: int,
) -> List[CandidateExample]:
    for idx, example in enumerate(examples):
        example.example_id = f"{prefix}_{start_id + idx:03d}"
    return examples


def _stable_group_sort_key(key: Any) -> str:
    return str(key)


def _round_robin_select(
    items: List[CandidateExample],
    n: int,
    key_fn: Callable[[CandidateExample], Any],
) -> List[CandidateExample]:
    grouped: Dict[Any, List[CandidateExample]] = {}
    for item in items:
        key = key_fn(item)
        grouped.setdefault(key, []).append(item)

    ordered_keys = sorted(grouped.keys(), key=_stable_group_sort_key)
    positions = {key: 0 for key in ordered_keys}

    selected: List[CandidateExample] = []

    while len(selected) < n:
        made_progress = False

        for key in ordered_keys:
            pos = positions[key]
            bucket = grouped[key]

            if pos < len(bucket):
                selected.append(bucket[pos])
                positions[key] += 1
                made_progress = True

                if len(selected) == n:
                    break

        if not made_progress:
            break

    return selected


def _round_robin_fill(
    selected: List[CandidateExample],
    candidates: List[CandidateExample],
    target_n: int,
    key_fn: Callable[[CandidateExample], Any],
) -> List[CandidateExample]:
    selected_ids = {example.example_id for example in selected}
    remaining = [example for example in candidates if example.example_id not in selected_ids]

    needed = max(0, target_n - len(selected))
    if needed == 0:
        return selected

    selected.extend(_round_robin_select(remaining, needed, key_fn=key_fn))
    return selected


def _quota_select(
    items: List[CandidateExample],
    n: int,
    primary_key_fn: Callable[[CandidateExample], Any],
    secondary_key_fn: Callable[[CandidateExample], Any] | None = None,
) -> List[CandidateExample]:
    """
    Stronger balanced selector than plain round robin.

    Stage 1:
        Give each primary group roughly the same quota.
    Stage 2:
        Within each primary group, optionally diversify by secondary groups.
    Stage 3:
        Fill any leftovers by balanced round robin across remaining items.
    """
    if n <= 0 or not items:
        return []

    primary_groups: Dict[Any, List[CandidateExample]] = {}
    for item in items:
        primary_groups.setdefault(primary_key_fn(item), []).append(item)

    ordered_primary = sorted(primary_groups.keys(), key=_stable_group_sort_key)
    num_groups = len(ordered_primary)

    base_quota = n // num_groups
    remainder = n % num_groups

    selected: List[CandidateExample] = []
    selected_ids = set()

    for idx, primary_key in enumerate(ordered_primary):
        group_items = primary_groups[primary_key]
        group_quota = base_quota + (1 if idx < remainder else 0)

        if secondary_key_fn is None:
            chosen = group_items[:group_quota]
        else:
            chosen = _round_robin_select(
                group_items,
                min(group_quota, len(group_items)),
                key_fn=secondary_key_fn,
            )

        for item in chosen:
            if item.example_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.example_id)

    if len(selected) < n:
        remaining = [item for item in items if item.example_id not in selected_ids]
        fill = _round_robin_select(
            remaining,
            n - len(selected),
            key_fn=primary_key_fn,
        )
        for item in fill:
            if item.example_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.example_id)

    return selected[:n]


def _task_balance_key(example: CandidateExample) -> Tuple[Any, ...]:
    if example.task_name == "extractive_qa":
        return (
            example.metadata.get("answer_field"),
            example.template_name,
        )

    return (example.template_name,)


def _qa_generation_secondary_key(example: CandidateExample) -> Tuple[Any, ...]:
    return (
        example.template_name,
        example.instruction,
    )


def _default_generation_balance_key(example: CandidateExample) -> Tuple[Any, ...]:
    return (
        example.template_name,
        example.instruction,
    )


def generate_single_label_candidates(start_id: int = 0) -> List[CandidateExample]:
    pool: List[CandidateExample] = []
    counter = 0

    for template_idx, template in enumerate(single_label_templates):
        uses_product = "{product}" in template["pattern"]

        for label_idx, (label, value_options) in enumerate(single_label_value_map.items()):
            for value_idx, values in enumerate(value_options):
                if uses_product:
                    for product_idx, product in enumerate(products):
                        text = template["pattern"].format(
                            product=product,
                            descriptor_1=values["descriptor_1"],
                            descriptor_2=values["descriptor_2"],
                            descriptor_3=values["descriptor_3"],
                        )

                        instruction = _get_single_label_instruction(
                            template,
                            template_idx + label_idx + value_idx + product_idx,
                        )

                        pool.append(
                            CandidateExample(
                                example_id=f"slc_tmp_{counter:04d}",
                                task_name="single_label_classification",
                                template_name=template["template_name"],
                                instruction=instruction,
                                input_data={"text": text},
                                gold_output={"label": label},
                                metadata={"label_set": SINGLE_LABEL_SET},
                            )
                        )
                        counter += 1
                else:
                    text = template["pattern"].format(
                        descriptor_1=values["descriptor_1"],
                        descriptor_2=values["descriptor_2"],
                        descriptor_3=values["descriptor_3"],
                    )

                    instruction = _get_single_label_instruction(
                        template,
                        template_idx + label_idx + value_idx,
                    )

                    pool.append(
                        CandidateExample(
                            example_id=f"slc_tmp_{counter:04d}",
                            task_name="single_label_classification",
                            template_name=template["template_name"],
                            instruction=instruction,
                            input_data={"text": text},
                            gold_output={"label": label},
                            metadata={"label_set": SINGLE_LABEL_SET},
                        )
                    )
                    counter += 1

    selected = _quota_select(
        pool,
        TARGET_CANDIDATES_PER_TASK,
        primary_key_fn=lambda ex: ex.template_name,
        secondary_key_fn=lambda ex: (ex.gold_output["label"], ex.instruction),
    )

    return _reindex_examples(selected, "slc", start_id)


def generate_multi_label_candidates(start_id: int = 0) -> List[CandidateExample]:
    """
    Build a large multi-label pool, then explicitly balance selection by template first.
    This is the direct fix for the current collapse where the final selected set ends up
    using only `company_news` despite multiple templates existing in the inventory.
    """
    pool: List[CandidateExample] = []
    counter = 0
    seen_texts = set()

    actors = [
        "government",
        "committee",
        "startup",
        "university",
        "city council",
        "public agency",
        "industry group",
        "research institute",
    ]

    for actor_idx, actor in enumerate(actors):
        for template_idx, template in enumerate(multi_label_templates):
            for combo_idx, combo in enumerate(multi_label_combinations):
                topic_item = topic_pool[combo[0]][(combo_idx + template_idx) % len(topic_pool[combo[0]])]
                secondary_label = combo[1] if len(combo) > 1 else combo[0]
                secondary_item = topic_pool[secondary_label][
                    (combo_idx + actor_idx + template_idx) % len(topic_pool[secondary_label])
                ]
                company = companies[(combo_idx + template_idx + actor_idx) % len(companies)]

                text = template["pattern"].format(
                    actor=actor,
                    company=company,
                    topic_item=topic_item,
                    secondary_item=secondary_item,
                )

                if text in seen_texts:
                    continue

                seen_texts.add(text)

                instruction = _get_multi_label_instruction(
                    template,
                    actor_idx + template_idx + combo_idx,
                )

                pool.append(
                    CandidateExample(
                        example_id=f"mlc_tmp_{counter:04d}",
                        task_name="multi_label_classification",
                        template_name=template["template_name"],
                        instruction=instruction,
                        input_data={"text": text},
                        gold_output={"labels": sorted(combo)},
                        metadata={"label_set": MULTI_LABEL_SET},
                    )
                )
                counter += 1

    selected = _quota_select(
        pool,
        TARGET_CANDIDATES_PER_TASK,
        primary_key_fn=lambda ex: ex.template_name,
        secondary_key_fn=lambda ex: (tuple(ex.gold_output["labels"]), ex.instruction),
    )

    return _reindex_examples(selected, "mlc", start_id)


def generate_ie_candidates(start_id: int = 0) -> List[CandidateExample]:
    pool: List[CandidateExample] = []
    counter = 0
    seen_texts = set()

    for template_idx, template in enumerate(ie_templates):
        for person_idx, person in enumerate(people):
            for date_idx, date in enumerate(dates):
                location = locations[(person_idx + date_idx + template_idx) % len(locations)]
                other_date = other_dates[(person_idx + template_idx + date_idx) % len(other_dates)]
                other_location = other_locations[(template_idx + date_idx + person_idx) % len(other_locations)]

                text = template["pattern"].format(
                    person=person,
                    date=date,
                    location=location,
                    other_date=other_date,
                    other_location=other_location,
                )

                if text in seen_texts:
                    continue

                seen_texts.add(text)

                instruction = _get_ie_instruction(
                    template,
                    template_idx + person_idx + date_idx,
                )

                pool.append(
                    CandidateExample(
                        example_id=f"ie_tmp_{counter:04d}",
                        task_name="information_extraction",
                        template_name=template["template_name"],
                        instruction=instruction,
                        input_data={"text": text},
                        gold_output={
                            "person": person,
                            "date": date,
                            "location": location,
                        },
                        metadata={"schema_keys": IE_SCHEMA_KEYS},
                    )
                )
                counter += 1

    selected = _quota_select(
        pool,
        TARGET_CANDIDATES_PER_TASK,
        primary_key_fn=lambda ex: ex.template_name,
        secondary_key_fn=_default_generation_balance_key,
    )

    return _reindex_examples(selected, "ie", start_id)


def apply_rule(text: str, rule_name: str) -> str:
    if rule_name == "lowercase":
        return text.lower()

    if rule_name == "remove_punctuation":
        return re.sub(r"[^\w\s]", "", text)

    if rule_name == "replace_numbers_with_NUM":
        return re.sub(r"\d+", "<NUM>", text)

    if rule_name == "remove_words_longer_than_6":
        words = text.split()
        kept_words = [
            word for word in words
            if len(re.sub(r"[^\w]", "", word)) <= 6
        ]
        return " ".join(kept_words)

    raise ValueError(f"Unknown rule: {rule_name}")


def generate_transformation_candidates(start_id: int = 0) -> List[CandidateExample]:
    pool: List[CandidateExample] = []
    counter = 0
    seen_inputs = set()

    number_pairs = [
        (3, 12), (7, 25), (10, 42), (15, 99), (21, 5),
        (8, 14), (11, 37), (19, 44), (23, 61), (30, 2),
    ]

    rule_cycle = [
        "lowercase",
        "remove_punctuation",
        "replace_numbers_with_NUM",
        "remove_words_longer_than_6",
    ]

    for template_idx, template in enumerate(transformation_input_templates):
        for pair_idx, (n, n2) in enumerate(number_pairs):
            for person_idx, person in enumerate(people):
                location = locations[(template_idx + pair_idx + person_idx) % len(locations)]
                rule_name = rule_cycle[(template_idx + pair_idx + person_idx) % len(rule_cycle)]

                text = template["pattern"].format(
                    person=person,
                    location=location,
                    n=n,
                    n2=n2,
                )

                if text in seen_inputs:
                    continue

                seen_inputs.add(text)
                gold_text = apply_rule(text, rule_name)

                instruction = _get_rule_instruction(
                    rule_name,
                    template_idx + pair_idx + person_idx,
                )

                pool.append(
                    CandidateExample(
                        example_id=f"rbt_tmp_{counter:04d}",
                        task_name="rule_based_transformation",
                        template_name=f"{template['template_name']}__{rule_name}",
                        instruction=instruction,
                        input_data={"text": text},
                        gold_output={"text": gold_text},
                        metadata={"rule_name": rule_name},
                    )
                )
                counter += 1

    selected = _quota_select(
        pool,
        TARGET_CANDIDATES_PER_TASK,
        primary_key_fn=lambda ex: ex.template_name,
        secondary_key_fn=lambda ex: (ex.metadata["rule_name"], ex.instruction),
    )

    return _reindex_examples(selected, "rbt", start_id)


def _build_qa_context(
    template: Dict[str, Any],
    template_idx: int,
    person_idx: int,
    date_idx: int,
) -> Dict[str, str]:
    person = people[person_idx % len(people)]
    location = locations[(person_idx + date_idx + template_idx) % len(locations)]
    date = dates[date_idx % len(dates)]
    company = companies[(person_idx + template_idx) % len(companies)]
    product = products[(date_idx + template_idx) % len(products)]

    other_location = other_locations[(template_idx + person_idx + date_idx) % len(other_locations)]
    other_person = other_people[(template_idx + person_idx + date_idx) % len(other_people)]
    other_company = other_companies[(template_idx + person_idx + date_idx) % len(other_companies)]
    other_date = other_dates[(template_idx + person_idx + date_idx) % len(other_dates)]

    context = {
        "person": person,
        "location": location,
        "date": date,
        "company": company,
        "product": product,
        "other_location": other_location,
        "other_person": other_person,
        "other_company": other_company,
        "other_date": other_date,
    }

    if context["other_location"] == context["location"]:
        context["other_location"] = other_locations[(template_idx + person_idx + date_idx + 1) % len(other_locations)]

    if context["other_person"] == context["person"]:
        context["other_person"] = other_people[(template_idx + person_idx + date_idx + 1) % len(other_people)]

    if context["other_company"] == context["company"]:
        context["other_company"] = other_companies[(template_idx + person_idx + date_idx + 1) % len(other_companies)]

    if context["other_date"] == context["date"]:
        context["other_date"] = other_dates[(template_idx + person_idx + date_idx + 1) % len(other_dates)]

    return context


def generate_qa_candidates(start_id: int = 0) -> List[CandidateExample]:
    """
    Build a richer QA pool and then explicitly balance by answer field first, and by
    template/instruction within each answer field second.

    This is the direct fix for the current drift where the final QA set becomes
    over-concentrated in a single answer field even after template edits.
    """
    pool: List[CandidateExample] = []
    counter = 0
    seen_inputs = set()

    for template_idx, template in enumerate(qa_templates):
        for person_idx, _person in enumerate(people):
            for date_idx, _date in enumerate(dates):
                context = _build_qa_context(
                    template=template,
                    template_idx=template_idx,
                    person_idx=person_idx,
                    date_idx=date_idx,
                )

                passage = template["passage"].format(**context)
                question = template["question"].format(**context)

                rendered = f"PASSAGE: {passage}\nQUESTION: {question}"
                if rendered in seen_inputs:
                    continue

                seen_inputs.add(rendered)

                answer_lookup = {
                    "person": context["person"],
                    "location": context["location"],
                    "date": context["date"],
                    "company": context["company"],
                    "product": context["product"],
                }

                answer_field = template["answer_field"]
                answer = answer_lookup[answer_field]

                if passage.count(answer) != 1:
                    continue

                instruction = _get_qa_instruction(
                    template,
                    template_idx + person_idx + date_idx,
                )

                pool.append(
                    CandidateExample(
                        example_id=f"qa_tmp_{counter:04d}",
                        task_name="extractive_qa",
                        template_name=template["template_name"],
                        instruction=instruction,
                        input_data={
                            "passage": passage,
                            "question": question,
                        },
                        gold_output={"answer": answer},
                        metadata={"answer_field": answer_field},
                    )
                )
                counter += 1

    selected = _quota_select(
        pool,
        TARGET_CANDIDATES_PER_TASK,
        primary_key_fn=lambda ex: ex.metadata.get("answer_field"),
        secondary_key_fn=_qa_generation_secondary_key,
    )

    return _reindex_examples(selected, "qa", start_id)


def generate_all_candidates() -> List[CandidateExample]:
    all_candidates = []

    single_label = generate_single_label_candidates(start_id=0)
    all_candidates.extend(single_label)

    multi_label = generate_multi_label_candidates(start_id=len(all_candidates))
    all_candidates.extend(multi_label)

    ie_examples = generate_ie_candidates(start_id=len(all_candidates))
    all_candidates.extend(ie_examples)

    transformation_examples = generate_transformation_candidates(start_id=len(all_candidates))
    all_candidates.extend(transformation_examples)

    qa_examples = generate_qa_candidates(start_id=len(all_candidates))
    all_candidates.extend(qa_examples)

    return all_candidates


def render_input_for_review(example: CandidateExample) -> str:
    if example.task_name == "extractive_qa":
        return (
            f"PASSAGE: {example.input_data['passage']}\n"
            f"QUESTION: {example.input_data['question']}"
        )

    return example.input_data["text"]


def candidate_to_review_row(example: CandidateExample) -> Dict[str, Any]:
    return {
        "example_id": example.example_id,
        "task_name": example.task_name,
        "template_name": example.template_name,
        "instruction": example.instruction,
        "rendered_input": render_input_for_review(example),
        "gold_output": json.dumps(example.gold_output, ensure_ascii=False),
        "review_status": example.review_status,
        "review_note": example.review_note,
    }


def count_substring_occurrences(text: str, substring: str) -> int:
    return text.count(substring)


def auto_flag_candidate(example: CandidateExample) -> List[str]:
    issues = []
    rendered = render_input_for_review(example)

    if len(rendered) < 20:
        issues.append("input_too_short")

    if len(rendered) > 600:
        issues.append("input_too_long")

    if example.task_name == "extractive_qa":
        passage = example.input_data["passage"]
        answer = example.gold_output["answer"]
        if count_substring_occurrences(passage, answer) != 1:
            issues.append("qa_answer_not_unique_in_passage")

    if example.task_name == "multi_label_classification":
        labels = example.gold_output["labels"]
        if len(labels) != len(set(labels)):
            issues.append("duplicate_labels")

    return issues


def find_exact_duplicate_inputs(candidates: List[CandidateExample]) -> Dict[str, List[str]]:
    seen: Dict[str, List[str]] = {}

    for example in candidates:
        rendered = render_input_for_review(example)
        seen.setdefault(rendered, []).append(example.example_id)

    duplicates = {
        rendered: ids
        for rendered, ids in seen.items()
        if len(ids) > 1
    }

    return duplicates


def set_review_status(
    candidates: List[CandidateExample],
    example_id: str,
    status: str,
    note: str = ""
) -> None:
    valid_statuses = {"pending", "approved", "rejected"}
    if status not in valid_statuses:
        raise ValueError(f"Invalid status: {status}")

    for example in candidates:
        if example.example_id == example_id:
            example.review_status = status
            example.review_note = note
            return

    raise ValueError(f"Example id not found: {example_id}")


def get_examples_by_status(
    candidates: List[CandidateExample],
    status: str
) -> List[CandidateExample]:
    return [example for example in candidates if example.review_status == status]


def select_final_base_examples(
    candidates: List[CandidateExample],
    n_per_task: int = 50
) -> List[CandidateExample]:
    """
    Select approved examples only.

    For QA:
        balance by answer field first, then template.
    For other tasks:
        balance by template.
    """
    selected = []
    task_names = sorted({example.task_name for example in candidates})

    for task_name in task_names:
        approved = [
            example
            for example in candidates
            if example.task_name == task_name and example.review_status == "approved"
        ]

        if len(approved) < n_per_task:
            print(f"Warning: task '{task_name}' has only {len(approved)} approved examples.")

        if task_name == "extractive_qa":
            chosen = _quota_select(
                approved,
                min(n_per_task, len(approved)),
                primary_key_fn=lambda ex: ex.metadata.get("answer_field"),
                secondary_key_fn=lambda ex: ex.template_name,
            )
        else:
            chosen = _quota_select(
                approved,
                min(n_per_task, len(approved)),
                primary_key_fn=lambda ex: ex.template_name,
                secondary_key_fn=lambda ex: ex.instruction,
            )

        selected.extend(chosen)

    return selected


def select_base_examples_exact(
    candidates: List[CandidateExample],
    n_per_task: int = 50
) -> List[CandidateExample]:
    """
    Select exactly n_per_task examples for each task.

    Selection priority:
    1. approved
    2. pending
    3. rejected

    For QA:
        balance by answer field first, then template.
    For multi-label:
        enforce template balancing directly.
    For others:
        balance by template.
    """
    selected = []
    task_names = sorted({example.task_name for example in candidates})

    for task_name in task_names:
        approved = [
            example
            for example in candidates
            if example.task_name == task_name and example.review_status == "approved"
        ]

        pending = [
            example
            for example in candidates
            if example.task_name == task_name and example.review_status == "pending"
        ]

        rejected = [
            example
            for example in candidates
            if example.task_name == task_name and example.review_status == "rejected"
        ]

        def _task_specific_fill(
            chosen: List[CandidateExample],
            pool: List[CandidateExample],
            target_n: int,
        ) -> List[CandidateExample]:
            chosen_ids = {example.example_id for example in chosen}
            remaining = [example for example in pool if example.example_id not in chosen_ids]
            needed = target_n - len(chosen)
            if needed <= 0 or not remaining:
                return chosen

            if task_name == "extractive_qa":
                fill = _quota_select(
                    remaining,
                    needed,
                    primary_key_fn=lambda ex: ex.metadata.get("answer_field"),
                    secondary_key_fn=lambda ex: ex.template_name,
                )
            else:
                fill = _quota_select(
                    remaining,
                    needed,
                    primary_key_fn=lambda ex: ex.template_name,
                    secondary_key_fn=lambda ex: ex.instruction,
                )

            chosen.extend(fill)
            return chosen

        chosen: List[CandidateExample] = []
        chosen = _task_specific_fill(chosen, approved, n_per_task)
        chosen = _task_specific_fill(chosen, pending, n_per_task)

        used_rejected = False
        before_rejected = len(chosen)
        chosen = _task_specific_fill(chosen, rejected, n_per_task)
        if len(chosen) > before_rejected:
            used_rejected = True

        if used_rejected:
            print(
                f"WARNING: task '{task_name}' needed rejected examples as fallback "
                f"(approved={len(approved)}, pending={len(pending)}, rejected={len(rejected)})."
            )

        if len(chosen) != n_per_task:
            raise ValueError(
                f"Task '{task_name}' has only {len(chosen)} total available examples "
                f"(approved={len(approved)}, pending={len(pending)}, rejected={len(rejected)})."
            )

        selected.extend(chosen)

    if len(selected) != n_per_task * len(task_names):
        raise ValueError(
            f"Expected {n_per_task * len(task_names)} total selected examples, got {len(selected)}."
        )

    return selected


def example_to_dict(example: CandidateExample) -> Dict[str, Any]:
    return asdict(example)


def save_candidates_to_jsonl(candidates: List[CandidateExample], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in candidates:
            f.write(json.dumps(example_to_dict(example), ensure_ascii=False) + "\n")


def load_candidates_from_jsonl(input_path: str) -> List[CandidateExample]:
    candidates = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            candidates.append(
                CandidateExample(
                    example_id=row["example_id"],
                    task_name=row["task_name"],
                    template_name=row["template_name"],
                    instruction=row["instruction"],
                    input_data=row["input_data"],
                    gold_output=row["gold_output"],
                    metadata=row["metadata"],
                    review_status=row.get("review_status", "pending"),
                    review_note=row.get("review_note", ""),
                )
            )

    return candidates


def count_examples_by_task(candidates: List[CandidateExample]) -> Dict[str, int]:
    counts = {}
    for example in candidates:
        counts[example.task_name] = counts.get(example.task_name, 0) + 1
    return counts


def count_examples_by_status(candidates: List[CandidateExample]) -> Dict[str, int]:
    counts = {}
    for example in candidates:
        counts[example.review_status] = counts.get(example.review_status, 0) + 1
    return counts


def count_by_task_and_status(candidates: List[CandidateExample]) -> Dict[str, Dict[str, int]]:
    summary = {}
    for example in candidates:
        task = example.task_name
        status = example.review_status
        if task not in summary:
            summary[task] = {}
        summary[task][status] = summary[task].get(status, 0) + 1
    return summary