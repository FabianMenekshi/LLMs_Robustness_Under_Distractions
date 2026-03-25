import json
import random
import re
from dataclasses import asdict
from typing import List, Dict, Any
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

# Reproducibility: generation will be deterministic across runs
random.seed(42)

# Sample one phrase associated with a topic label.
def sample_topic_phrase(label: str) -> str: 
    return random.choice(topic_pool[label])

# Generate candidate examples for single-label classification.
def generate_single_label_candidates(start_id: int = 0) -> List[CandidateExample]:
    candidates = []
    counter = start_id

    for template in single_label_templates:
        for label, value_options in single_label_value_map.items():
            for values in value_options:
                text = template["pattern"].format(
                    product=random.choice(products),
                    adjective=values["adjective"],
                    ending=values["ending"]
                )

                candidate = CandidateExample(
                    example_id=f"slc_{counter:03d}",
                    task_name="single_label_classification",
                    template_name=template["template_name"],
                    instruction=template["instruction"],
                    input_data={"text": text},
                    gold_output={"label": label},
                    metadata={"label_set": SINGLE_LABEL_SET}
                )

                candidates.append(candidate)
                counter += 1

    return candidates

# Generate candidate examples for multi-label classification.
def generate_multi_label_candidates(start_id: int = 0) -> List[CandidateExample]:
    candidates = []
    counter = start_id

    actors = ["government", "committee", "startup", "university", "city council"]

    for template in multi_label_templates:
        for combo in multi_label_combinations:
            topic_item = sample_topic_phrase(combo[0])

            if len(combo) > 1:
                secondary_item = sample_topic_phrase(combo[1])
            else:
                secondary_item = sample_topic_phrase(combo[0])

            text = template["pattern"].format(
                actor=random.choice(actors),
                company=random.choice(companies),
                topic_item=topic_item,
                secondary_item=secondary_item
            )

            candidate = CandidateExample(
                example_id=f"mlc_{counter:03d}",
                task_name="multi_label_classification",
                template_name=template["template_name"],
                instruction=template["instruction"],
                input_data={"text": text},
                gold_output={"labels": sorted(combo)},
                metadata={"label_set": MULTI_LABEL_SET}
            )

            candidates.append(candidate)
            counter += 1

    return candidates

# Generate candidate examples for information extraction.
def generate_ie_candidates(start_id: int = 0) -> List[CandidateExample]:

    candidates = []
    counter = start_id

    for template in ie_templates:
        for person, date, location in zip(people, dates, locations):
            text = template["pattern"].format(
                person=person,
                date=date,
                location=location
            )

            candidate = CandidateExample(
                example_id=f"ie_{counter:03d}",
                task_name="information_extraction",
                template_name=template["template_name"],
                instruction=template["instruction"],
                input_data={"text": text},
                gold_output={
                    "person": person,
                    "date": date,
                    "location": location
                },
                metadata={"schema_keys": IE_SCHEMA_KEYS}
            )

            candidates.append(candidate)
            counter += 1

    return candidates

# Apply a deterministic transformation rule to text.
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

# Generate candidate examples for rule-based transformation.
def generate_transformation_candidates(start_id: int = 0) -> List[CandidateExample]:
    candidates = []
    counter = start_id

    for template in transformation_input_templates:
        for rule_name in RULE_SET:
            for n, n2 in [(3, 12), (7, 25), (10, 42)]:
                text = template["pattern"].format(
                    person=random.choice(people),
                    location=random.choice(locations),
                    n=n,
                    n2=n2
                )

                gold_text = apply_rule(text, rule_name)

                candidate = CandidateExample(
                    example_id=f"rbt_{counter:03d}",
                    task_name="rule_based_transformation",
                    template_name=f"{template['template_name']}__{rule_name}",
                    instruction=rule_instructions[rule_name],
                    input_data={"text": text},
                    gold_output={"text": gold_text},
                    metadata={"rule_name": rule_name}
                )

                candidates.append(candidate)
                counter += 1

    return candidates

# Generate candidate examples for extractive QA.
def generate_qa_candidates(start_id: int = 0) -> List[CandidateExample]:
    candidates = []
    counter = start_id

    for template in qa_templates:
        for person, location, date in zip(people, locations, dates):
            company = random.choice(companies)
            product = random.choice(products)

            passage = template["passage"].format(
                person=person,
                location=location,
                date=date,
                company=company,
                product=product
            )

            question = template["question"].format(
                person=person,
                location=location,
                date=date,
                company=company,
                product=product
            )

            answer_lookup = {
                "person": person,
                "location": location,
                "date": date,
                "company": company,
                "product": product,
            }

            answer = answer_lookup[template["answer_field"]]

            candidate = CandidateExample(
                example_id=f"qa_{counter:03d}",
                task_name="extractive_qa",
                template_name=template["template_name"],
                instruction=template["instruction"],
                input_data={
                    "passage": passage,
                    "question": question
                },
                gold_output={"answer": answer},
                metadata={"answer_field": template["answer_field"]}
            )

            candidates.append(candidate)
            counter += 1

    return candidates

# Generate all candidate examples across all five tasks.
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

# Convert structured input into a compact review string.
def render_input_for_review(example: CandidateExample) -> str:
    if example.task_name == "extractive_qa":
        return (
            f"PASSAGE: {example.input_data['passage']}\n"
            f"QUESTION: {example.input_data['question']}"
        )

    return example.input_data["text"]

# Flatten a candidate example into a review-friendly dictionary.
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

# Count non-overlapping occurrences of a substring in text.
def count_substring_occurrences(text: str, substring: str) -> int:
    return text.count(substring)

# Automatically flag potentially problematic examples. Note that these are warning flags only. They do not automatically reject examples.
def auto_flag_candidate(example: CandidateExample) -> List[str]:

    issues = []
    rendered = render_input_for_review(example)

    if len(rendered) < 20:
        issues.append("input_too_short")

    if len(rendered) > 300:
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

# Find exact duplicate rendered inputs across candidate examples.
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

# Update the manual review status of one example.
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

# Return all examples with a given review status
def get_examples_by_status(
    candidates: List[CandidateExample],
    status: str
) -> List[CandidateExample]:

    return [example for example in candidates if example.review_status == status]

# Select final approved base examples.
def select_final_base_examples(
    candidates: List[CandidateExample],
    n_per_task: int = 50
) -> List[CandidateExample]:

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

        selected.extend(approved[:n_per_task])

    return selected

# Convert a CandidateExample into a JSON-serializable dictionary.
def example_to_dict(example: CandidateExample) -> Dict[str, Any]:
    return asdict(example)

# Save candidate examples to a JSONL file.
def save_candidates_to_jsonl(candidates: List[CandidateExample], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in candidates:
            f.write(json.dumps(example_to_dict(example), ensure_ascii=False) + "\n")