'''
This is the file that takes the selected CandidateExample objects 
and converts them into the final exported base dataset format.

It contains:
    - conversion from CandidateExample to plain dictionary records
    - construction of the base dataset list
    - summary-building for auditing
    - JSON / JSONL saving and loading utilities
'''

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any

from src.generation import CandidateExample


def candidate_example_to_base_record(example: CandidateExample) -> Dict[str, Any]:
    return {
        "example_id": example.example_id,
        "task_name": example.task_name,
        "template_name": example.template_name,
        "instruction": example.instruction,
        "input_data": example.input_data,
        "gold_output": example.gold_output,
        "metadata": example.metadata,
    }


def build_base_dataset(selected_examples: List[CandidateExample]) -> List[Dict[str, Any]]:
    return [candidate_example_to_base_record(example) for example in selected_examples]


def _sorted_counter(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _sorted_nested_counter(nested: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    return {
        outer_key: _sorted_counter(counter)
        for outer_key, counter in sorted(nested.items(), key=lambda item: item[0])
    }


def build_dataset_summary(base_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a richer dataset summary for auditing the final selected base dataset.

    In addition to basic counts, this summary now includes:
    - counts by template
    - counts by task and template
    - counts by instruction
    - counts by task and instruction
    - QA answer-field counts
    - QA answer-field counts by template
    """
    counts_by_task = Counter()
    counts_by_template = Counter()
    counts_by_instruction = Counter()

    instructions_by_task = defaultdict(set)
    templates_by_task = defaultdict(set)

    counts_by_task_and_template: Dict[str, Counter] = defaultdict(Counter)
    counts_by_task_and_instruction: Dict[str, Counter] = defaultdict(Counter)

    qa_answer_field_counts = Counter()
    qa_answer_field_by_template: Dict[str, Counter] = defaultdict(Counter)
    qa_answer_field_by_instruction: Dict[str, Counter] = defaultdict(Counter)

    for record in base_records:
        task_name = record["task_name"]
        template_name = record["template_name"]
        instruction = record["instruction"]

        counts_by_task[task_name] += 1
        counts_by_template[template_name] += 1
        counts_by_instruction[instruction] += 1

        instructions_by_task[task_name].add(instruction)
        templates_by_task[task_name].add(template_name)

        counts_by_task_and_template[task_name][template_name] += 1
        counts_by_task_and_instruction[task_name][instruction] += 1

        if task_name == "extractive_qa":
            answer_field = record.get("metadata", {}).get("answer_field", "<missing>")
            qa_answer_field_counts[answer_field] += 1
            qa_answer_field_by_template[template_name][answer_field] += 1
            qa_answer_field_by_instruction[instruction][answer_field] += 1

    summary = {
        "total_records": len(base_records),
        "counts_by_task": _sorted_counter(counts_by_task),
        "counts_by_template": _sorted_counter(counts_by_template),
        "counts_by_instruction": _sorted_counter(counts_by_instruction),
        "instructions_by_task": {
            task_name: sorted(instructions)
            for task_name, instructions in sorted(instructions_by_task.items(), key=lambda item: item[0])
        },
        "templates_by_task": {
            task_name: sorted(template_names)
            for task_name, template_names in sorted(templates_by_task.items(), key=lambda item: item[0])
        },
        "counts_by_task_and_template": _sorted_nested_counter(counts_by_task_and_template),
        "counts_by_task_and_instruction": _sorted_nested_counter(counts_by_task_and_instruction),
        "instruction_diversity_by_task": {
            task_name: len(instructions)
            for task_name, instructions in sorted(instructions_by_task.items(), key=lambda item: item[0])
        },
        "template_diversity_by_task": {
            task_name: len(template_names)
            for task_name, template_names in sorted(templates_by_task.items(), key=lambda item: item[0])
        },
        "qa_summary": {
            "answer_field_counts": _sorted_counter(qa_answer_field_counts),
            "answer_field_counts_by_template": _sorted_nested_counter(qa_answer_field_by_template),
            "answer_field_counts_by_instruction": _sorted_nested_counter(qa_answer_field_by_instruction),
        },
    }

    return summary


def save_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(data: Any, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records