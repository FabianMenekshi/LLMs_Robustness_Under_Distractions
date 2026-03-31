import json
import os
from collections import Counter
from typing import List, Dict, Any

from src.templates import CandidateExample

# Convert a CandidateExample into the final clean base-dataset record format.

def candidate_to_base_record(example: CandidateExample) -> Dict[str, Any]:
    record = {
        "example_id": example.example_id,
        "task_name": example.task_name,
        "template_name": example.template_name,
        "instruction": example.instruction,
        "input_data": example.input_data,
        "gold_output": example.gold_output,
        "metadata": example.metadata,
    }

    if not isinstance(record["input_data"], dict):
        raise TypeError(f"input_data must be a dict for {record['example_id']}")
    if not isinstance(record["gold_output"], dict):
        raise TypeError(f"gold_output must be a dict for {record['example_id']}")
    if not isinstance(record["metadata"], dict):
        raise TypeError(f"metadata must be a dict for {record['example_id']}")

    return record

# Convert a list of CandidateExample objects into final dataset records.
def build_base_dataset(candidates: List[CandidateExample]) -> List[Dict[str, Any]]:
    return [candidate_to_base_record(example) for example in candidates]

# Count how many records belong to each task.
def count_records_by_task(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(record["task_name"] for record in records)
    return dict(counter)

# Collect unique instructions used for each task.
def collect_instructions_by_task(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    by_task: Dict[str, set] = {}

    for record in records:
        task_name = record["task_name"]
        instruction = record["instruction"]
        by_task.setdefault(task_name, set()).add(instruction)

    return {task: sorted(list(instructions)) for task, instructions in by_task.items()}

# Build a compact summary of the clean base dataset
def build_dataset_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_records": len(records),
        "counts_by_task": count_records_by_task(records),
        "instructions_by_task": collect_instructions_by_task(records),
    }

# Save records as JSONL.
def save_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

# Save any JSON-serializable object as pretty JSON.
def save_json(data: Any, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)

# Load JSONL records from disk.
def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records