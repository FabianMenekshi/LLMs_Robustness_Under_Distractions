from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

TASKS = [
    "single_label_classification",
    "multi_label_classification",
    "information_extraction",
    "rule_based_transformation",
    "extractive_qa",
]
REGIMES = ["bounded", "unbounded"]
DISTRACTION_TYPES = [
    "clean",
    "irrelevant_prefix",
    "irrelevant_suffix",
    "instruction_in_the_middle",
    "conflicting_instruction",
    "negation_distraction",
    "style_distraction",
    "length_stress",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_prompt_ids(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(str(record["prompt_id"]) + "\n")


def build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts_by_task = Counter(r["task_name"] for r in records)
    counts_by_regime = Counter(r["regime"] for r in records)
    counts_by_distraction = Counter(r["distraction_type"] for r in records)
    counts_by_cell = Counter((r["task_name"], r["regime"], r["distraction_type"]) for r in records)

    return {
        "total_records": len(records),
        "counts_by_task": dict(sorted(counts_by_task.items())),
        "counts_by_regime": dict(sorted(counts_by_regime.items())),
        "counts_by_distraction_type": dict(sorted(counts_by_distraction.items())),
        "counts_by_task_regime_distraction": {
            f"{task}__{regime}__{dist}": counts_by_cell[(task, regime, dist)]
            for task in TASKS for regime in REGIMES for dist in DISTRACTION_TYPES
        },
    }


def select_balanced_pilot(records: List[Dict[str, Any]], target_total: int) -> List[Dict[str, Any]]:
    cells: List[Tuple[str, str, str]] = [
        (task, regime, distraction)
        for task in TASKS
        for regime in REGIMES
        for distraction in DISTRACTION_TYPES
    ]
    num_cells = len(cells)
    if target_total < num_cells:
        raise ValueError(f"target_total={target_total} is too small; must be at least {num_cells} to cover all cells once.")

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda r: (r["prompt_id"],)):
        key = (record["task_name"], record["regime"], record["distraction_type"])
        grouped[key].append(record)

    missing_cells = [cell for cell in cells if not grouped.get(cell)]
    if missing_cells:
        raise ValueError(f"Benchmark is missing required cells: {missing_cells}")

    base = target_total // num_cells
    remainder = target_total % num_cells

    selected: List[Dict[str, Any]] = []
    for idx, cell in enumerate(cells):
        need = base + (1 if idx < remainder else 0)
        bucket = grouped[cell]
        if len(bucket) < need:
            raise ValueError(
                f"Not enough records in cell {cell}. Need {need}, found {len(bucket)}."
            )
        selected.extend(bucket[:need])

    return sorted(selected, key=lambda r: (r["task_name"], r["regime"], r["distraction_type"], r["prompt_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced pilot subset from prompt_instances.jsonl")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to prompt_instances.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to pilot subset JSONL")
    parser.add_argument("--prompt-ids-output", type=Path, required=True, help="Path to write pilot prompt_ids.txt")
    parser.add_argument("--summary-output", type=Path, required=True, help="Path to write pilot subset summary JSON")
    parser.add_argument("--target-total", type=int, default=200, help="Total prompts in the pilot subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.benchmark)
    print(f"Loaded benchmark records: {len(records)}")

    selected = select_balanced_pilot(records, target_total=args.target_total)
    print(f"Selected pilot records: {len(selected)}")

    summary = build_summary(selected)
    write_jsonl(selected, args.output)
    write_prompt_ids(selected, args.prompt_ids_output)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Wrote pilot subset: {args.output}")
    print(f"Wrote pilot prompt_ids: {args.prompt_ids_output}")
    print(f"Wrote pilot summary: {args.summary_output}")


if __name__ == "__main__":
    main()
