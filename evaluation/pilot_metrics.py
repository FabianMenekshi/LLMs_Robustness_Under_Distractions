from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def add_metric_rows(rows: List[Dict[str, Any]], level: str, group_value: str, records: List[Dict[str, Any]]) -> None:
    total = len(records)
    correct = sum(1 for r in records if bool(r.get("is_correct")))
    parse_failures = sum(1 for r in records if not bool(r.get("parse_success")))

    error_counts = Counter(
        (r.get("parse_error_code") or r.get("score_error_code") or "none")
        for r in records
    )
    top_error, top_error_count = (error_counts.most_common(1)[0] if error_counts else ("none", 0))

    rows.append({
        "level": level,
        "group": group_value,
        "num_examples": total,
        "num_correct": correct,
        "accuracy": f"{safe_rate(correct, total):.6f}",
        "num_parse_failures": parse_failures,
        "parse_failure_rate": f"{safe_rate(parse_failures, total):.6f}",
        "top_error_code": top_error,
        "top_error_count": top_error_count,
    })


def build_metrics_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    add_metric_rows(rows, "overall", "all", records)

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_regime: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_distraction: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_task_regime: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_task_distraction: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in records:
        task = str(r.get("task_name"))
        regime = str(r.get("regime"))
        distraction = str(r.get("distraction_type"))
        by_task[task].append(r)
        by_regime[regime].append(r)
        by_distraction[distraction].append(r)
        by_task_regime[f"{task}__{regime}"].append(r)
        by_task_distraction[f"{task}__{distraction}"].append(r)

    for key, group in sorted(by_task.items()):
        add_metric_rows(rows, "task", key, group)
    for key, group in sorted(by_regime.items()):
        add_metric_rows(rows, "regime", key, group)
    for key, group in sorted(by_distraction.items()):
        add_metric_rows(rows, "distraction_type", key, group)
    for key, group in sorted(by_task_regime.items()):
        add_metric_rows(rows, "task__regime", key, group)
    for key, group in sorted(by_task_distraction.items()):
        add_metric_rows(rows, "task__distraction_type", key, group)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pilot scored results into a CSV metrics table")
    parser.add_argument("--scored-results", type=Path, required=True, help="Path to scored JSONL from evaluate_model.py")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to pilot_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.scored_results)
    print(f"Loaded scored records: {len(records)}")
    rows = build_metrics_rows(records)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "group",
                "num_examples",
                "num_correct",
                "accuracy",
                "num_parse_failures",
                "parse_failure_rate",
                "top_error_code",
                "top_error_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote pilot metrics CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
