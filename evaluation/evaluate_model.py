from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from parsing import parse_prediction
from scoring import score_prediction
from run_inference import (
    append_jsonl,
    generate_one,
    load_json,
    load_jsonl,
    load_model_and_tokenizer,
    read_completed_prompt_ids,
    resolve_model_settings,
    select_records,
    set_determinism,
    validate_benchmark_record,
)


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def build_result_row(
    benchmark_record: Dict[str, Any],
    model_name: str,
    raw_output: str,
) -> Dict[str, Any]:
    task_name = benchmark_record["task_name"]
    gold_output = benchmark_record["gold_output"]

    parse_result = parse_prediction(
        raw_output=raw_output,
        task_name=task_name,
        instance_record=benchmark_record,
    )
    score_result = score_prediction(
        parsed_result=parse_result,
        gold_output=gold_output,
        task_name=task_name,
    )

    row: Dict[str, Any] = {
        "prompt_id": benchmark_record.get("prompt_id"),
        "base_example_id": benchmark_record.get("base_example_id"),
        "model_name": model_name,
        "task_name": benchmark_record.get("task_name"),
        "regime": benchmark_record.get("regime"),
        "distraction_type": benchmark_record.get("distraction_type"),
        "distraction_subtype": benchmark_record.get("distraction_subtype"),
        "is_clean": benchmark_record.get("is_clean"),
        "prompt_surface_type": benchmark_record.get("prompt_surface_type"),
        "surface_id": benchmark_record.get("surface_id"),
        "surface_name": benchmark_record.get("surface_name"),
        "opener_id": benchmark_record.get("opener_id"),
        "layout_id": benchmark_record.get("layout_id"),
        "layout_name": benchmark_record.get("layout_name"),
        "placement": benchmark_record.get("placement"),
        "source_instruction": benchmark_record.get("source_instruction"),
        "source_template_name": benchmark_record.get("source_template_name"),
        "raw_output": raw_output,
        "parsed_output": parse_result.parsed_output,
        "parse_success": parse_result.success,
        "parse_status": "success" if parse_result.success else "failure",
        "parse_error_code": parse_result.error_code,
        "parse_error_detail": parse_result.error_detail,
        "score_success": score_result.success,
        "is_correct": score_result.is_correct,
        "score": score_result.score,
        "score_error_code": score_result.error_code,
        "score_error_detail": score_result.error_detail,
        "gold_output": gold_output,
    }
    return row


def build_result_row_from_raw_record(raw_record: Dict[str, Any], benchmark_record: Dict[str, Any]) -> Dict[str, Any]:
    raw_output = str(raw_record.get("raw_output", ""))
    model_name = str(
        raw_record.get("model_name")
        or raw_record.get("model_key")
        or raw_record.get("model_hf_id")
        or "unknown_model"
    )
    row = build_result_row(benchmark_record=benchmark_record, model_name=model_name, raw_output=raw_output)

    for key in [
        "model_key",
        "model_hf_id",
        "input_token_count",
        "output_token_count",
        "latency_s",
        "max_new_tokens",
        "use_chat_template",
        "dtype",
        "inference_error",
    ]:
        if key in raw_record:
            row[key] = raw_record[key]
    return row


def evaluate_from_raw_predictions(
    benchmark_records: List[Dict[str, Any]],
    raw_predictions: List[Dict[str, Any]],
    output_path: Path,
    overwrite: bool,
    log_every: int,
) -> None:
    benchmark_by_prompt_id = {record["prompt_id"]: record for record in benchmark_records}

    if overwrite and output_path.exists():
        output_path.unlink()

    completed_prompt_ids = read_completed_prompt_ids(output_path)
    processed = 0

    for raw_record in raw_predictions:
        prompt_id = raw_record.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id not in benchmark_by_prompt_id:
            raise KeyError(f"Raw prediction has unknown or missing prompt_id: {prompt_id!r}")
        if prompt_id in completed_prompt_ids:
            continue

        result_row = build_result_row_from_raw_record(
            raw_record=raw_record,
            benchmark_record=benchmark_by_prompt_id[prompt_id],
        )
        append_jsonl(output_path, [result_row])
        completed_prompt_ids.add(prompt_id)
        processed += 1

        if processed % log_every == 0:
            print(f"Scored records written: {processed}")

    print(f"Finished scoring raw predictions. Newly written records: {processed}")


def evaluate_with_live_inference(
    benchmark_records: List[Dict[str, Any]],
    model_config_path: Path,
    model_key: str,
    output_path: Path,
    overwrite: bool,
    log_every: int,
) -> None:
    model_config = load_json(model_config_path)
    settings = resolve_model_settings(model_config, model_key)

    if overwrite and output_path.exists():
        output_path.unlink()

    completed_prompt_ids = read_completed_prompt_ids(output_path)
    pending_records = [record for record in benchmark_records if record["prompt_id"] not in completed_prompt_ids]

    print(f"Already completed: {len(completed_prompt_ids)}")
    print(f"Pending records: {len(pending_records)}")
    print(f"Loading model: {settings.model_key} ({settings.hf_id})")

    model, tokenizer = load_model_and_tokenizer(settings)
    model_name = settings.model_key

    processed = 0
    for index, record in enumerate(pending_records, start=1):
        prompt_id = record["prompt_id"]
        task_name = record["task_name"]
        max_new_tokens = settings.max_new_tokens_by_task.get(task_name, settings.max_new_tokens_default)
        print(f"Generating prompt {index}/{len(pending_records)}: {prompt_id}")

        try:
            generation = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_text=record["prompt_text"],
                use_chat_template=settings.use_chat_template,
                max_new_tokens=max_new_tokens,
            )
            result_row = build_result_row(
                benchmark_record=record,
                model_name=model_name,
                raw_output=str(generation["raw_output"]),
            )
            result_row.update(
                {
                    "model_key": settings.model_key,
                    "model_hf_id": settings.hf_id,
                    "input_token_count": generation.get("input_token_count"),
                    "output_token_count": generation.get("output_token_count"),
                    "latency_s": generation.get("latency_s"),
                    "max_new_tokens": max_new_tokens,
                    "use_chat_template": settings.use_chat_template,
                    "dtype": settings.dtype,
                    "inference_error": None,
                }
            )
        except Exception as exc:
            result_row = build_result_row(
                benchmark_record=record,
                model_name=model_name,
                raw_output="",
            )
            result_row.update(
                {
                    "model_key": settings.model_key,
                    "model_hf_id": settings.hf_id,
                    "input_token_count": None,
                    "output_token_count": None,
                    "latency_s": None,
                    "max_new_tokens": max_new_tokens,
                    "use_chat_template": settings.use_chat_template,
                    "dtype": settings.dtype,
                    "inference_error": repr(exc),
                }
            )

        append_jsonl(output_path, [result_row])
        processed += 1

        if processed % log_every == 0:
            print(f"Evaluated records written: {processed}")

    print(f"Finished end-to-end evaluation. Newly written records: {processed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end evaluation: inference + parsing + scoring.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to prompt_instances.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to scored results JSONL")
    parser.add_argument("--model-config", type=Path, help="Path to model_config.json (required for live inference mode)")
    parser.add_argument("--model-key", type=str, help="Model key from model_config.json (required for live inference mode)")
    parser.add_argument("--raw-predictions", type=Path, help="Optional path to raw predictions JSONL from run_inference.py")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of benchmark records")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into the benchmark records")
    parser.add_argument("--prompt-ids", type=Path, default=None, help="Optional file containing prompt_ids to run, one per line")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)

    benchmark_records = load_jsonl(args.benchmark)
    for record in benchmark_records:
        validate_benchmark_record(record)

    selected_records = select_records(
        benchmark_records,
        limit=args.limit,
        start_index=args.start_index,
        prompt_ids_path=args.prompt_ids,
    )

    print(f"Loaded benchmark records: {len(benchmark_records)}")
    print(f"Selected records: {len(selected_records)}")

    if args.raw_predictions is not None:
        raw_predictions = load_jsonl(args.raw_predictions)
        print(f"Loaded raw prediction rows: {len(raw_predictions)}")
        evaluate_from_raw_predictions(
            benchmark_records=selected_records,
            raw_predictions=raw_predictions,
            output_path=args.output,
            overwrite=args.overwrite,
            log_every=args.log_every,
        )
        return

    if args.model_config is None or args.model_key is None:
        raise ValueError("Live inference mode requires --model-config and --model-key")

    evaluate_with_live_inference(
        benchmark_records=selected_records,
        model_config_path=args.model_config,
        model_key=args.model_key,
        output_path=args.output,
        overwrite=args.overwrite,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
