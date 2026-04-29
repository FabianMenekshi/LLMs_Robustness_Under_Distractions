#!/usr/bin/env python3
"""
Run raw LLM inference on the frozen prompt benchmark.

In this script, we:
- load benchmark JSONL records
- send each record['prompt_text'] to one model
- save raw model outputs plus copied benchmark metadata
- support safe checkpoint/resume

This script intentionally does NOT parse or score outputs. Parsing/scoring belongs to
Phase 2 so the raw generation artifact stays independent and auditable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_BENCHMARK_FIELDS = {
    "prompt_id",
    "base_example_id",
    "task_name",
    "regime",
    "distraction_type",
    "is_clean",
    "prompt_text",
    "gold_output",
}

DEFAULT_METADATA_FIELDS_TO_COPY = [
    "distraction_subtype",
    "distraction_variant_id",
    "prompt_surface_type",
    "surface_id",
    "surface_name",
    "opener_id",
    "layout_id",
    "layout_name",
    "placement",
    "noise_block_id",
    "noise_block_id_2",
    "long_noise_block_id",
    "conflict_variant_id",
    "conflict_subtype",
    "negation_variant_id",
    "negation_subtype",
    "style_id",
    "style_family",
    "source_instruction",
    "source_template_name",
]


@dataclass(frozen=True)
class ModelSettings:
    model_key: str
    hf_id: str
    dtype: str
    use_chat_template: bool
    trust_remote_code: bool
    max_new_tokens_default: int
    max_new_tokens_by_task: Dict[str, int]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()


def validate_benchmark_record(record: Dict[str, Any]) -> None:
    missing = REQUIRED_BENCHMARK_FIELDS - set(record)
    if missing:
        prompt_id = record.get("prompt_id", "<missing_prompt_id>")
        raise ValueError(f"Benchmark record {prompt_id} is missing fields: {sorted(missing)}")

    if not isinstance(record.get("prompt_text"), str) or not record["prompt_text"].strip():
        raise ValueError(f"Benchmark record {record.get('prompt_id')} has empty prompt_text")


def read_completed_prompt_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()

    completed: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A partially written final line can happen if a job is killed.
                # Do not resume blindly; surface the issue to the user.
                raise ValueError(
                    f"Existing output file has invalid JSON at line {line_number}: {output_path}. "
                    "Fix or remove the partial line before resuming."
                )
            prompt_id = row.get("prompt_id")
            if isinstance(prompt_id, str):
                completed.add(prompt_id)
    return completed


def resolve_model_settings(model_config: Dict[str, Any], model_key: str) -> ModelSettings:
    models = model_config.get("models", [])
    model_entry = next((m for m in models if m.get("model_key") == model_key), None)
    if model_entry is None:
        available = [m.get("model_key") for m in models]
        raise KeyError(f"Unknown model_key={model_key!r}. Available: {available}")

    inference_defaults = model_config.get("inference_defaults", {})
    max_new_tokens_default = int(inference_defaults.get("max_new_tokens_default", 64))
    max_new_tokens_by_task = {
        str(k): int(v)
        for k, v in inference_defaults.get("max_new_tokens_by_task", {}).items()
    }

    return ModelSettings(
        model_key=str(model_entry["model_key"]),
        hf_id=str(model_entry["hf_id"]),
        dtype=str(model_entry.get("dtype", inference_defaults.get("dtype", "bfloat16"))),
        use_chat_template=bool(model_entry.get("use_chat_template", inference_defaults.get("use_chat_template", True))),
        trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
        max_new_tokens_default=max_new_tokens_default,
        max_new_tokens_by_task=max_new_tokens_by_task,
    )


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_json_schema_instruction(task_name: str) -> str:
    if task_name == "single_label_classification":
        return (
            'Return ONLY valid JSON with exactly one key: "label". '
            'Do not add any explanation, markdown, or extra text. '
            'Example: {"label": "positive"}'
        )
    if task_name == "multi_label_classification":
        return (
            'Return ONLY valid JSON with exactly one key: "labels". '
            'The value must be a JSON list of strings. '
            'Do not add any explanation, markdown, or extra text. '
            'Example: {"labels": ["health", "tech"]}'
        )
    if task_name == "information_extraction":
        return (
            'Return ONLY valid JSON with exactly these keys: '
            '"person", "date", "location". '
            'All values must be strings. '
            'Do not add any explanation, markdown, or extra text. '
            'Example: {"person": "Alice Smith", "date": "2025-04-30", "location": "Rome"}'
        )
    if task_name == "rule_based_transformation":
        return (
            'Return ONLY valid JSON with exactly one key: "text". '
            'Do not add any explanation, markdown, or extra text. '
            'Example: {"text": "alice visited rome"}'
        )
    if task_name == "extractive_qa":
        return (
            'Return ONLY valid JSON with exactly one key: "answer". '
            'The answer must be an exact span from the passage when the task asks for that. '
            'Do not add any explanation, markdown, or extra text. '
            'Example: {"answer": "Rome"}'
        )
    raise ValueError(f"Unsupported task_name for JSON schema instruction: {task_name}")

def build_model_input(
    tokenizer: Any,
    prompt_text: str,
    task_name: str,
    use_chat_template: bool,
) -> str:
    schema_instruction = get_json_schema_instruction(task_name)
    wrapped_prompt = (
        f"{schema_instruction}\n\n"
        f"Your response must be valid JSON only.\n"
        f"Do not include prose before or after the JSON object.\n\n"
        f"{prompt_text}"
    )

    if not use_chat_template:
        return wrapped_prompt

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return wrapped_prompt

    messages = [{"role": "user", "content": wrapped_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(settings: ModelSettings):
    dtype = torch_dtype_from_name(settings.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_id,
        trust_remote_code=settings.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=settings.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    task_name: str,
    use_chat_template: bool,
    max_new_tokens: int,
) -> Dict[str, Any]:
    rendered_input = build_model_input(
    tokenizer=tokenizer,
    prompt_text=prompt_text,
    task_name=task_name,
    use_chat_template=use_chat_template,
)

    inputs = tokenizer(
        rendered_input,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_token_count = int(inputs["input_ids"].shape[-1])
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    latency_s = time.perf_counter() - start_time
    generated_ids = output_ids[0, input_token_count:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "raw_output": raw_output.strip(),
        "input_token_count": input_token_count,
        "output_token_count": int(generated_ids.shape[-1]),
        "latency_s": latency_s,
    }


def build_output_row(
    benchmark_record: Dict[str, Any],
    settings: ModelSettings,
    generation: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "prompt_id": benchmark_record["prompt_id"],
        "base_example_id": benchmark_record["base_example_id"],
        "model_key": settings.model_key,
        "model_hf_id": settings.hf_id,
        "task_name": benchmark_record["task_name"],
        "regime": benchmark_record["regime"],
        "distraction_type": benchmark_record["distraction_type"],
        "is_clean": benchmark_record["is_clean"],
        "gold_output": benchmark_record["gold_output"],
        "raw_output": generation["raw_output"],
        "input_token_count": generation["input_token_count"],
        "output_token_count": generation["output_token_count"],
        "latency_s": generation["latency_s"],
        "max_new_tokens": max_new_tokens,
        "use_chat_template": settings.use_chat_template,
        "dtype": settings.dtype,
    }

    for field in DEFAULT_METADATA_FIELDS_TO_COPY:
        if field in benchmark_record:
            row[field] = benchmark_record[field]

    return row


def select_records(
    records: List[Dict[str, Any]],
    limit: Optional[int],
    start_index: int,
    prompt_ids_path: Optional[Path],
) -> List[Dict[str, Any]]:
    selected = records[start_index:]

    if prompt_ids_path is not None:
        wanted_ids = {
            line.strip()
            for line in prompt_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = [record for record in selected if record.get("prompt_id") in wanted_ids]

    if limit is not None:
        selected = selected[:limit]

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw inference on prompt_instances.jsonl")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to prompt_instances.jsonl")
    parser.add_argument("--model-config", type=Path, required=True, help="Path to model_config.json")
    parser.add_argument("--model-key", type=str, required=True, help="Model key from model_config.json")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path for raw predictions")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to run")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset into benchmark records")
    parser.add_argument("--prompt-ids", type=Path, default=None, help="Optional file containing prompt_ids to run, one per line")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of resuming")
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

    model_config = load_json(args.model_config)
    settings = resolve_model_settings(model_config, args.model_key)

    if args.overwrite and args.output.exists():
        args.output.unlink()

    completed_prompt_ids = read_completed_prompt_ids(args.output)
    pending_records = [
        record for record in selected_records
        if record["prompt_id"] not in completed_prompt_ids
    ]

    print(f"Loaded benchmark records: {len(benchmark_records)}")
    print(f"Selected records: {len(selected_records)}")
    print(f"Already completed: {len(completed_prompt_ids)}")
    print(f"Pending records: {len(pending_records)}")
    print(f"Loading model: {settings.model_key} ({settings.hf_id})")

    model, tokenizer = load_model_and_tokenizer(settings)

    try:
        for index, record in enumerate(pending_records, start=1):
            task_name = record["task_name"]
            max_new_tokens = settings.max_new_tokens_by_task.get(
                task_name,
                settings.max_new_tokens_default,
            )

            try:
                generation = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=record["prompt_text"],
                    task_name=record["task_name"],
                    use_chat_template=settings.use_chat_template,
                    max_new_tokens=max_new_tokens,
                )
                row = build_output_row(record, settings, generation, max_new_tokens)
            except Exception as exc:  # Save errors as rows so failures are auditable.
                row = {
                    "prompt_id": record["prompt_id"],
                    "base_example_id": record["base_example_id"],
                    "model_key": settings.model_key,
                    "model_hf_id": settings.hf_id,
                    "task_name": record["task_name"],
                    "regime": record["regime"],
                    "distraction_type": record["distraction_type"],
                    "is_clean": record["is_clean"],
                    "gold_output": record["gold_output"],
                    "raw_output": "",
                    "inference_error": repr(exc),
                    "max_new_tokens": max_new_tokens,
                    "use_chat_template": settings.use_chat_template,
                    "dtype": settings.dtype,
                }

            append_jsonl(args.output, [row])

            if index % args.log_every == 0 or index == len(pending_records):
                print(f"Completed {index}/{len(pending_records)} pending records")

    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()