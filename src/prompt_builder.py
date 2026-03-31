import json
import os
from collections import defaultdict
from typing import Dict, Any, List

from src.prompt_templates import (
    DISTRACTION_TEMPLATES,
    NOISE_LIBRARY,
    LONG_NOISE_LIBRARY,
    STYLE_DISTRACTIONS,
    CONFLICTING_INSTRUCTION_TEXT,
    get_negation_text,
    render_bounded_clean_prompt,
    render_unbounded_clean_prompt,
    build_prompt_design_spec,
)


def load_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL records from disk.
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: Any, output_path: str) -> None:
    """
    Save a JSON-serializable object as pretty JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def choose_noise_block(index: int) -> Dict[str, str]:
    """
    Deterministically select one short noise block and one long noise block.
    """
    short_categories = sorted(NOISE_LIBRARY.keys())
    short_category = short_categories[index % len(short_categories)]
    short_options = NOISE_LIBRARY[short_category]
    short_text = short_options[index % len(short_options)]

    long_categories = sorted(LONG_NOISE_LIBRARY.keys())
    long_category = long_categories[index % len(long_categories)]
    long_text = LONG_NOISE_LIBRARY[long_category]

    return {
        "short_category": short_category,
        "short_text": short_text,
        "long_category": long_category,
        "long_text": long_text,
    }


def render_clean_prompt(record: Dict[str, Any], regime: str) -> str:
    """
    Render a canonical clean prompt under the specified regime.
    """
    if regime == "bounded":
        return render_bounded_clean_prompt(record)
    if regime == "unbounded":
        return render_unbounded_clean_prompt(record)

    raise ValueError(f"Unknown regime: {regime}")


def render_distracted_prompt(
    record: Dict[str, Any],
    regime: str,
    distraction_name: str,
    noise_index: int = 0,
    style_index: int = 0,
) -> str:
    """
    Render one distracted prompt for a given base record, regime, and distraction type.
    """
    clean_prompt = render_clean_prompt(record, regime)
    template = DISTRACTION_TEMPLATES[distraction_name]
    selected_noise = choose_noise_block(noise_index)

    negation_text = get_negation_text(record)
    style_text = STYLE_DISTRACTIONS[style_index % len(STYLE_DISTRACTIONS)]

    template_key = f"{regime}_template"
    wrapper = template[template_key]

    return wrapper.format(
        clean_prompt=clean_prompt,
        noise_block=selected_noise["short_text"],
        noise_block_1=selected_noise["short_text"],
        noise_block_2=choose_noise_block(noise_index + 1)["short_text"],
        long_noise_block=selected_noise["long_text"],
        negation_text=negation_text,
        style_text=style_text,
        conflicting_text=CONFLICTING_INSTRUCTION_TEXT,
    )


def build_clean_prompt_record(record: Dict[str, Any], regime: str) -> Dict[str, Any]:
    """
    Build a structured clean prompt record.
    """
    return {
        "example_id": record["example_id"],
        "task_name": record["task_name"],
        "regime": regime,
        "prompt_type": "clean",
        "prompt_text": render_clean_prompt(record, regime),
        "gold_output": record["gold_output"],
    }


def build_distracted_prompt_record(
    record: Dict[str, Any],
    regime: str,
    distraction_name: str,
    noise_index: int = 0,
    style_index: int = 0,
) -> Dict[str, Any]:
    """
    Build a structured distracted prompt record.
    """
    selected_noise = choose_noise_block(noise_index)

    return {
        "example_id": record["example_id"],
        "task_name": record["task_name"],
        "regime": regime,
        "prompt_type": "distracted",
        "distraction_name": distraction_name,
        "noise_category": selected_noise["short_category"],
        "long_noise_category": selected_noise["long_category"],
        "prompt_text": render_distracted_prompt(
            record=record,
            regime=regime,
            distraction_name=distraction_name,
            noise_index=noise_index,
            style_index=style_index,
        ),
        "gold_output": record["gold_output"],
    }


def select_preview_records_by_task(
    records: List[Dict[str, Any]],
    examples_per_task: int = 2,
) -> List[Dict[str, Any]]:
    """
    Select a small preview subset that covers all task families.

    This avoids preview files that only show one task family.
    """
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task_name"]].append(record)

    selected = []
    for task_name in sorted(grouped.keys()):
        selected.extend(grouped[task_name][:examples_per_task])

    return selected


def build_prompt_previews(
    records: List[Dict[str, Any]],
    examples_per_task: int = 2,
) -> List[Dict[str, Any]]:
    """
    Build a preview set of clean and distracted prompts for inspection.

    By default, this covers every task family.
    """
    previews = []
    subset = select_preview_records_by_task(records, examples_per_task=examples_per_task)

    for i, record in enumerate(subset):
        for regime in ["bounded", "unbounded"]:
            previews.append(build_clean_prompt_record(record, regime))

            for distraction_name in DISTRACTION_TEMPLATES.keys():
                previews.append(
                    build_distracted_prompt_record(
                        record=record,
                        regime=regime,
                        distraction_name=distraction_name,
                        noise_index=i,
                        style_index=i,
                    )
                )

    return previews


def save_prompt_previews(previews: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save prompt previews to JSON.
    """
    save_json(previews, output_path)


def export_prompt_design_spec(output_path: str) -> None:
    """
    Save the prompt-design specification for Phase 4.
    """
    spec = build_prompt_design_spec()
    save_json(spec, output_path)