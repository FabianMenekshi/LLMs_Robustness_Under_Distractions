import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional

from src.prompt_templates import (
    DISTRACTION_TEMPLATES,
    build_prompt_design_spec,
    render_bounded_clean_prompt,
    render_unbounded_clean_prompt,
    choose_bounded_opener,
    choose_bounded_layout,
    choose_unbounded_surface,
    choose_short_noise,
    choose_long_noise,
    choose_conflicting_instruction,
    choose_negation_text,
    choose_style_distraction,
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


def _stable_index(*parts: Any) -> int:
    """
    Deterministic selector used across prompt rendering helpers.
    """
    joined = "||".join(str(part) for part in parts if part is not None)
    return sum(ord(ch) for ch in joined)


def choose_clean_prompt_components(
    record: Dict[str, Any],
    regime: str,
    variant_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Choose the surface components for a clean prompt without yet rendering it.
    """
    base_index = (
        variant_index
        if variant_index is not None
        else _stable_index(record["example_id"], regime, "clean_surface")
    )

    if regime == "bounded":
        opener = choose_bounded_opener(record, base_index)
        layout = choose_bounded_layout(record, base_index + 1)
        return {
            "regime": regime,
            "prompt_surface_type": "bounded_tagged",
            "opener": opener,
            "layout": layout,
            "surface": None,
            "opener_id": opener.get("opener_id"),
            "opener_text": opener.get("text"),
            "layout_id": layout.get("layout_id"),
            "layout_name": layout.get("name"),
            "surface_id": None,
            "surface_name": None,
        }

    if regime == "unbounded":
        surface = choose_unbounded_surface(record, base_index)
        return {
            "regime": regime,
            "prompt_surface_type": surface.get("surface_family"),
            "opener": None,
            "layout": None,
            "surface": surface,
            "opener_id": None,
            "opener_text": None,
            "layout_id": None,
            "layout_name": None,
            "surface_id": surface.get("surface_id"),
            "surface_name": surface.get("name"),
        }

    raise ValueError(f"Unknown regime: {regime}")


def render_clean_prompt(
    record: Dict[str, Any],
    regime: str,
    variant_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Render a clean prompt and return both prompt text and rendering metadata.
    """
    components = choose_clean_prompt_components(
        record=record,
        regime=regime,
        variant_index=variant_index,
    )

    if regime == "bounded":
        prompt_text = render_bounded_clean_prompt(
            record,
            opener=components["opener"],
            layout=components["layout"],
        )
    elif regime == "unbounded":
        prompt_text = render_unbounded_clean_prompt(
            record,
            surface=components["surface"],
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return {
        "prompt_text": prompt_text,
        "prompt_surface_type": components["prompt_surface_type"],
        "surface_id": components["surface_id"],
        "surface_name": components["surface_name"],
        "opener_id": components["opener_id"],
        "opener_text": components["opener_text"],
        "layout_id": components["layout_id"],
        "layout_name": components["layout_name"],
    }


def choose_distraction_material(
    record: Dict[str, Any],
    regime: str,
    distraction_name: str,
    variant_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Select the structured material used to create a distraction.
    """
    base_index = (
        variant_index
        if variant_index is not None
        else _stable_index(record["example_id"], regime, distraction_name, "distraction")
    )

    if distraction_name == "clean":
        return {
            "distraction_subtype": None,
            "distraction_variant_id": None,
            "placement": None,
            "noise_block": None,
            "noise_block_2": None,
            "long_noise_block": None,
            "conflict_block": None,
            "negation_block": None,
            "style_block": None,
        }

    if distraction_name == "irrelevant_prefix":
        noise = choose_short_noise(record, base_index)
        return {
            "distraction_subtype": "short_prefix_noise",
            "distraction_variant_id": noise.get("block_id"),
            "placement": "prefix",
            "noise_block": noise,
            "noise_block_2": None,
            "long_noise_block": None,
            "conflict_block": None,
            "negation_block": None,
            "style_block": None,
        }

    if distraction_name == "irrelevant_suffix":
        noise = choose_short_noise(record, base_index)
        return {
            "distraction_subtype": "short_suffix_noise",
            "distraction_variant_id": noise.get("block_id"),
            "placement": "suffix",
            "noise_block": noise,
            "noise_block_2": None,
            "long_noise_block": None,
            "conflict_block": None,
            "negation_block": None,
            "style_block": None,
        }

    if distraction_name == "instruction_in_the_middle":
        noise_before = choose_short_noise(record, base_index)
        noise_after = choose_short_noise(record, base_index + 1)
        return {
            "distraction_subtype": "middle_burial",
            "distraction_variant_id": f"{noise_before.get('block_id')}__{noise_after.get('block_id')}",
            "placement": "sandwich",
            "noise_block": noise_before,
            "noise_block_2": noise_after,
            "long_noise_block": None,
            "conflict_block": None,
            "negation_block": None,
            "style_block": None,
        }

    if distraction_name == "conflicting_instruction":
        conflict = choose_conflicting_instruction(record, base_index)
        return {
            "distraction_subtype": conflict.get("subtype"),
            "distraction_variant_id": conflict.get("variant_id"),
            "placement": conflict.get("placement", "suffix"),
            "noise_block": None,
            "noise_block_2": None,
            "long_noise_block": None,
            "conflict_block": conflict,
            "negation_block": None,
            "style_block": None,
        }

    if distraction_name == "negation_distraction":
        negation = choose_negation_text(record, base_index)
        return {
            "distraction_subtype": negation.get("subtype"),
            "distraction_variant_id": negation.get("variant_id"),
            "placement": negation.get("placement", "suffix"),
            "noise_block": None,
            "noise_block_2": None,
            "long_noise_block": None,
            "conflict_block": None,
            "negation_block": negation,
            "style_block": None,
        }

    if distraction_name == "style_distraction":
        style = choose_style_distraction(record, base_index)
        return {
            "distraction_subtype": style.get("style_family"),
            "distraction_variant_id": style.get("style_id"),
            "placement": style.get("placement", "suffix"),
            "noise_block": None,
            "noise_block_2": None,
            "long_noise_block": None,
            "conflict_block": None,
            "negation_block": None,
            "style_block": style,
        }

    if distraction_name == "length_stress":
        long_noise = choose_long_noise(record, base_index)
        return {
            "distraction_subtype": long_noise.get("category"),
            "distraction_variant_id": long_noise.get("block_id"),
            "placement": long_noise.get("placement", "prefix"),
            "noise_block": None,
            "noise_block_2": None,
            "long_noise_block": long_noise,
            "conflict_block": None,
            "negation_block": None,
            "style_block": None,
        }

    raise ValueError(f"Unknown distraction_name: {distraction_name}")


def _insert_irrelevant_prefix(clean_prompt: str, noise_text: str) -> str:
    return f"{noise_text}\n\n{clean_prompt}"


def _insert_irrelevant_suffix(clean_prompt: str, noise_text: str) -> str:
    return f"{clean_prompt}\n\n{noise_text}"


def _insert_instruction_in_the_middle(
    clean_prompt: str,
    noise_before: str,
    noise_after: str,
) -> str:
    return f"{noise_before}\n\n{clean_prompt}\n\n{noise_after}"


def _insert_conflicting_instruction(
    clean_prompt: str,
    conflict_text: str,
    placement: str,
) -> str:
    if placement == "prefix":
        return f"{conflict_text}\n\n{clean_prompt}"
    if placement == "suffix":
        return f"{clean_prompt}\n\n{conflict_text}"
    if placement == "sandwich":
        return f"{conflict_text}\n\n{clean_prompt}"
    raise ValueError(f"Unknown placement for conflicting instruction: {placement}")


def _insert_negation(clean_prompt: str, negation_text: str, placement: str) -> str:
    if placement == "prefix":
        return f"{negation_text}\n\n{clean_prompt}"
    if placement == "suffix":
        return f"{clean_prompt}\n\n{negation_text}"
    raise ValueError(f"Unknown placement for negation distraction: {placement}")


def _insert_style(clean_prompt: str, style_text: str, placement: str) -> str:
    if placement == "prefix":
        return f"{style_text}\n\n{clean_prompt}"
    if placement == "suffix":
        return f"{clean_prompt}\n\n{style_text}"
    raise ValueError(f"Unknown placement for style distraction: {placement}")


def _insert_length_stress(clean_prompt: str, long_noise_text: str, placement: str) -> str:
    if placement == "prefix":
        return f"{long_noise_text}\n\n{clean_prompt}"
    if placement == "suffix":
        return f"{clean_prompt}\n\n{long_noise_text}"
    raise ValueError(f"Unknown placement for length stress: {placement}")


def apply_distraction(
    clean_prompt: str,
    distraction_name: str,
    materials: Dict[str, Any],
) -> str:
    """
    Apply a distraction to a rendered clean prompt.
    """
    if distraction_name == "clean":
        return clean_prompt

    if distraction_name == "irrelevant_prefix":
        return _insert_irrelevant_prefix(
            clean_prompt=clean_prompt,
            noise_text=materials["noise_block"]["text"],
        )

    if distraction_name == "irrelevant_suffix":
        return _insert_irrelevant_suffix(
            clean_prompt=clean_prompt,
            noise_text=materials["noise_block"]["text"],
        )

    if distraction_name == "instruction_in_the_middle":
        return _insert_instruction_in_the_middle(
            clean_prompt=clean_prompt,
            noise_before=materials["noise_block"]["text"],
            noise_after=materials["noise_block_2"]["text"],
        )

    if distraction_name == "conflicting_instruction":
        return _insert_conflicting_instruction(
            clean_prompt=clean_prompt,
            conflict_text=materials["conflict_block"]["text"],
            placement=materials["placement"],
        )

    if distraction_name == "negation_distraction":
        return _insert_negation(
            clean_prompt=clean_prompt,
            negation_text=materials["negation_block"]["text"],
            placement=materials["placement"],
        )

    if distraction_name == "style_distraction":
        return _insert_style(
            clean_prompt=clean_prompt,
            style_text=materials["style_block"]["text"],
            placement=materials["placement"],
        )

    if distraction_name == "length_stress":
        return _insert_length_stress(
            clean_prompt=clean_prompt,
            long_noise_text=materials["long_noise_block"]["text"],
            placement=materials["placement"],
        )

    raise ValueError(f"Unknown distraction_name: {distraction_name}")


def render_distracted_prompt(
    record: Dict[str, Any],
    regime: str,
    distraction_name: str,
    variant_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Render one distracted prompt for a given base record, regime, and distraction type.
    Returns prompt text together with surface/distraction metadata.
    """
    clean_info = render_clean_prompt(
        record=record,
        regime=regime,
        variant_index=variant_index,
    )

    materials = choose_distraction_material(
        record=record,
        regime=regime,
        distraction_name=distraction_name,
        variant_index=variant_index,
    )

    prompt_text = apply_distraction(
        clean_prompt=clean_info["prompt_text"],
        distraction_name=distraction_name,
        materials=materials,
    )

    noise_block = materials.get("noise_block")
    noise_block_2 = materials.get("noise_block_2")
    long_noise_block = materials.get("long_noise_block")
    conflict_block = materials.get("conflict_block")
    negation_block = materials.get("negation_block")
    style_block = materials.get("style_block")

    return {
        "prompt_text": prompt_text,
        "prompt_surface_type": clean_info.get("prompt_surface_type"),
        "surface_id": clean_info.get("surface_id"),
        "surface_name": clean_info.get("surface_name"),
        "opener_id": clean_info.get("opener_id"),
        "opener_text": clean_info.get("opener_text"),
        "layout_id": clean_info.get("layout_id"),
        "layout_name": clean_info.get("layout_name"),
        "distraction_subtype": materials.get("distraction_subtype"),
        "distraction_variant_id": materials.get("distraction_variant_id"),
        "placement": materials.get("placement"),
        "noise_block_id": noise_block.get("block_id") if noise_block else None,
        "noise_category": noise_block.get("category") if noise_block else None,
        "noise_length": noise_block.get("length") if noise_block else None,
        "noise_block_id_2": noise_block_2.get("block_id") if noise_block_2 else None,
        "noise_category_2": noise_block_2.get("category") if noise_block_2 else None,
        "long_noise_block_id": long_noise_block.get("block_id") if long_noise_block else None,
        "long_noise_category": long_noise_block.get("category") if long_noise_block else None,
        "long_noise_length": long_noise_block.get("length") if long_noise_block else None,
        "conflict_variant_id": conflict_block.get("variant_id") if conflict_block else None,
        "conflict_subtype": conflict_block.get("subtype") if conflict_block else None,
        "negation_variant_id": negation_block.get("variant_id") if negation_block else None,
        "negation_subtype": negation_block.get("subtype") if negation_block else None,
        "style_id": style_block.get("style_id") if style_block else None,
        "style_family": style_block.get("style_family") if style_block else None,
    }


def build_clean_prompt_record(
    record: Dict[str, Any],
    regime: str,
    variant_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a structured clean prompt record for preview or inspection.
    """
    clean_info = render_clean_prompt(
        record=record,
        regime=regime,
        variant_index=variant_index,
    )

    return {
        "example_id": record["example_id"],
        "task_name": record["task_name"],
        "regime": regime,
        "prompt_type": "clean",
        "prompt_text": clean_info["prompt_text"],
        "gold_output": record["gold_output"],
        "source_instruction": record["instruction"],
        "source_template_name": record.get("template_name"),
        "prompt_surface_type": clean_info.get("prompt_surface_type"),
        "surface_id": clean_info.get("surface_id"),
        "surface_name": clean_info.get("surface_name"),
        "opener_id": clean_info.get("opener_id"),
        "opener_text": clean_info.get("opener_text"),
        "layout_id": clean_info.get("layout_id"),
        "layout_name": clean_info.get("layout_name"),
    }


def build_distracted_prompt_record(
    record: Dict[str, Any],
    regime: str,
    distraction_name: str,
    variant_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a structured distracted prompt record for preview or inspection.
    """
    distracted_info = render_distracted_prompt(
        record=record,
        regime=regime,
        distraction_name=distraction_name,
        variant_index=variant_index,
    )

    return {
        "example_id": record["example_id"],
        "task_name": record["task_name"],
        "regime": regime,
        "prompt_type": "distracted",
        "distraction_name": distraction_name,
        "prompt_text": distracted_info["prompt_text"],
        "gold_output": record["gold_output"],
        "source_instruction": record["instruction"],
        "source_template_name": record.get("template_name"),
        "prompt_surface_type": distracted_info.get("prompt_surface_type"),
        "surface_id": distracted_info.get("surface_id"),
        "surface_name": distracted_info.get("surface_name"),
        "opener_id": distracted_info.get("opener_id"),
        "opener_text": distracted_info.get("opener_text"),
        "layout_id": distracted_info.get("layout_id"),
        "layout_name": distracted_info.get("layout_name"),
        "distraction_subtype": distracted_info.get("distraction_subtype"),
        "distraction_variant_id": distracted_info.get("distraction_variant_id"),
        "placement": distracted_info.get("placement"),
        "noise_block_id": distracted_info.get("noise_block_id"),
        "noise_category": distracted_info.get("noise_category"),
        "noise_length": distracted_info.get("noise_length"),
        "noise_block_id_2": distracted_info.get("noise_block_id_2"),
        "noise_category_2": distracted_info.get("noise_category_2"),
        "long_noise_block_id": distracted_info.get("long_noise_block_id"),
        "long_noise_category": distracted_info.get("long_noise_category"),
        "long_noise_length": distracted_info.get("long_noise_length"),
        "conflict_variant_id": distracted_info.get("conflict_variant_id"),
        "conflict_subtype": distracted_info.get("conflict_subtype"),
        "negation_variant_id": distracted_info.get("negation_variant_id"),
        "negation_subtype": distracted_info.get("negation_subtype"),
        "style_id": distracted_info.get("style_id"),
        "style_family": distracted_info.get("style_family"),
    }


def select_preview_records_by_task(
    records: List[Dict[str, Any]],
    examples_per_task: int = 2,
) -> List[Dict[str, Any]]:
    """
    Select a small preview subset that covers all task families.
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
            previews.append(
                build_clean_prompt_record(
                    record=record,
                    regime=regime,
                    variant_index=i,
                )
            )

            for distraction_name in DISTRACTION_TEMPLATES.keys():
                previews.append(
                    build_distracted_prompt_record(
                        record=record,
                        regime=regime,
                        distraction_name=distraction_name,
                        variant_index=i,
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
    Save the prompt-design specification.
    """
    spec = build_prompt_design_spec()
    save_json(spec, output_path)