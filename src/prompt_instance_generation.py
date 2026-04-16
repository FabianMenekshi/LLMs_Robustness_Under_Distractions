import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

from src.prompt_templates import (
    PROMPT_REGIMES,
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


REGIMES = list(PROMPT_REGIMES.keys())

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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def _stable_index(*parts: Any) -> int:
    joined = "||".join(str(part) for part in parts if part is not None)
    return sum(ord(ch) for ch in joined)


def _build_balanced_variant_counters(
    base_records: List[Dict[str, Any]],
    regimes: List[str],
    distraction_types: List[str],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    counters: Dict[Tuple[str, str], int] = defaultdict(int)
    variant_lookup: Dict[Tuple[str, str], Dict[str, int]] = {}

    ordered_records = sorted(
        base_records,
        key=lambda r: (r["task_name"], r["example_id"]),
    )

    for regime in regimes:
        for distraction_type in distraction_types:
            for record in ordered_records:
                key = (record["example_id"], regime)
                variant_lookup.setdefault(key, {})
                variant_lookup[key][distraction_type] = counters[(regime, distraction_type)]
                counters[(regime, distraction_type)] += 1

    return variant_lookup


def _choose_clean_prompt_and_metadata(
    base_record: Dict[str, Any],
    regime: str,
    variant_index: int,
) -> Dict[str, Any]:
    if regime == "bounded":
        opener = choose_bounded_opener(base_record, variant_index)
        layout = choose_bounded_layout(base_record, variant_index)

        prompt_text = render_bounded_clean_prompt(
            base_record,
            opener=opener,
            layout=layout,
        )

        return {
            "prompt_text": prompt_text,
            "prompt_surface_type": "bounded_tagged",
            "opener_id": opener.get("opener_id"),
            "opener_text": opener.get("text"),
            "layout_id": layout.get("layout_id"),
            "layout_name": layout.get("name"),
            "surface_id": None,
            "surface_name": None,
        }

    if regime == "unbounded":
        surface = choose_unbounded_surface(base_record, variant_index)
        prompt_text = render_unbounded_clean_prompt(
            base_record,
            surface=surface,
        )

        return {
            "prompt_text": prompt_text,
            "prompt_surface_type": surface.get("surface_family"),
            "opener_id": None,
            "opener_text": None,
            "layout_id": None,
            "layout_name": None,
            "surface_id": surface.get("surface_id"),
            "surface_name": surface.get("name"),
        }

    raise ValueError(f"Unknown regime: {regime}")


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
    raise ValueError(f"Unknown conflicting placement: {placement}")


def _insert_negation(clean_prompt: str, negation_text: str, placement: str) -> str:
    if placement == "prefix":
        return f"{negation_text}\n\n{clean_prompt}"
    if placement == "suffix":
        return f"{clean_prompt}\n\n{negation_text}"
    raise ValueError(f"Unknown negation placement: {placement}")


def _insert_style(clean_prompt: str, style_text: str, placement: str) -> str:
    if placement == "prefix":
        return f"{style_text}\n\n{clean_prompt}"
    if placement == "suffix":
        return f"{clean_prompt}\n\n{style_text}"
    raise ValueError(f"Unknown style placement: {placement}")


def _insert_length_stress(clean_prompt: str, long_noise_text: str, placement: str) -> str:
    if placement == "prefix":
        return f"{long_noise_text}\n\n{clean_prompt}"
    if placement == "suffix":
        return f"{clean_prompt}\n\n{long_noise_text}"
    raise ValueError(f"Unknown length-stress placement: {placement}")


def _choose_distraction_material(
    base_record: Dict[str, Any],
    regime: str,
    distraction_type: str,
    variant_index: int,
) -> Dict[str, Any]:
    if distraction_type == "clean":
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

    if distraction_type == "irrelevant_prefix":
        noise = choose_short_noise(base_record, variant_index)
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

    if distraction_type == "irrelevant_suffix":
        noise = choose_short_noise(base_record, variant_index)
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

    if distraction_type == "instruction_in_the_middle":
        noise_before = choose_short_noise(base_record, variant_index)
        noise_after = choose_short_noise(base_record, variant_index + 1)
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

    if distraction_type == "conflicting_instruction":
        conflict = choose_conflicting_instruction(base_record, variant_index)
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

    if distraction_type == "negation_distraction":
        negation = choose_negation_text(base_record, variant_index)
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

    if distraction_type == "style_distraction":
        style = choose_style_distraction(base_record, variant_index)
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

    if distraction_type == "length_stress":
        long_noise = choose_long_noise(base_record, variant_index)
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

    raise ValueError(f"Unknown distraction_type: {distraction_type}")


def _apply_distraction(
    clean_prompt: str,
    distraction_type: str,
    materials: Dict[str, Any],
) -> str:
    if distraction_type == "clean":
        return clean_prompt

    if distraction_type == "irrelevant_prefix":
        return _insert_irrelevant_prefix(
            clean_prompt=clean_prompt,
            noise_text=materials["noise_block"]["text"],
        )

    if distraction_type == "irrelevant_suffix":
        return _insert_irrelevant_suffix(
            clean_prompt=clean_prompt,
            noise_text=materials["noise_block"]["text"],
        )

    if distraction_type == "instruction_in_the_middle":
        return _insert_instruction_in_the_middle(
            clean_prompt=clean_prompt,
            noise_before=materials["noise_block"]["text"],
            noise_after=materials["noise_block_2"]["text"],
        )

    if distraction_type == "conflicting_instruction":
        return _insert_conflicting_instruction(
            clean_prompt=clean_prompt,
            conflict_text=materials["conflict_block"]["text"],
            placement=materials["placement"],
        )

    if distraction_type == "negation_distraction":
        return _insert_negation(
            clean_prompt=clean_prompt,
            negation_text=materials["negation_block"]["text"],
            placement=materials["placement"],
        )

    if distraction_type == "style_distraction":
        return _insert_style(
            clean_prompt=clean_prompt,
            style_text=materials["style_block"]["text"],
            placement=materials["placement"],
        )

    if distraction_type == "length_stress":
        return _insert_length_stress(
            clean_prompt=clean_prompt,
            long_noise_text=materials["long_noise_block"]["text"],
            placement=materials["placement"],
        )

    raise ValueError(f"Unknown distraction_type: {distraction_type}")


def build_prompt_id(
    base_example_id: str,
    regime: str,
    distraction_type: str,
) -> str:
    return f"{base_example_id}__{regime}__{distraction_type}"


def build_prompt_record(
    base_record: Dict[str, Any],
    regime: str,
    distraction_type: str,
    variant_index: int,
) -> Dict[str, Any]:
    clean_info = _choose_clean_prompt_and_metadata(
        base_record=base_record,
        regime=regime,
        variant_index=variant_index,
    )

    materials = _choose_distraction_material(
        base_record=base_record,
        regime=regime,
        distraction_type=distraction_type,
        variant_index=variant_index,
    )

    prompt_text = _apply_distraction(
        clean_prompt=clean_info["prompt_text"],
        distraction_type=distraction_type,
        materials=materials,
    )

    noise_block = materials.get("noise_block")
    noise_block_2 = materials.get("noise_block_2")
    long_noise_block = materials.get("long_noise_block")
    conflict_block = materials.get("conflict_block")
    negation_block = materials.get("negation_block")
    style_block = materials.get("style_block")

    return {
        "prompt_id": build_prompt_id(
            base_example_id=base_record["example_id"],
            regime=regime,
            distraction_type=distraction_type,
        ),
        "base_example_id": base_record["example_id"],
        "task_name": base_record["task_name"],
        "distraction_type": distraction_type,
        "distraction_subtype": materials.get("distraction_subtype"),
        "distraction_variant_id": materials.get("distraction_variant_id"),
        "regime": regime,
        "is_clean": distraction_type == "clean",
        "prompt_text": prompt_text,
        "gold_output": base_record["gold_output"],
        "source_instruction": base_record["instruction"],
        "source_template_name": base_record.get("template_name"),
        "prompt_surface_type": clean_info.get("prompt_surface_type"),
        "surface_id": clean_info.get("surface_id"),
        "surface_name": clean_info.get("surface_name"),
        "opener_id": clean_info.get("opener_id"),
        "opener_text": clean_info.get("opener_text"),
        "layout_id": clean_info.get("layout_id"),
        "layout_name": clean_info.get("layout_name"),
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
        "variant_index": variant_index,
    }


def build_all_prompt_instances(
    base_records: List[Dict[str, Any]],
    regimes: Optional[List[str]] = None,
    distraction_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    prompt_records: List[Dict[str, Any]] = []

    active_regimes = regimes or REGIMES
    active_distraction_types = distraction_types or DISTRACTION_TYPES

    variant_lookup = _build_balanced_variant_counters(
        base_records=base_records,
        regimes=active_regimes,
        distraction_types=active_distraction_types,
    )

    ordered_records = sorted(
        base_records,
        key=lambda r: (r["task_name"], r["example_id"]),
    )

    for base_record in ordered_records:
        for regime in active_regimes:
            for distraction_type in active_distraction_types:
                variant_index = variant_lookup[(base_record["example_id"], regime)][distraction_type]
                prompt_records.append(
                    build_prompt_record(
                        base_record=base_record,
                        regime=regime,
                        distraction_type=distraction_type,
                        variant_index=variant_index,
                    )
                )

    return prompt_records


def _sorted_counter(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def build_prompt_summary(prompt_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts_by_task = Counter(record["task_name"] for record in prompt_records)
    counts_by_regime = Counter(record["regime"] for record in prompt_records)
    counts_by_distraction = Counter(record["distraction_type"] for record in prompt_records)
    counts_by_subtype = Counter(
        record["distraction_subtype"]
        for record in prompt_records
        if record.get("distraction_subtype")
    )
    counts_by_surface_type = Counter(
        record["prompt_surface_type"]
        for record in prompt_records
        if record.get("prompt_surface_type")
    )
    counts_by_layout = Counter(
        record["layout_name"]
        for record in prompt_records
        if record.get("layout_name")
    )
    counts_by_clean_flag = Counter(str(record["is_clean"]).lower() for record in prompt_records)

    # New finer-grained counts
    counts_by_surface_id = Counter(
        record["surface_id"]
        for record in prompt_records
        if record.get("surface_id")
    )
    counts_by_surface_name = Counter(
        record["surface_name"]
        for record in prompt_records
        if record.get("surface_name")
    )
    counts_by_opener_id = Counter(
        record["opener_id"]
        for record in prompt_records
        if record.get("opener_id")
    )
    counts_by_conflict_variant_id = Counter(
        record["conflict_variant_id"]
        for record in prompt_records
        if record.get("conflict_variant_id")
    )
    counts_by_negation_variant_id = Counter(
        record["negation_variant_id"]
        for record in prompt_records
        if record.get("negation_variant_id")
    )
    counts_by_style_id = Counter(
        record["style_id"]
        for record in prompt_records
        if record.get("style_id")
    )
    counts_by_noise_block_id = Counter(
        record["noise_block_id"]
        for record in prompt_records
        if record.get("noise_block_id")
    )
    counts_by_noise_block_id_2 = Counter(
        record["noise_block_id_2"]
        for record in prompt_records
        if record.get("noise_block_id_2")
    )
    counts_by_long_noise_block_id = Counter(
        record["long_noise_block_id"]
        for record in prompt_records
        if record.get("long_noise_block_id")
    )

    task_regime = defaultdict(int)
    task_distraction = defaultdict(int)
    task_subtype = defaultdict(int)

    for record in prompt_records:
        task_regime[f"{record['task_name']}__{record['regime']}"] += 1
        task_distraction[f"{record['task_name']}__{record['distraction_type']}"] += 1
        if record.get("distraction_subtype"):
            task_subtype[f"{record['task_name']}__{record['distraction_subtype']}"] += 1

    return {
        "total_prompt_instances": len(prompt_records),
        "counts_by_task": _sorted_counter(counts_by_task),
        "counts_by_regime": _sorted_counter(counts_by_regime),
        "counts_by_distraction_type": _sorted_counter(counts_by_distraction),
        "counts_by_distraction_subtype": _sorted_counter(counts_by_subtype),
        "counts_by_prompt_surface_type": _sorted_counter(counts_by_surface_type),
        "counts_by_layout_name": _sorted_counter(counts_by_layout),
        "counts_by_clean_flag": _sorted_counter(counts_by_clean_flag),
        "counts_by_surface_id": _sorted_counter(counts_by_surface_id),
        "counts_by_surface_name": _sorted_counter(counts_by_surface_name),
        "counts_by_opener_id": _sorted_counter(counts_by_opener_id),
        "counts_by_conflict_variant_id": _sorted_counter(counts_by_conflict_variant_id),
        "counts_by_negation_variant_id": _sorted_counter(counts_by_negation_variant_id),
        "counts_by_style_id": _sorted_counter(counts_by_style_id),
        "counts_by_noise_block_id": _sorted_counter(counts_by_noise_block_id),
        "counts_by_noise_block_id_2": _sorted_counter(counts_by_noise_block_id_2),
        "counts_by_long_noise_block_id": _sorted_counter(counts_by_long_noise_block_id),
        "counts_by_task_and_regime": dict(sorted(task_regime.items(), key=lambda item: item[0])),
        "counts_by_task_and_distraction_type": dict(sorted(task_distraction.items(), key=lambda item: item[0])),
        "counts_by_task_and_distraction_subtype": dict(sorted(task_subtype.items(), key=lambda item: item[0])),
    }


def build_prompt_preview_samples(
    prompt_records: List[Dict[str, Any]],
    n_per_condition: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build a more representative preview sample.

    Strategy:
    - group by (task, regime, distraction_type)
    - within each group, bucket by visible prompt-variation signature
    - rotate across buckets
    - start from a deterministic offset instead of always bucket 0
    so previews do not always overrepresent earliest ids like us_001 / bl_001 / bo_001
    """
    grouped = defaultdict(list)

    for record in prompt_records:
        key = (
            record["task_name"],
            record["regime"],
            record["distraction_type"],
        )
        grouped[key].append(record)

    preview_records: List[Dict[str, Any]] = []

    for key in sorted(grouped.keys()):
        records = grouped[key]

        buckets = defaultdict(list)
        for record in records:
            bucket_key = (
                record.get("surface_id"),
                record.get("layout_id"),
                record.get("distraction_variant_id"),
                record.get("conflict_variant_id"),
                record.get("negation_variant_id"),
                record.get("style_id"),
                record.get("noise_block_id"),
                record.get("noise_block_id_2"),
                record.get("long_noise_block_id"),
            )
            buckets[bucket_key].append(record)

        ordered_bucket_keys = sorted(
            buckets.keys(),
            key=lambda bucket: tuple("" if value is None else str(value) for value in bucket),
        )

        # deterministic offset so every condition does not always start from the first bucket
        offset_seed = "||".join(str(part) for part in key)
        start_index = sum(ord(ch) for ch in offset_seed) % len(ordered_bucket_keys)

        rotated_bucket_keys = (
            ordered_bucket_keys[start_index:] + ordered_bucket_keys[:start_index]
        )

        selected = []
        bucket_positions = {bucket_key: 0 for bucket_key in rotated_bucket_keys}

        while len(selected) < n_per_condition:
            made_progress = False

            for bucket_key in rotated_bucket_keys:
                pos = bucket_positions[bucket_key]
                bucket = buckets[bucket_key]

                if pos < len(bucket):
                    selected.append(bucket[pos])
                    bucket_positions[bucket_key] += 1
                    made_progress = True

                    if len(selected) == n_per_condition:
                        break

            if not made_progress:
                break

        preview_records.extend(selected)

    return preview_records