import json
from pathlib import Path
from collections import Counter

import pandas as pd


SCORED_FILES = {
    "qwen7b": "evaluation/hpc_results/scored/full_scored_qwen25_7b_instruct_jsonforced.jsonl",
    "qwen14b": "evaluation/hpc_results/scored/full_scored_qwen25_14b_instruct_jsonforced.jsonl",
    "mistral_instruct": "evaluation/hpc_results/scored/full_scored_mistral_7b_instruct_v03_jsonforced.jsonl",
    "mistral_base": "evaluation/hpc_results/scored/full_scored_mistral_7b_base_v03_jsonforced.jsonl",
    "llama": "evaluation/hpc_results/scored/full_scored_llama31_8b_instruct_jsonforced.jsonl",
    "gemma": "evaluation/hpc_results/scored/full_scored_gemma2_9b_it_jsonforced.jsonl",
}

BENCHMARK_PATH = "data/prompts/prompt_instances.jsonl"

OUTPUT_DIR = Path("evaluation/analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_all_scored():
    all_rows = []

    for model, path in SCORED_FILES.items():
        rows = load_jsonl(path)
        for row in rows:
            row["model"] = model
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def load_benchmark_prompts():
    rows = load_jsonl(BENCHMARK_PATH)
    return {
        row["prompt_id"]: row.get("prompt_text", "")
        for row in rows
    }


def get_gold_value(row):
    gold = row.get("gold_output")
    task = row.get("task_name")

    if not isinstance(gold, dict):
        return None

    if task == "extractive_qa":
        return gold.get("answer")
    if task == "single_label_classification":
        return gold.get("label")
    if task == "multi_label_classification":
        return gold.get("labels")
    if task == "information_extraction":
        return gold
    if task == "rule_based_transformation":
        return gold.get("text")

    return gold


def get_pred_value(row):
    pred = row.get("parsed_output")
    task = row.get("task_name")

    if not isinstance(pred, dict):
        return None

    if task == "extractive_qa":
        return pred.get("answer")
    if task == "single_label_classification":
        return pred.get("label")
    if task == "multi_label_classification":
        return pred.get("labels")
    if task == "information_extraction":
        return pred
    if task == "rule_based_transformation":
        return pred.get("text")

    return pred


def suggest_failure_mode(row):
    task = row.get("task_name")
    distraction = row.get("distraction_type")
    parse_success = bool(row.get("parse_success"))
    parse_error = row.get("parse_error_code")
    score_error = row.get("score_error_code")

    gold_value = get_gold_value(row)
    pred_value = get_pred_value(row)

    raw_output = str(row.get("raw_output", "") or "")

    # 1. Format/schema failures
    if not parse_success:
        if parse_error == "invalid_json":
            return "invalid_json_or_extra_text"
        if parse_error == "invalid_label":
            return "invalid_label"
        if parse_error in {"missing_key", "extra_key", "wrong_schema", "wrong_type"}:
            return "schema_violation"
        return f"parse_failure_{parse_error}"

    # 2. Extractive QA specific failures
    if task == "extractive_qa" and score_error == "wrong_answer":
        if pred_value == "":
            return "empty_answer"
        if isinstance(gold_value, str) and isinstance(pred_value, str):
            if gold_value in pred_value and gold_value != pred_value:
                return "span_inflation"
            if gold_value.lower() == pred_value.lower() and gold_value != pred_value:
                return "case_or_surface_mismatch"
        return "wrong_span"

    # 3. Multi-label failures
    if task == "multi_label_classification":
        if score_error == "wrong_answer":
            return "wrong_label_set"
        return "multi_label_failure"

    # 4. Rule transformation failures
    if task == "rule_based_transformation":
        if score_error == "wrong_answer":
            return "transformation_mismatch"
        return "transformation_failure"

    # 5. Information extraction failures
    if task == "information_extraction":
        if score_error == "wrong_answer":
            return "entity_or_field_mismatch"
        return "information_extraction_failure"

    # 6. Single label failures
    if task == "single_label_classification":
        if score_error == "wrong_answer":
            return "wrong_single_label"
        return "single_label_failure"

    # 7. Distraction-specific fallback
    if distraction == "conflicting_instruction":
        return "possible_instruction_override"
    if distraction == "style_distraction":
        return "possible_style_interference"
    if distraction == "length_stress":
        return "possible_context_overload"

    return score_error or "unknown_failure"


def truncate_text(text, max_chars=600):
    text = "" if text is None else str(text)
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"


def build_failure_summary(df):
    failures = df[df["is_correct"] == False].copy()

    summary = (
        failures
        .groupby(["model", "task_name", "distraction_type", "suggested_failure_mode"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )

    return summary


def sample_failures(df, prompts_by_id, per_group=3, random_state=42):
    failures = df[df["is_correct"] == False].copy()

    # Avoid letting Mistral base dominate because it fails almost everywhere.
    group_cols = ["model", "task_name", "distraction_type"]

    samples = (
        failures
        .groupby(group_cols, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), per_group), random_state=random_state))
        .reset_index(drop=True)
    )

    samples["prompt_text"] = samples["prompt_id"].map(prompts_by_id)

    # Human-friendly columns
    samples["gold_value"] = samples.apply(get_gold_value, axis=1)
    samples["pred_value"] = samples.apply(get_pred_value, axis=1)

    samples["prompt_text_short"] = samples["prompt_text"].apply(lambda x: truncate_text(x, 700))
    samples["raw_output_short"] = samples["raw_output"].apply(lambda x: truncate_text(x, 300))
    samples["gold_short"] = samples["gold_value"].apply(lambda x: truncate_text(x, 300))
    samples["pred_short"] = samples["pred_value"].apply(lambda x: truncate_text(x, 300))

    samples["manual_failure_label"] = ""
    samples["manual_notes"] = ""

    keep_cols = [
        "model",
        "prompt_id",
        "base_example_id",
        "task_name",
        "regime",
        "distraction_type",
        "distraction_subtype",
        "prompt_surface_type",
        "parse_success",
        "parse_error_code",
        "score_error_code",
        "suggested_failure_mode",
        "gold_short",
        "pred_short",
        "raw_output_short",
        "prompt_text_short",
        "manual_failure_label",
        "manual_notes",
    ]

    keep_cols = [c for c in keep_cols if c in samples.columns]
    return samples[keep_cols], samples


def main():
    df = load_all_scored()
    prompts_by_id = load_benchmark_prompts()

    df["suggested_failure_mode"] = df.apply(suggest_failure_mode, axis=1)
    df["parse_fail"] = (~df["parse_success"].astype(bool)).astype(int)
    df["correct"] = df["is_correct"].astype(int)

    # Overall failure summary by model
    failure_overview = (
        df.groupby("model")
        .agg(
            n=("prompt_id", "count"),
            num_correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            num_failures=("correct", lambda x: int((x == 0).sum())),
            parse_failure_rate=("parse_fail", "mean"),
        )
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    failure_overview.to_csv(OUTPUT_DIR / "failure_overview.csv", index=False)

    # Failure modes by model/task/distraction
    failure_summary = build_failure_summary(df)
    failure_summary.to_csv(OUTPUT_DIR / "failure_mode_summary.csv", index=False)

    # Representative samples
    sample_short, sample_full = sample_failures(df, prompts_by_id, per_group=3)

    sample_short.to_csv(OUTPUT_DIR / "failure_samples.csv", index=False)

    with open(OUTPUT_DIR / "failure_samples.jsonl", "w", encoding="utf-8") as f:
        for row in sample_full.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Extra targeted summaries
    task_summary = (
        df[df["is_correct"] == False]
        .groupby(["model", "task_name", "suggested_failure_mode"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "task_name", "count"], ascending=[True, True, False])
    )
    task_summary.to_csv(OUTPUT_DIR / "failure_modes_by_task.csv", index=False)

    distraction_summary = (
        df[df["is_correct"] == False]
        .groupby(["model", "distraction_type", "suggested_failure_mode"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "distraction_type", "count"], ascending=[True, True, False])
    )
    distraction_summary.to_csv(OUTPUT_DIR / "failure_modes_by_distraction.csv", index=False)

    print("Saved:")
    print(f"- {OUTPUT_DIR / 'failure_overview.csv'}")
    print(f"- {OUTPUT_DIR / 'failure_mode_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'failure_samples.csv'}")
    print(f"- {OUTPUT_DIR / 'failure_samples.jsonl'}")
    print(f"- {OUTPUT_DIR / 'failure_modes_by_task.csv'}")
    print(f"- {OUTPUT_DIR / 'failure_modes_by_distraction.csv'}")


if __name__ == "__main__":
    main()