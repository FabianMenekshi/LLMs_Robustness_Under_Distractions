import json
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("hpc_results/scored")

def load_model_results(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df

def compute_metrics(df):
    df["correct"] = df["is_correct"].astype(int)
    df["parse_fail"] = (~df["parse_success"]).astype(int)

    return df

def aggregate_all(models):
    all_rows = []

    for model_name, path in models.items():
        df = load_model_results(path)
        df = compute_metrics(df)
        df["model"] = model_name
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)

def main():

    models = {
        "qwen7b": "hpc_results/scored/full_scored_qwen25_7b_instruct_jsonforced.jsonl",
        "qwen14b": "hpc_results/scored/full_scored_qwen25_14b_instruct_jsonforced.jsonl",
        "mistral_instruct": "hpc_results/scored/full_scored_mistral_7b_instruct_v03_jsonforced.jsonl",
        "mistral_base": "hpc_results/scored/full_scored_mistral_7b_base_v03_jsonforced.jsonl",
        "llama": "hpc_results/scored/full_scored_llama31_8b_instruct_jsonforced.jsonl",
        "gemma": "hpc_results/scored/full_scored_gemma2_9b_it_jsonforced.jsonl",
    }

    df = aggregate_all(models)

    # -------- OVERALL --------
    overall = df.groupby("model").agg(
        accuracy=("correct", "mean"),
        parse_failure_rate=("parse_fail", "mean"),
        n=("correct", "count")
    ).reset_index()

    overall.to_csv("aggregate_results/overall_metrics.csv", index=False)

    # -------- BY TASK --------
    by_task = df.groupby(["model", "task_name"]).agg(
        accuracy=("correct", "mean"),
        parse_failure_rate=("parse_fail", "mean"),
        n=("correct", "count")
    ).reset_index()

    by_task.to_csv("aggregate_results/metrics_by_task.csv", index=False)

    # -------- BY REGIME --------
    by_regime = df.groupby(["model", "regime"]).agg(
        accuracy=("correct", "mean"),
        parse_failure_rate=("parse_fail", "mean"),
        n=("correct", "count")
    ).reset_index()

    by_regime.to_csv("aggregate_results/metrics_by_regime.csv", index=False)

    # -------- BY DISTRACTION --------
    by_distraction = df.groupby(["model", "distraction_type"]).agg(
        accuracy=("correct", "mean"),
        parse_failure_rate=("parse_fail", "mean"),
        n=("correct", "count")
    ).reset_index()

    by_distraction.to_csv("aggregate_results/metrics_by_distraction_type.csv", index=False)

    # -------- MODEL × TASK × REGIME --------
    deep = df.groupby(["model", "task_name", "regime"]).agg(
        accuracy=("correct", "mean"),
        parse_failure_rate=("parse_fail", "mean"),
        n=("correct", "count")
    ).reset_index()

    deep.to_csv("aggregate_results/metrics_by_model_task_regime.csv", index=False)

    # -------- CLEAN VS DISTRACTED --------
    clean_vs = df.groupby(["model", "is_clean"]).agg(
        accuracy=("correct", "mean")
    ).reset_index()

    clean_vs.to_csv("aggregate_results/metrics_clean_vs_distracted.csv", index=False)

    print("All metrics saved.")

if __name__ == "__main__":
    main()