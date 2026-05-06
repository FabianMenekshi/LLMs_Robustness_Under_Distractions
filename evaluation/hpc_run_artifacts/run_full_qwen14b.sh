#!/bin/bash
#SBATCH --job-name=full_qwen14
#SBATCH --output=/home/3241043/slurm_logs/full_qwen14_%j.out
#SBATCH --error=/home/3241043/slurm_logs/full_qwen14_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --account=3241043
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1

mkdir -p /home/3241043/slurm_logs
cd /mnt/beegfsstudents/home/3241043/projects/LLMs_Robustness_Under_Distractions || exit 1

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate robustness_eval

python --version
nvidia-smi

mkdir -p evaluation/results/raw
mkdir -p evaluation/results/scored
mkdir -p evaluation/results/full

echo "=== Starting FULL Qwen2.5-14B inference ==="

python evaluation/run_inference.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --model-config evaluation/model_config.json \
  --model-key qwen25_14b_instruct \
  --output evaluation/results/raw/full_predictions_qwen25_14b_instruct_jsonforced.jsonl

echo "=== Inference complete, starting scoring ==="

python evaluation/evaluate_model.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --raw-predictions evaluation/results/raw/full_predictions_qwen25_14b_instruct_jsonforced.jsonl \
  --output evaluation/results/scored/full_scored_qwen25_14b_instruct_jsonforced.jsonl

echo "=== Scoring complete, building metrics ==="

python evaluation/pilot_metrics.py \
  --scored-results evaluation/results/scored/full_scored_qwen25_14b_instruct_jsonforced.jsonl \
  --output-csv evaluation/results/full/full_metrics_qwen25_14b_instruct_jsonforced.csv

echo "=== FULL Qwen2.5-14B run complete ==="
