#!/bin/bash
#SBATCH --job-name=pilot_qwen25
#SBATCH --output=/home/3241043/slurm_logs/pilot_qwen25_%j.out
#SBATCH --error=/home/3241043/slurm_logs/pilot_qwen25_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
mkdir -p evaluation/results

echo "=== Starting inference ==="

python evaluation/run_inference.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --model-config evaluation/model_config.json \
  --model-key qwen25_7b_instruct \
  --output evaluation/results/raw/pilot_predictions_qwen25_7b_instruct_jsonforced.jsonl \
  --prompt-ids evaluation/pilot/pilot_prompt_ids.txt

echo "=== Inference complete, starting evaluation ==="

python evaluation/evaluate_model.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --raw-predictions evaluation/results/raw/pilot_predictions_qwen25_7b_instruct_jsonforced.jsonl \
  --output evaluation/results/scored/pilot_scored_qwen25_7b_instruct_jsonforced.jsonl

echo "=== Evaluation complete, building metrics ==="

python evaluation/pilot_metrics.py \
  --scored-results evaluation/results/scored/pilot_scored_qwen25_7b_instruct_jsonforced.jsonl \
  --output-csv evaluation/results/pilot_metrics_qwen25_7b_instruct_jsonforced.csv

echo "=== Pilot run complete ==="
