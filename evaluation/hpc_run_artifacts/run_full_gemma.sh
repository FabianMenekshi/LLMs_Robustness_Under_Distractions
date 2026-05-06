#!/bin/bash
#SBATCH --job-name=full_gemma2
#SBATCH --output=/home/3241043/slurm_logs/full_gemma2_%j.out
#SBATCH --error=/home/3241043/slurm_logs/full_gemma2_%j.err
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
mkdir -p evaluation/results/full

echo "=== Starting FULL Gemma inference ==="

python evaluation/run_inference.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --model-config evaluation/model_config.json \
  --model-key gemma2_9b_it \
  --output evaluation/results/raw/full_predictions_gemma2_9b_it_jsonforced.jsonl

echo "=== Inference complete, starting scoring ==="

python evaluation/evaluate_model.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --raw-predictions evaluation/results/raw/full_predictions_gemma2_9b_it_jsonforced.jsonl \
  --output evaluation/results/scored/full_scored_gemma2_9b_it_jsonforced.jsonl

echo "=== Scoring complete, building metrics ==="

python evaluation/pilot_metrics.py \
  --scored-results evaluation/results/scored/full_scored_gemma2_9b_it_jsonforced.jsonl \
  --output-csv evaluation/results/full/full_metrics_gemma2_9b_it_jsonforced.csv

echo "=== FULL Gemma run complete ==="
