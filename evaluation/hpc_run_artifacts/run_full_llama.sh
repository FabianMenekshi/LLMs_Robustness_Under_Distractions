#!/bin/bash
#SBATCH --job-name=full_llama31
#SBATCH --output=/home/3241043/slurm_logs/full_llama31_%j.out
#SBATCH --error=/home/3241043/slurm_logs/full_llama31_%j.err
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

echo "=== Starting FULL Llama inference ==="

python evaluation/run_inference.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --model-config evaluation/model_config.json \
  --model-key llama31_8b_instruct \
  --output evaluation/results/raw/full_predictions_llama31_8b_instruct_jsonforced.jsonl

echo "=== Inference complete, starting scoring ==="

python evaluation/evaluate_model.py \
  --benchmark data/prompts/prompt_instances.jsonl \
  --raw-predictions evaluation/results/raw/full_predictions_llama31_8b_instruct_jsonforced.jsonl \
  --output evaluation/results/scored/full_scored_llama31_8b_instruct_jsonforced.jsonl

echo "=== Scoring complete, building metrics ==="

python evaluation/pilot_metrics.py \
  --scored-results evaluation/results/scored/full_scored_llama31_8b_instruct_jsonforced.jsonl \
  --output-csv evaluation/results/full/full_metrics_llama31_8b_instruct_jsonforced.csv

echo "=== FULL Llama run complete ==="
