#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=captioning
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate deeplearning

export RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
echo "Run: $RUN_NAME; MAX_IMAGES=${MAX_IMAGES:-0}; EPOCHS=${EPOCHS:-15}"
nvidia-smi
python -u train.py
