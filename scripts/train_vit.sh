#!/bin/bash
# ============================================================
# SLURM submission script for training ViT
# Usage: sbatch scripts/train_vit.sh [SEED]
# Example: sbatch scripts/train_vit.sh 42
# ============================================================

#SBATCH --job-name=train_vit
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_vit_%j.log
#SBATCH --error=logs/train_vit_%j.log

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate chart-vit-xai

SEED=${1:-42}

python modeling/train_vit.py \
    --image_days 30 \
    --patch_size 32 \
    --num_layers 2 \
    --seed "$SEED" \
    --fraction 0.1 \
    --num_epochs 10 \
    --num_rounds 10 \
    --patience 10 \
    --wandb