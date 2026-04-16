#!/bin/bash
# ============================================================
# SLURM submission script for training CNN baseline (xiu_20)
# Usage: sbatch scripts/train_cnn.sh [SEED]
# Example: sbatch scripts/train_cnn.sh 42
# ============================================================
 
#SBATCH --job-name=train_cnn
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_cnn_%j.log
#SBATCH --error=logs/train_cnn_%j.log
 
# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate chart-vit-xai
 
SEED=${1:-42}
 
echo "Running with seed=$SEED"
 
python modeling/train_cnn.py --seed "$SEED"