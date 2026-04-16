#!/bin/bash
# ============================================================
# SLURM submission script for CNN baseline inference
# Usage: sbatch scripts/test_cnn.sh
# ============================================================

#SBATCH --job-name=test_cnn
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_cnn_%j.log
#SBATCH --error=logs/test_cnn_%j.log

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate chart-vit-xai

python modeling/test_cnn.py