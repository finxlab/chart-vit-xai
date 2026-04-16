#!/bin/bash
# ============================================================
# SLURM submission script for ViT inference
# Usage: sbatch scripts/test_vit.sh
#
# Run for each image_days configuration (20, 25, 30) as needed.
# Uncomment the lines below to test different settings.
# ============================================================

#SBATCH --job-name=test_vit
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_vit_%j.log
#SBATCH --error=logs/test_vit_%j.log

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate chart-vit-xai

# ============================================================
# Main configuration (used in the paper)
# ============================================================
python modeling/test_vit.py \
    --patch_size 32 \
    --image_days 30 \
    --num_layers 2 \
    --exp_name "enc2_batch1024_rs2ratio0.1_lr0.0001_wd0.05"

# ============================================================
# Other configurations (optional)
# ============================================================
# B/32 — 20d, 25d
# python modeling/test_vit.py --patch_size 32 --image_days 20 --num_layers 2 --exp_name "enc2_batch1024_rs2ratio0.1_lr0.0001_wd0.05"
# python modeling/test_vit.py --patch_size 32 --image_days 25 --num_layers 2 --exp_name "enc2_batch1024_rs2ratio0.1_lr0.0001_wd0.05"

# B/32 — 30d with more encoder layers
# python modeling/test_vit.py --patch_size 32 --image_days 30 --num_layers 4 --exp_name "enc4_batch1024_rs2ratio0.1_lr0.0001_wd0.05"
# python modeling/test_vit.py --patch_size 32 --image_days 30 --num_layers 6 --exp_name "enc6_batch1024_rs2ratio0.1_lr0.0001_wd0.05"

# B/16 — 30d
# python modeling/test_vit.py --patch_size 16 --image_days 30 --num_layers 2 --exp_name "enc2_batch1024_rs2ratio0.1_lr0.0001_wd0.05"