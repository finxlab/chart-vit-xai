import os
import h5py
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torchvision.models import vit_b_32, vit_b_16
from tqdm import tqdm

SEEDS = [14, 51, 60, 71, 92]

# ---------------------------------------------------------
# 1. Model Architecture (MUST match train_vit.py exactly)
# ---------------------------------------------------------
 
def build_vit(patch_size=32, num_encoder_layers=2, num_classes=2):
    """
    Build ViT-B/32 or ViT-B/16 with reduced encoder layers.
    Identical to train_vit.py — do NOT modify independently.
 
    Args:
        patch_size (int): 32 → vit_b_32, 16 → vit_b_16
        num_encoder_layers (int): number of transformer encoder blocks to keep
        num_classes (int): output classes (default 2: up/down)
    """
    if patch_size == 32:
        model = vit_b_32(weights=None)
    elif patch_size == 16:
        model = vit_b_16(weights=None)
    else:
        raise ValueError(f"Unsupported patch_size: {patch_size}. Choose 16 or 32.")
 
    # Trim encoder layers
    model.encoder.layers = nn.Sequential(
        *[model.encoder.layers[i] for i in range(num_encoder_layers)]
    )
 
    # Replace classification head
    in_features = model.heads.head.in_features  # 768 for both B/16 and B/32
    model.heads.head = nn.Linear(in_features, num_classes)
 
    nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
    nn.init.zeros_(model.heads.head.bias)
 
    return model


# ---------------------------------------------------------
# 2. Test Dataset
# ---------------------------------------------------------
 
class TestHDF5Dataset(Dataset):
    """Test HDF5 Dataset — no labels, returns image + date + permno."""
 
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5_file = None
        with h5py.File(self.hdf5_path, 'r') as f:
            self._len = len(f['images'])
 
    def __len__(self):
        return self._len
 
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
 
        image_np = self.h5_file['images'][idx]
        date = self.h5_file['dates'][idx]
        permno = self.h5_file['permnos'][idx]
 
        # Normalize to match training (HDF5Dataset does / 255.0)
        image_tensor = torch.from_numpy(image_np).float() / 255.0
 
        return image_tensor, int(date), int(permno)


# ---------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------
 
def get_predictions(seed, model_dir, dataloader, device, patch_size=32, num_layers=2):
    """Load trained model and run inference."""
 
    model = build_vit(patch_size=patch_size, num_encoder_layers=num_layers, num_classes=2)
 
    model_path = os.path.join(model_dir, f'seed{seed}', 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
 
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
 
    probs = []
 
    print(f">> Inference for Seed {seed}...")
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc=f"Seed {seed}", leave=False):
            images = images.to(device, non_blocking=True)
 
            with autocast(device_type='cuda'):
                outputs = model(images)
 
            # Softmax → probability of class 1 (up)
            batch_probs = torch.softmax(outputs.float(), dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())
 
    return probs

# ---------------------------------------------------------
# 4. Main
# ---------------------------------------------------------

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description='ViT-B Test Inference (B/16 and B/32)')
    parser.add_argument('--patch_size', type=int, default=32, choices=[16, 32],
                        help='ViT patch size: 32 → vit_b_32, 16 → vit_b_16')
    parser.add_argument('--image_days', type=int, default=30, choices=[20, 25, 30])
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of encoder layers (e.g., 2, 4, 6)')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment subfolder name (e.g., enc2_batch1024_rs2ratio0.1_lr0.0001_wd0.05)')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
 
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
 
    # --- Paths ---
    test_h5    = f'DB/test/rgb_{args.image_days}d_test.h5'
    model_dir  = f'experiments/ViT_B{args.patch_size}_{args.image_days}d/{args.exp_name}'
    output_csv = f'result/b{args.patch_size}_{args.image_days}d_enc{args.num_layers}_prediction.csv'



    os.makedirs('result', exist_ok=True)
 
    print(f"\n{'='*60}")
    print(f"  ViT-B/{args.patch_size} Test Inference | {args.image_days}d | enc{args.num_layers}")
    print(f"{'='*60}")
    print(f"  Test HDF5 : {test_h5}")
    print(f"  Model dir : {model_dir}")
    print(f"  Output    : {output_csv}")
    print(f"  Device    : {device}")
 
    # --- Dataset ---
    test_dataset = TestHDF5Dataset(test_h5)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
 
    print(f"  Test samples: {len(test_dataset):,}")
 
    # --- Load dates & permnos ---
    with h5py.File(test_h5, 'r') as f:
        all_dates   = f['dates'][:]
        all_permnos = f['permnos'][:]
 
    df = pd.DataFrame({
        'date':   all_dates,
        'permno': all_permnos
    })
 
    # --- Run inference for each seed ---
    for seed in SEEDS:
        try:
            seed_probs = get_predictions(
                seed, model_dir, test_loader, device,
                patch_size=args.patch_size, num_layers=args.num_layers
            )
            df[f'prob_{seed}'] = seed_probs
            print(f"  Seed {seed}: done ({len(seed_probs):,} predictions)")
        except Exception as e:
            print(f"  [SKIP] Seed {seed}: {e}")
 
    df.to_csv(output_csv, index=False)
    print(f"\nSaved → {output_csv}")
    print("Done!")