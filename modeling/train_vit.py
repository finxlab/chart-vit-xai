"""
ViT-B/16 Training for Stock Chart Images (224x224x3)

Key settings (following ViT/DeiT best practices):
  - Initialization: torchvision default (trunc_normal)
  - Warmup: Linear warmup for N steps 
  - Schedule: Warmup + Cosine annealing
  - Weight decay: 0.05
  - AdamW: betas=(0.9, 0.999)
  - Mixed precision: Yes

Supports:
  - RS2 training (fraction < 1.0) or standard training (fraction = 1.0)
  - Full checkpoint for resume
  - WandB logging
"""

import os
import gc
import h5py
import random
import argparse
import numpy as np
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast

from torchvision.models import vit_b_32, vit_b_16

import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ============================================================
# Dataset
# ============================================================

class HDF5Dataset(Dataset):
    """HDF5 Dataset with lazy loading for multiprocessing."""
    
    def __init__(self, hdf5_path, normalize=True):
        self.hdf5_path = hdf5_path
        self.h5_file = None
        self.normalize = normalize
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self._len = len(f['labels'])
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
        
        image = self.h5_file['images'][idx]
        label_raw = self.h5_file['labels'][idx]
        label = 1 if label_raw > 0 else 0
        
        if self.normalize:
            image_tensor = torch.from_numpy(image).float() / 255.0
        else:
            image_tensor = torch.from_numpy(image).float()
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


# ============================================================
# Model
# ============================================================

def build_vit(patch_size=32, num_encoder_layers=2, num_classes=2):
    """
    Build ViT-B/32 or ViT-B/16 with reduced encoder layers.
    Uses torchvision default initialization (trunc_normal).
 
    Both share the same hidden dim (768), heads (12), MLP dim (3072).
    Only patch size differs: 32 → 49 tokens, 16 → 196 tokens.
    """
    if patch_size == 32:
        model = vit_b_32(weights=None)
    elif patch_size == 16:
        model = vit_b_16(weights=None)
    else:
        raise ValueError(f"Unsupported patch_size: {patch_size}. Choose 16 or 32.")
 
    # Reduce encoder layers
    model.encoder.layers = nn.Sequential(
        *[model.encoder.layers[i] for i in range(num_encoder_layers)]
    )
 
    # Replace classification head
    in_features = model.heads.head.in_features  # 768
    model.heads.head = nn.Linear(in_features, num_classes)
 
    # Initialize only the new head (rest keeps torchvision default)
    nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
    nn.init.zeros_(model.heads.head.bias)
 
    return model

# ============================================================
# Learning Rate Schedule: Warmup + Cosine
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                     min_lr_ratio=0.01):
    """
    Linear warmup for warmup_steps, then cosine decay to min_lr.
    
    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
    num_warmup_steps : int
        Number of warmup steps (not epochs!)
    num_training_steps : int
        Total number of training steps
    min_lr_ratio : float
        Minimum LR as fraction of initial LR (default: 0.01 = 1%)
    """
    
    def lr_lambda(current_step):
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)

# ============================================================
# Seed & Reproducibility
# ============================================================

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_states():
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }

def set_rng_states(states):
    torch.set_rng_state(states['torch'])
    if states['cuda'] is not None:
        torch.cuda.set_rng_state_all(states['cuda'])
    np.random.set_state(states['numpy'])
    random.setstate(states['python'])


def save_checkpoint(path, epoch, round_idx, global_step, model, optimizer, scheduler,
                    scaler, best_loss, patience_counter, history):
    torch.save({
        'epoch': epoch,
        'round': round_idx,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': best_loss,
        'patience_counter': patience_counter, 
        'history': history,
        'rng_states': get_rng_states(),
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    set_rng_states(ckpt['rng_states'])
    return ckpt

# ============================================================
# Training Functions
# ============================================================

def train_one_round(model, loader, criterion, optimizer, scheduler, scaler, device):
    """Train for one round, stepping scheduler after each batch."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler after each batch (for step-based warmup)
        scheduler.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Valid", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return total_loss / total, correct / total

# ============================================================
# Main
# ============================================================

def main(args):

    # --- Setup ---
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    # --- Paths ---
    hdf5_path = f'DB/train/rgb_{args.image_days}d_train.h5'
    split_path = f'DB/train/split_indices/rgb_{args.image_days}d/seed_{args.seed}.npz'
    
    result_dir = (f'experiments/ViT_B{args.patch_size}_{args.image_days}d/'
                  f'enc{args.num_layers}_batch{args.batch_size}_rs2ratio{args.fraction}_lr{args.lr}_wd{args.weight_decay}/'
                  f'seed{args.seed}')

    os.makedirs(result_dir, exist_ok=True)

    # --- Data ---
    print(f"Loading data from {hdf5_path}")
    full_dataset = HDF5Dataset(hdf5_path)
    
    split_data = np.load(split_path)
    train_indices = split_data['train_indices']
    valid_indices = split_data['valid_indices']
    
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)
    
    print(f"Train: {len(train_dataset):,} | Valid: {len(valid_dataset):,}")
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # --- Calculate training steps ---
    train_indices_array = np.array(train_indices)
    
    if args.fraction < 1.0:
        # RS2 mode
        samples_per_round = int((len(train_indices_array) // args.batch_size) * args.fraction) * args.batch_size
        steps_per_round = samples_per_round // args.batch_size
        total_rounds = args.num_epochs * args.num_rounds
        total_steps = steps_per_round * total_rounds
    else:
        # Standard mode (fraction = 1.0)
        steps_per_epoch = len(train_indices_array) // args.batch_size
        total_rounds = args.num_epochs  # Each epoch = 1 round
        total_steps = steps_per_epoch * args.num_epochs
        samples_per_round = len(train_indices_array)
        steps_per_round = steps_per_epoch
        args.num_rounds = 1  # Override
    
    print(f"\nTraining Configuration:")
    print(f"  Mode: {'RS2' if args.fraction < 1.0 else 'Standard'}")
    print(f"  Samples per round: {samples_per_round:,}")
    print(f"  Steps per round: {steps_per_round:,}")
    print(f"  Total rounds: {total_rounds}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {args.warmup_steps}")

    # --- Model ---
    print(f"\nBuilding ViT-B/{args.patch_size} with {args.num_layers} encoder layers")
    model = build_vit(patch_size=args.patch_size, num_encoder_layers=args.num_layers, num_classes=2)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # --- Training setup ---
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.01
    )
    
    scaler = GradScaler()

    # --- Resume ---
    start_epoch = 0
    start_round = 0
    global_step = 0
    best_loss = float('inf')
    patience_counter = 0 
    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'lr': []}
    
    ckpt_path = os.path.join(result_dir, 'checkpoint.pth')

    if args.resume and os.path.exists(ckpt_path):
        print(f"\nResuming from {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
        start_epoch = ckpt['epoch']
        start_round = ckpt['round'] + 1
        global_step = ckpt['global_step']
        best_loss = ckpt['best_loss']
        history = ckpt['history']
        patience_counter = ckpt.get('patience_counter', 0)
        
        if start_round >= args.num_rounds:
            start_epoch += 1
            start_round = 0
        
        print(f"  Epoch: {start_epoch}, Round: {start_round}, Step: {global_step}")
        print(f"  Best loss: {best_loss:.6f}")

    # --- WandB ---
    if args.wandb:
        wandb.init(
            project='chart-vit-xai', 
            name=f'{args.image_days}d_patch{args.patch_size}_seed{args.seed}',
            config=vars(args),
            resume='allow'
        )

    # --- Training Loop ---
    print(f"\n{'='*60}")
    print("  Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        np.random.shuffle(train_indices_array)
        
        round_start = start_round if epoch == start_epoch else 0
        
        for round_idx in range(round_start, args.num_rounds):
            # Get subset
            if args.fraction < 1.0:
                # RS2 mode
                one_block = int((len(train_indices_array) // args.batch_size) * args.fraction) * args.batch_size
                if round_idx == args.num_rounds - 1:
                    subset_indices = train_indices_array[one_block * round_idx:]
                else:
                    subset_indices = train_indices_array[one_block * round_idx : one_block * (round_idx + 1)]
            else:
                # Standard mode
                subset_indices = train_indices_array
            
            subset = Subset(full_dataset, subset_indices)
            train_loader = DataLoader(
                subset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, generator=g
            )
            
            # Train
            train_loss, train_acc = train_one_round(
                model, train_loader, criterion, optimizer, scheduler, scaler, device
            )
            
            # Validate
            valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
            
            # Update step count
            global_step += len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_acc'].append(train_acc)
            history['valid_acc'].append(valid_acc)
            history['lr'].append(current_lr)
            
            if args.fraction < 1.0:
                print(f"Epoch {epoch+1}/{args.num_epochs} Round {round_idx+1}/{args.num_rounds} | "
                      f"Step {global_step:,} | "
                      f"Train: {train_loss:.6f} ({train_acc:.4f}) | "
                      f"Valid: {valid_loss:.6f} ({valid_acc:.4f}) | "
                      f"LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch+1}/{args.num_epochs} | "
                      f"Step {global_step:,} | "
                      f"Train: {train_loss:.6f} ({train_acc:.4f}) | "
                      f"Valid: {valid_loss:.6f} ({valid_acc:.4f}) | "
                      f"LR: {current_lr:.2e}")
            
            if args.wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'round': round_idx + 1,
                    'global_step': global_step,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                    'lr': current_lr,
                })
            
            # Save
            np.savetxt(f'{result_dir}/train_losses.txt', history['train_loss'])
            np.savetxt(f'{result_dir}/valid_losses.txt', history['valid_loss'])
            np.savetxt(f'{result_dir}/train_accs.txt', history['train_acc'])
            np.savetxt(f'{result_dir}/valid_accs.txt', history['valid_acc'])
            np.savetxt(f'{result_dir}/learning_rates.txt', history['lr'])
            
            save_checkpoint(
                ckpt_path, epoch, round_idx, global_step,
                model, optimizer, scheduler, scaler, best_loss, patience_counter, history
            )
        
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0  # Reset
                torch.save(model.state_dict(), f'{result_dir}/best_model.pth')
                print(f"  → New best model saved!")
            else:
                patience_counter += 1

            # Early stopping check
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            gc.collect()
            torch.cuda.empty_cache()
    
        if args.patience > 0 and patience_counter >= args.patience:
            break

    if args.wandb:
        wandb.finish()
    
    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"{'='*60}")
    print(f"  Total steps: {global_step:,}")
    print(f"  Best valid loss: {best_loss:.6f}")
    print(f"  Results saved to: {result_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT-B Training for Stock Charts (B/16 and B/32)')
    
    # Data
    parser.add_argument('--image_days', type=int, default=20, choices=[20, 25, 30],
                        help='Number of trading days in chart')
    
    parser.add_argument('--seed', type=int, default=42)
    # Model
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer encoder layers')
    parser.add_argument('--patch_size', type=int, default=32, choices=[16, 32],
                    help='Patch size for ViT')

    # Training mode
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='Rounds per epoch (RS2). Use 1 for standard training.')
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='Fraction of data per round. Use 1.0 for standard training.')
    
    # Optimization
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='AdamW weight decay')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing (0.0 = off)')
    parser.add_argument('--patience', type=int, default=10,
                    help='Early stopping patience (0 = disabled)')
    
    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"  ViT-B/{args.patch_size} Training Configuration")
    print("="*60)

    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")
    
    main(args)