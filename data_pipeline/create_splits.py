"""
Split Index Generation for Training (ViT RGB + CNN Gray)
=========================================================
Creates stratified train/validation splits for each seed.

Directory structure after running:
DB/train/
├── rgb_20d_train.h5
├── rgb_25d_train.h5
├── rgb_30d_train.h5
├── gray_train.h5
└── split_indices/
    ├── rgb_20d/
    │   ├── seed_14.npz
    │   └── ...
    ├── rgb_25d/
    ├── rgb_30d/
    └── gray/
        ├── seed_14.npz
        └── ...

Each .npz contains:
  - train_indices: array of indices for training
  - valid_indices: array of indices for validation

Usage:
    python create_splits.py
"""

import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

BASE_DIR = 'DB/train'
TEST_SIZE = 0.3

# Seeds are generated deterministically from np.random.seed(42)
np.random.seed(42)
SEEDS = sorted(np.random.randint(0, 100, size=5))

# Configurations to process:
#   ('config_name', 'hdf5_filename', 'output_subdir')
CONFIGS = [
    ('RGB 20d',  'rgb_20d_train.h5', 'rgb_20d'),
    ('RGB 25d',  'rgb_25d_train.h5', 'rgb_25d'),
    ('RGB 30d',  'rgb_30d_train.h5', 'rgb_30d'),
    ('Gray',     'gray_train.h5',    'gray'),
]

# ============================================================
# Functions
# ============================================================

def create_splits_for_config(hdf5_path, seeds, output_dir, test_size=0.3):
    """
    Create stratified train/validation splits for a single HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 file.
    seeds : list of int
        Random seeds for reproducibility.
    output_dir : str
        Directory to save .npz split files.
    test_size : float
        Fraction for validation (default 0.3 = 30% validation).

    Returns
    -------
    n_samples : int
        Number of samples in the HDF5 file.
    """

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading labels from: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as hf:
        labels = hf['labels'][:]
        n_samples = len(labels)

    print(f"  Total samples: {n_samples:,}")

    # Binary stratification: return > 0 or not
    binary_labels = (labels > 0).astype(int)
    pos_ratio = binary_labels.mean()
    print(f"  Positive ratio: {pos_ratio:.4f} "
          f"({binary_labels.sum():,} / {n_samples:,})")

    indices = np.arange(n_samples)

    for seed in tqdm(seeds, desc="  Generating splits"):

        train_idx, valid_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=binary_labels,
            random_state=seed,
        )

        save_path = os.path.join(output_dir, f'seed_{seed}.npz')
        np.savez_compressed(
            save_path,
            train_indices=train_idx,
            valid_indices=valid_idx,
        )

    print(f"  Saved {len(seeds)} splits to {output_dir}")
    print(f"  Train: {len(train_idx):,} | Valid: {len(valid_idx):,}")

    return n_samples


def create_all_splits(base_dir, seeds, configs, test_size=0.3):
    """
    Create splits for all configurations (RGB + Gray).

    Parameters
    ----------
    base_dir : str
        Base directory containing HDF5 files (e.g., 'DB/train').
    seeds : list of int
        Random seeds.
    configs : list of (name, hdf5_filename, output_subdir) tuples
        Configurations to process.
    test_size : float
        Validation fraction.

    Returns
    -------
    results : dict
        Mapping from config name to number of samples.
    """

    print("=" * 60)
    print("  SPLIT GENERATION FOR ALL CONFIGURATIONS")
    print("=" * 60)
    print(f"  Seeds: {seeds}")
    print(f"  Test size: {test_size}")
    print(f"  Configurations: {[c[0] for c in configs]}")

    results = {}

    for name, hdf5_name, subdir in configs:

        hdf5_path = os.path.join(base_dir, hdf5_name)
        output_dir = os.path.join(base_dir, 'split_indices', subdir)

        if not os.path.exists(hdf5_path):
            print(f"\n[SKIP] {name}: {hdf5_path} not found.")
            continue

        print(f"\n--- {name} ---")
        n_samples = create_splits_for_config(
            hdf5_path=hdf5_path,
            seeds=seeds,
            output_dir=output_dir,
            test_size=test_size,
        )

        results[name] = n_samples

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, n in results.items():
        print(f"  {name}: {n:,} samples")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    print(f"Seeds: {SEEDS}")

    results = create_all_splits(
        base_dir=BASE_DIR,
        seeds=SEEDS,
        configs=CONFIGS,
        test_size=TEST_SIZE,
    )