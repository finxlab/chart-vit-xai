"""
Test HDF5 Generation — Grayscale (CNN baseline)
================================================
Adapted from legacy 4_test_gen_step2_gray.ipynb.

Image spec (unchanged from legacy / SSRN CNN baseline):
  - 64 x 60 x 1 grayscale
  - Price: 51px (scaled 0-50), Volume: 13px (12 + 1 blank row)
  - 20 trading days, 3px per day (1px O, 1px HL wick, 1px C)
  - MA20: 1px cv2.line

Requires: rebalance_date.csv from Step 1 (with ghost dates removed)

Output:
    DB/test/gray_test.h5
      - images:  (N, 1, 64, 60)  uint8
      - permnos: (N,)             int32
      - dates:   (N,)             int32
"""

import os
import cv2
import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed

# ============================================================
# Configuration
# ============================================================

IMAGE_H = 64     # 51 (price) + 13 (volume)
IMAGE_W = 60
IMAGE_DAYS = 20
MA_WINDOW = 20

# ============================================================
# Scaling Functions
# ============================================================

def price_scaling(dataframe, size):
    if np.nanmin(dataframe) == np.nanmax(dataframe):
        return np.round((dataframe) / (np.nanmax(dataframe)) * size // 2, 0)
    else:
        return np.round(
            (dataframe - np.nanmin(dataframe)) /
            (np.nanmax(dataframe) - np.nanmin(dataframe)) * size, 0
        )

def vol_scaling(dataframe, size):
    dataframe = dataframe.copy()
    if dataframe.max().values == 0:
        dataframe['VOL'] = [0] * len(dataframe)
        return dataframe
    return (np.ceil((dataframe / dataframe.max()) / (1 / size))).astype(int)

# ============================================================
# Image Generation
# ============================================================

def generate_gray_data_from_pickle(pickle_path, rebalance_dates_set):
    """Generate grayscale 64x60 chart images on rebalance dates only."""

    permno = int(pickle_path.split('/')[-1].split('.pkl')[0])

    stock = pd.read_pickle(pickle_path)     # 'DlyOpen','DlyHigh','DlyLow','DlyPrc','DlyVol','DlyRet','TradingStatusFlg' order

    stock.columns = ['OPEN','HIGH','LOW','CLOSE','VOL','RET','TRADESTATUSFLAG']
    stock = stock[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'RET','TRADESTATUSFLAG']].copy()

    stock['CLOSE'] = np.abs(stock['CLOSE'])
    stock['OPEN'] = np.abs(stock['OPEN'])
    stock['HIGH'] = np.abs(stock['HIGH'])
    stock['LOW'] = np.abs(stock['LOW'])

    stock['ADJ_CLOSE'] = (stock['RET'] + 1).cumprod() * stock['CLOSE'].iloc[0]
    stock['ADJ_OPEN'] = stock['ADJ_CLOSE'] / stock['CLOSE'] * stock['OPEN']
    stock['ADJ_HIGH'] = stock['ADJ_CLOSE'] / stock['CLOSE'] * stock['HIGH']
    stock['ADJ_LOW'] = stock['ADJ_CLOSE'] / stock['CLOSE'] * stock['LOW']

    stock["MA20"] = stock['ADJ_CLOSE'].rolling(window=MA_WINDOW, min_periods=1).mean()
    stock = stock.iloc[19:].copy()

    stock = stock[['ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'MA20', 'VOL','TRADESTATUSFLAG']].copy()
    stock['VOL'] = stock['VOL'].fillna(0)

    image_lst, date_lst, permno_lst = [], [], []

    for i in range(IMAGE_DAYS, len(stock) + 1):
        date = str(stock.index[i - 1]).split(' ')[0]

        if date not in rebalance_dates_set:
            continue

        date_int = int(date.replace('-', ''))
        temp = stock.iloc[i - IMAGE_DAYS:i].copy()

        if len(temp) != IMAGE_DAYS:
            continue

        if (temp['TRADESTATUSFLAG'] == 'A').sum()!=IMAGE_DAYS:
            continue


        scaled_price = price_scaling(temp[['ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'MA20']], 50)
        scaled_vol = vol_scaling(temp[['VOL']], 12)

        # Volume matrix (12 + 1 blank = 13 rows)
        width = 60
        vol_height = 13
        vol_matrix = np.zeros((vol_height, width), dtype=np.uint8)
        for j in range(20):
            vol_matrix[0:scaled_vol['VOL'].iloc[j], 3 * j + 1] = 1
        vol_matrix = np.flipud(vol_matrix)

        # Price matrix (51 rows)
        idx_lst = np.array([3 * j + 1 for j in range(20)])
        price_height = 51
        price_matrix = np.zeros((price_height, width), dtype=np.uint8)

        o = 50 - scaled_price['ADJ_OPEN']
        h = 50 - scaled_price['ADJ_HIGH']
        l = 50 - scaled_price['ADJ_LOW']
        c = 50 - scaled_price['ADJ_CLOSE']

        price_matrix[np.array(o[~o.isna()].astype(int)), idx_lst[~o.isna()] - 1] = 1
        price_matrix[np.array(c[~c.isna()].astype(int)), idx_lst[~c.isna()] + 1] = 1

        not_na_h = h[~(h.isna() | l.isna())].astype(int).values
        not_na_l = l[~(h.isna() | l.isna())].astype(int).values
        not_na_idx = idx_lst[~(h.isna() | l.isna())]

        for j in range(len(not_na_idx)):
            price_matrix[not_na_h[j]:not_na_l[j] + 1, not_na_idx[j]] = 1

        ma = 50 - scaled_price['MA20'].astype(int).values
        for j in range(19):
            cv2.line(price_matrix, (idx_lst[j], ma[j]), (idx_lst[j + 1], ma[j + 1]), 1, 1)

        chart = np.vstack((price_matrix, vol_matrix)) * 255

        image_lst.append(chart)
        permno_lst.append(permno)
        date_lst.append(date_int)

    if len(image_lst) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.stack(image_lst), np.array(permno_lst), np.array(date_lst)

# ============================================================
# Per-Stock HDF5 Processing
# ============================================================

def process_single_stock_gray(pickle_path, output_dir, rebalance_dates_set):
    """Process one stock → per-stock HDF5 (parallel-safe)."""
    permno = int(pickle_path.split('/')[-1].split('.pkl')[0])
    out_path = os.path.join(output_dir, f'{permno}.h5')

    if os.path.exists(out_path):
        try:
            with h5py.File(out_path, 'r') as hf:
                n = hf['images'].shape[0]
            return (permno, n)
        except:
            os.remove(out_path)

    try:
        images, permnos, dates = generate_gray_data_from_pickle(pickle_path, rebalance_dates_set)

        if images.size == 0:
            return (permno, 0)

        n = len(permnos)
        images_chw = images[:, np.newaxis, :, :]  # (N,64,60) → (N,1,64,60)

        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset('images',  data=images_chw, dtype=np.uint8,
                              chunks=(min(128, n), 1, IMAGE_H, IMAGE_W),
                              compression='gzip')
            hf.create_dataset('permnos', data=permnos.astype(np.int32))
            hf.create_dataset('dates',   data=dates.astype(np.int32))

        return (permno, n)

    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return (permno, 0)


# ============================================================
# Build Pipeline
# ============================================================

def build_gray_test_hdf5(stock_dir, chunk_dir, final_h5, rebalance_dates_set,
                          n_jobs=-1, delete_chunks=False):

    print("=" * 60)
    print("  GRAY TEST: 64x60x1 (CNN baseline)")
    print("=" * 60)

    # --- Per-stock HDF5 (parallel) ---
    os.makedirs(chunk_dir, exist_ok=True)

    stock_list = sorted(glob(f'{stock_dir}/*.pkl'))
    stock_list = [Path(path).as_posix() for path in stock_list]
    print(f"\n  Found {len(stock_list)} stocks")

    existing = set(int(f.split('.h5')[0]) for f in os.listdir(chunk_dir) if f.endswith('.h5'))
    print(f"  Already completed: {len(existing)} stocks")

    print(f"\n  [Step 1] Per-stock HDF5 (n_jobs={n_jobs})...")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_stock_gray)(path, chunk_dir, rebalance_dates_set)
        for path in stock_list
    )

    total_samples = sum(n for _, n in results)
    success = sum(1 for _, n in results if n > 0)
    failed = sum(1 for _, n in results if n == 0)
    print(f"\n  Step 1 done: {success} success, {failed} failed/empty, {total_samples:,} samples")

    # --- Merge (sequential) ---
    h5_files = sorted(glob(f'{chunk_dir}/*.h5'))
    print(f"\n  [Step 2] Merging {len(h5_files)} files...")

    total_samples = 0
    valid_files = []
    for fpath in tqdm(h5_files, desc="  Counting"):
        try:
            with h5py.File(fpath, 'r') as hf:
                n = hf['images'].shape[0]
                if n > 0:
                    valid_files.append((fpath, n))
                    total_samples += n
        except Exception as e:
            print(f"    Skipping: {fpath} ({e})")

    print(f"  Valid: {len(valid_files)} files, Total: {total_samples:,} samples")

    with h5py.File(final_h5, 'w') as hf_out:
        img_ds    = hf_out.create_dataset('images',  shape=(total_samples, 1, IMAGE_H, IMAGE_W),
                                           dtype=np.uint8, chunks=(128, 1, IMAGE_H, IMAGE_W),
                                           compression='gzip', maxshape=(None, 1, IMAGE_H, IMAGE_W))
        permno_ds = hf_out.create_dataset('permnos', shape=(total_samples,), dtype=np.int32, maxshape=(None,))
        date_ds   = hf_out.create_dataset('dates',   shape=(total_samples,), dtype=np.int32, maxshape=(None,))

        current_idx = 0
        for fpath, n in tqdm(valid_files, desc="  Merging"):
            try:
                with h5py.File(fpath, 'r') as hf_in:
                    img_ds[current_idx:current_idx + n]    = hf_in['images'][:]
                    permno_ds[current_idx:current_idx + n] = hf_in['permnos'][:]
                    date_ds[current_idx:current_idx + n]   = hf_in['dates'][:]
                current_idx += n
            except Exception as e:
                print(f"\n  Failed: {fpath} ({e})")

        if current_idx < total_samples:
            print(f"\n  Trimming: expected {total_samples}, got {current_idx}")
            img_ds.resize(current_idx, axis=0)
            permno_ds.resize(current_idx, axis=0)
            date_ds.resize(current_idx, axis=0)

    print(f"\n  Done! Saved {current_idx:,} samples to {final_h5}")

    if delete_chunks:
        for fpath, _ in valid_files:
            os.remove(fpath)
        print("  Chunks deleted.")

    return current_idx


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    STOCK_DIR     = 'DB/test/stocks'
    CHUNK_DIR     = 'DB/test/chunks_h5/gray'
    FINAL_H5      = 'DB/test/gray_test.h5'
    REBALANCE_CSV = 'DB/rebalance_date.csv'

    # Load rebalance dates (from Step 1 with ghost dates removed)
    rebalance_df = pd.read_csv(REBALANCE_CSV, index_col=0)
    rebalance_dates_set = set(rebalance_df['0'].tolist())
    print(f"Loaded {len(rebalance_dates_set)} rebalance dates")

    # Build
    total = build_gray_test_hdf5(
        STOCK_DIR, CHUNK_DIR, FINAL_H5, rebalance_dates_set,
        n_jobs=-1, delete_chunks=False
    )

    # Verify
    with h5py.File(FINAL_H5, 'r') as hf:
        print(f"\nVerification:")
        print(f"  images:  {hf['images'].shape}, dtype={hf['images'].dtype}")
        print(f"  permnos: {hf['permnos'].shape}, dtype={hf['permnos'].dtype}")
        print(f"  dates:   {hf['dates'].shape}, dtype={hf['dates'].dtype}")