"""
Test HDF5 Generation (Step 2) — 224x224x3 RGB Chart Images
===========================================================

Directory structure after running:
DB/test/
├── stocks/                         ← from Step 1
├── chunks_h5/{IMAGE_DAYS}d/       ← per-stock H5 files (intermediate)
├── rgb_{IMAGE_DAYS}d_test.h5      ← final merged test dataset
rebalance_date.csv                  ← from Step 1

Each final HDF5 contains:
  - images:  (N, 3, 224, 224)  uint8
  - permnos: (N,)              int32
  - dates:   (N,)              int32
  (NO labels dataset — this is the test set)
"""

import os
import cv2
import h5py
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed

# ============================================================
# Configuration (MUST match train pipeline exactly)
# ============================================================

IMAGE_SIZE = 224

CANDLE_BODY_WIDTH = 6
WICK_WIDTH = 2

PRICE_HEIGHT = 160           # 5 patch rows (160 / 32 = 5)
GAP_HEIGHT = 1               # 1px blank between price and volume
VOL_HEIGHT = 63              # ceil-based volume scaling

assert PRICE_HEIGHT + GAP_HEIGHT + VOL_HEIGHT == IMAGE_SIZE, \
    f"Height mismatch: {PRICE_HEIGHT} + {GAP_HEIGHT} + {VOL_HEIGHT} != {IMAGE_SIZE}"

PRICE_SCALE = PRICE_HEIGHT - 1   # 0 to 159
VOL_SCALE = VOL_HEIGHT           # ceil-based, max 63

MA_WINDOW = 20

BLANK_MAP = {
    20: 5,
    25: 3,
    30: 1,
}

# ============================================================
# Layout + Scaling Functions (identical to train pipeline)
# ============================================================

def get_chart_layout(image_days):
    """
    Calculate chart layout given number of image days.
    Last candle has no trailing blank.
    """
    blank = BLANK_MAP[image_days]
    px_per_day = CANDLE_BODY_WIDTH + blank
    chart_width = CANDLE_BODY_WIDTH * image_days + blank * (image_days - 1)
    total_padding = IMAGE_SIZE - chart_width
    left_pad = total_padding // 2 + (total_padding % 2)
    right_pad = total_padding // 2
    return {
        'image_days': image_days,
        'blank': blank,
        'px_per_day': px_per_day,
        'chart_width': chart_width,
        'left_pad': left_pad,
        'right_pad': right_pad,
    }


def widen_ma_line(blue_matrix, price_height):
    """
    Manually widen a 1px MA line to 3px by expanding ±1 pixel vertically.
    """
    widened = blue_matrix.copy()
    ys, xs = np.where(blue_matrix > 0)
    for y, x in zip(ys, xs):
        if y > 0:
            widened[y - 1, x] = 1
        if y < price_height - 1:
            widened[y + 1, x] = 1
    return widened


def price_scaling(dataframe, size):
    """Scale price data to pixel range [0, size]."""
    if np.nanmin(dataframe) == np.nanmax(dataframe):
        return np.round((dataframe) / (np.nanmax(dataframe)) * size // 2, 0)
    else:
        return np.round(
            (dataframe - np.nanmin(dataframe)) /
            (np.nanmax(dataframe) - np.nanmin(dataframe)) * size, 0
        )


def vol_scaling(dataframe, size):
    """Scale volume data using ceiling division."""
    dataframe = dataframe.copy()
    if dataframe.max().values == 0:
        dataframe['VOL'] = [0] * len(dataframe)
        return dataframe
    scaled_vol = (np.ceil((dataframe / dataframe.max()) / (1 / size))).astype(int)
    return scaled_vol

# ============================================================
# Image Generation (TEST version — rebalance dates only, no labels)
# ============================================================

def generate_test_data_from_pickle(pickle_path, rebalance_dates_set, image_days=25):
    """
    Generate chart images for a single stock, ONLY on rebalance dates.
    
    Key difference from train:
    - Train: sliding window over every possible date, with forward return labels
    - Test:  only generates on rebalance dates, no labels
    
    Parameters:
    -----------
    pickle_path : str
        Path to per-stock pickle file
    rebalance_dates_set : set of str
        Set of rebalance date strings (e.g., {'2001-01-02', '2001-01-30', ...})
    image_days : int
        Number of trading days per image (20, 25, or 30)
    
    Returns:
    --------
    images_np  : np.ndarray  (N, 224, 224, 3) uint8
    permnos_np : np.ndarray  (N,) int
    dates_np   : np.ndarray  (N,) int  (YYYYMMDD format)
    """
    
    assert image_days in [20, 25, 30]

    layout = get_chart_layout(image_days)
    left_pad = layout['left_pad']
    px_per_day = layout['px_per_day']

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

    stock['CLOSE-OPEN'] = np.round(stock['ADJ_CLOSE'] - stock['ADJ_OPEN'], 8).replace(0, np.nan)

    conditions = [(stock['CLOSE-OPEN'] > 0), (stock['CLOSE-OPEN'] < 0)]
    choices = ['green', 'red']
    stock['COLOR'] = np.select(conditions, choices, default='nan')

    stock['lagged_ADJ_CLOSE'] = stock['ADJ_CLOSE'].shift(1)
    stock['ADJ_OPEN'] = stock['ADJ_OPEN'].fillna(stock['lagged_ADJ_CLOSE'])
    stock['ADJ_HIGH'] = stock['ADJ_HIGH'].fillna(stock[['ADJ_OPEN', 'ADJ_CLOSE']].max(axis=1))
    stock['ADJ_LOW'] = stock['ADJ_LOW'].fillna(stock[['ADJ_OPEN', 'ADJ_CLOSE']].min(axis=1))

    stock['PRICE_CHANGE'] = stock['ADJ_CLOSE'].diff().replace(0, np.nan)
    stock['PRICE_CHANGE'] = stock['PRICE_CHANGE'].ffill()
    stock['PRICE_CHANGE'] = stock['PRICE_CHANGE'].map(lambda x: 'green' if x > 0 else 'red')

    stock['COLOR'] = stock['COLOR'].replace('nan', np.nan).fillna(stock['PRICE_CHANGE'])
    stock["MA20"] = stock['ADJ_CLOSE'].rolling(window=MA_WINDOW, min_periods=1).mean()
    stock = stock.iloc[19:].copy()

    stock = stock[['ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'MA20', 'VOL', 'COLOR', 'PRICE_CHANGE','TRADESTATUSFLAG']].copy()
    stock['VOL'] = stock['VOL'].fillna(0)

    image_lst, date_lst, permno_lst = [], [], []

    for i in range(image_days, len(stock) + 1):
        date = str(stock.index[i - 1]).split(' ')[0]

        # ── TEST-SPECIFIC: only generate on rebalance dates ──
        if date not in rebalance_dates_set:
            continue

        date_int = int(date.replace('-', ''))
        temp = stock.iloc[i - image_days:i].copy()

        if len(temp) != image_days:
            continue

        if (temp['TRADESTATUSFLAG'] == 'A').sum()!=image_days:
            continue

        # ── Image generation (identical to train pipeline) ──
        scaled_price = price_scaling(
            temp[['ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'MA20']],
            PRICE_SCALE
        ).astype(int)

        scaled_vol = vol_scaling(
            temp[['VOL']],
            VOL_SCALE
        ).astype(int)

        color_lst = temp['COLOR'].reset_index(drop=True)
        green_idx = list(color_lst[color_lst == 'green'].index)
        red_idx = list(color_lst[color_lst == 'red'].index)

        o = scaled_price['ADJ_OPEN'].values
        h = scaled_price['ADJ_HIGH'].values
        l = scaled_price['ADJ_LOW'].values
        c = scaled_price['ADJ_CLOSE'].values
        ma = scaled_price['MA20'].values

        def candle_x(day_idx):
            x_start = left_pad + day_idx * px_per_day
            return x_start, x_start + CANDLE_BODY_WIDTH, x_start + 2, x_start + 4

        # RED (bearish)
        red_price = np.zeros((PRICE_HEIGHT, IMAGE_SIZE), dtype=np.uint8)
        for idx in red_idx:
            bx0, bx1, wx0, wx1 = candle_x(idx)
            red_price[c[idx]:o[idx] + 1, bx0:bx1] = 1
            red_price[l[idx]:h[idx] + 1, wx0:wx1] = 1
        red_price = np.flipud(red_price)

        red_vol = np.zeros((VOL_HEIGHT, IMAGE_SIZE), dtype=np.uint8)
        for idx in red_idx:
            bx0, bx1, _, _ = candle_x(idx)
            red_vol[0:scaled_vol['VOL'].iloc[idx], bx0:bx1] = 1
        red_vol = np.flipud(red_vol)

        # GREEN (bullish)
        green_price = np.zeros((PRICE_HEIGHT, IMAGE_SIZE), dtype=np.uint8)
        for idx in green_idx:
            bx0, bx1, wx0, wx1 = candle_x(idx)
            green_price[o[idx]:c[idx] + 1, bx0:bx1] = 1
            green_price[l[idx]:h[idx] + 1, wx0:wx1] = 1
        green_price = np.flipud(green_price)

        green_vol = np.zeros((VOL_HEIGHT, IMAGE_SIZE), dtype=np.uint8)
        for idx in green_idx:
            bx0, bx1, _, _ = candle_x(idx)
            green_vol[0:scaled_vol['VOL'].iloc[idx], bx0:bx1] = 1
        green_vol = np.flipud(green_vol)

        # BLUE (MA20: 1px → manual 3px)
        blue_price = np.zeros((PRICE_HEIGHT, IMAGE_SIZE), dtype=np.uint8)
        for j in range(image_days - 1):
            x1 = left_pad + j * px_per_day + CANDLE_BODY_WIDTH // 2
            x2 = left_pad + (j + 1) * px_per_day + CANDLE_BODY_WIDTH // 2
            cv2.line(blue_price, (x1, ma[j]), (x2, ma[j + 1]), 1, 1)
        blue_price = widen_ma_line(blue_price, PRICE_HEIGHT)
        blue_price = np.flipud(blue_price)

        blue_vol = np.zeros((VOL_HEIGHT, IMAGE_SIZE), dtype=np.uint8)

        # Combine
        rgb_price = np.dstack([red_price, green_price, blue_price]) * 255
        rgb_vol = np.dstack([red_vol, green_vol, blue_vol]) * 255

        gap_row = np.zeros((GAP_HEIGHT, IMAGE_SIZE, 3), dtype=np.uint8)
        chart = np.vstack([rgb_price, gap_row, rgb_vol])

        assert chart.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), \
            f"Chart shape mismatch: {chart.shape}"

        image_lst.append(chart.astype(np.uint8))
        permno_lst.append(permno)
        date_lst.append(date_int)

    if len(image_lst) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.stack(image_lst), np.array(permno_lst), np.array(date_lst)

# ============================================================
# Per-Stock HDF5 Processing (parallel-safe)
# ============================================================

def process_single_stock_test(pickle_path, output_dir, rebalance_dates_set, image_days=25):
    """
    Process one stock pickle → save as individual HDF5 file (test version).
    
    Each per-stock HDF5 contains:
      - images:  (N, 3, 224, 224)  uint8
      - permnos: (N,)              int32
      - dates:   (N,)              int32
    
    Returns:
      (permno, n_samples) on success, (permno, 0) on failure/empty
    """
    
    permno = int(pickle_path.split('/')[-1].split('.pkl')[0])
    out_path = os.path.join(output_dir, f'{permno}.h5')
    
    # Skip if already exists (resume-friendly)
    if os.path.exists(out_path):
        try:
            with h5py.File(out_path, 'r') as hf:
                n = hf['images'].shape[0]
            return (permno, n)
        except:
            os.remove(out_path)
    
    try:
        images, permnos, dates = generate_test_data_from_pickle(
            pickle_path, rebalance_dates_set, image_days
        )
        
        if images.size == 0:
            return (permno, 0)
        
        n = len(permnos)
        images_chw = np.transpose(images, (0, 3, 1, 2))  # (N,224,224,3) → (N,3,224,224)
        
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset('images',  data=images_chw, dtype=np.uint8,
                              chunks=(min(32, n), 3, IMAGE_SIZE, IMAGE_SIZE),
                              compression='gzip')
            hf.create_dataset('permnos', data=permnos.astype(np.int32))
            hf.create_dataset('dates',   data=dates.astype(np.int32))
        
        return (permno, n)
    
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return (permno, 0)
    
# ============================================================
# Build Pipeline Functions
# ============================================================

def build_per_stock_h5_test(stock_dir, output_dir, rebalance_dates_set, image_days=25, n_jobs=-1):
    """
    Step 1: Generate one HDF5 file per stock in parallel (test version).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stock_list = sorted(glob(f'{stock_dir}/*.pkl'))
    stock_list = [Path(path).as_posix() for path in stock_list]
    print(f"Found {len(stock_list)} stocks")
    
    existing = set(int(f.split('.h5')[0]) for f in os.listdir(output_dir) if f.endswith('.h5'))
    print(f"Already completed: {len(existing)} stocks")
    
    print(f"\n[Step 1] Generating per-stock HDF5 files (n_jobs={n_jobs})...")
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_stock_test)(path, output_dir, rebalance_dates_set, image_days)
        for path in stock_list
    )
    
    total_samples = sum(n for _, n in results)
    success = sum(1 for _, n in results if n > 0)
    failed = sum(1 for _, n in results if n == 0)
    
    print(f"\n{'='*50}")
    print(f"Step 1 Complete!")
    print(f"  Success: {success} stocks")
    print(f"  Failed/Empty: {failed} stocks")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*50}")
    
    return results


def merge_h5_files_test(chunk_dir, output_path, delete_chunks=False):
    """
    Step 2: Merge all per-stock HDF5 files into one final HDF5 (test version).
    
    Note: No 'labels' dataset — this is the test set.
    """
    h5_files = sorted(glob(f'{chunk_dir}/*.h5'))
    print(f"Found {len(h5_files)} per-stock HDF5 files to merge")
    
    if len(h5_files) == 0:
        print("No files to merge!")
        return 0
    
    # --- Pass 1: Count total samples ---
    print("\n[Pass 1] Counting total samples...")
    total_samples = 0
    valid_files = []
    
    for fpath in tqdm(h5_files, desc="Counting"):
        try:
            with h5py.File(fpath, 'r') as hf:
                n = hf['images'].shape[0]
                if n > 0:
                    valid_files.append((fpath, n))
                    total_samples += n
        except Exception as e:
            print(f"  Skipping corrupted file: {fpath} ({e})")
    
    print(f"Valid files: {len(valid_files)}, Total samples: {total_samples:,}")
    
    # --- Pass 2: Merge into single HDF5 ---
    print(f"\n[Pass 2] Merging into {output_path}...")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with h5py.File(output_path, 'w') as hf_out:
        img_ds    = hf_out.create_dataset(
            'images',
            shape=(total_samples, 3, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
            chunks=(32, 3, IMAGE_SIZE, IMAGE_SIZE),
            compression='gzip',
            maxshape=(None, 3, IMAGE_SIZE, IMAGE_SIZE),
        )
        permno_ds = hf_out.create_dataset('permnos', shape=(total_samples,), dtype=np.int32, maxshape=(None,))
        date_ds   = hf_out.create_dataset('dates',   shape=(total_samples,), dtype=np.int32, maxshape=(None,))
        
        current_idx = 0
        
        for fpath, n in tqdm(valid_files, desc="Merging"):
            try:
                with h5py.File(fpath, 'r') as hf_in:
                    img_ds[current_idx:current_idx + n]    = hf_in['images'][:]
                    permno_ds[current_idx:current_idx + n] = hf_in['permnos'][:]
                    date_ds[current_idx:current_idx + n]   = hf_in['dates'][:]
                
                current_idx += n
                
            except Exception as e:
                print(f"\nFailed to read: {fpath} ({e})")
        
        # Trim if some files failed during merge
        if current_idx < total_samples:
            print(f"\nTrimming: expected {total_samples}, got {current_idx}")
            img_ds.resize(current_idx, axis=0)
            permno_ds.resize(current_idx, axis=0)
            date_ds.resize(current_idx, axis=0)
    
    print(f"\nDone! Saved {current_idx:,} samples to {output_path}")
    
    if delete_chunks:
        print("Deleting per-stock HDF5 files...")
        for fpath, _ in valid_files:
            os.remove(fpath)
        print("Deleted.")
    
    return current_idx

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    
    # ============================================================
    # Configuration
    # ============================================================
    parser = argparse.ArgumentParser(description='Generate RGB chart images for test inference')
    parser.add_argument('--image_days', type=int, default=30, choices=[20, 25, 30],
                        help='Number of trading days per image (20, 25, or 30)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel workers (-1 = all cores)')
    args = parser.parse_args()
    
    IMAGE_DAYS = args.image_days
    STOCK_DIR  = 'DB/test/stocks'
    CHUNK_DIR  = f'DB/test/chunks_h5/{IMAGE_DAYS}d'
    FINAL_H5   = f'DB/test/rgb_{IMAGE_DAYS}d_test.h5'
    
    REBALANCE_CSV = 'DB/rebalance_date.csv'
    
    # ============================================================
    # Load Rebalance Dates
    # ============================================================
    rebalance_df = pd.read_csv(REBALANCE_CSV, index_col=0)
    rebalance_dates_set = set(rebalance_df['0'].tolist())
    print(f"Loaded {len(rebalance_dates_set)} rebalance dates")
    
    # ============================================================
    # Step 1: Per-stock HDF5 (parallel)
    # ============================================================
    results = build_per_stock_h5_test(
        STOCK_DIR, CHUNK_DIR, rebalance_dates_set, image_days=IMAGE_DAYS, n_jobs=args.n_jobs
    )
    
    # ============================================================
    # Step 2: Merge into one file
    # ============================================================
    total = merge_h5_files_test(CHUNK_DIR, FINAL_H5, delete_chunks=False)
    
    # ============================================================
    # Verification
    # ============================================================
    with h5py.File(FINAL_H5, 'r') as hf:
        print(f"\nVerification:")
        print(f"  images:  {hf['images'].shape}, dtype={hf['images'].dtype}")
        print(f"  permnos: {hf['permnos'].shape}, dtype={hf['permnos'].dtype}")
        print(f"  dates:   {hf['dates'].shape}, dtype={hf['dates'].dtype}")
        print(f"  (No labels — this is the test set)")



    

    
