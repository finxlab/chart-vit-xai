"""
Chart Image Generation for ViT-B/32 (224x224x3)

Professor's Final Configuration:
- 20 days: 6px candle + 5px blank → 215px chart + 9px padding
- 25 days: 6px candle + 3px blank → 222px chart + 2px padding
- 30 days: 6px candle + 1px blank → 209px chart + 15px padding

Common:
- Image: 224 x 224 x 3 (RGB)
- Price: 160px (scale 0-159) | Gap: 1px | Volume: 63px (ceil-based)
- Candle body: 6px | Wick: 2px centered (pixels 2-3)
- MA20: 1px cv2.line → manual 3px widen (±1 vertically)
"""

import os
import cv2
import h5py
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224

CANDLE_BODY_WIDTH = 6
WICK_WIDTH = 2

PRICE_HEIGHT = 160           # 5 patch rows (160 / 32 = 5)
GAP_HEIGHT = 1               # FIX #1: 1px blank between price and volume
VOL_HEIGHT = 63              # FIX #5: 63px (was 64)

# Verify: 160 + 1 + 63 = 224
assert PRICE_HEIGHT + GAP_HEIGHT + VOL_HEIGHT == IMAGE_SIZE, \
    f"Height mismatch: {PRICE_HEIGHT} + {GAP_HEIGHT} + {VOL_HEIGHT} != {IMAGE_SIZE}"

PRICE_SCALE = PRICE_HEIGHT - 1   # 0 to 159
VOL_SCALE = VOL_HEIGHT           # FIX #5: ceil-based, max 63 (was 64)

MA_WINDOW = 20
HORIZON_DAYS = 20

BLANK_MAP = {
    20: 5,
    25: 3,
    30: 1,
}

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
    
    Rules:
    - Find all nonzero pixels in the matrix
    - For each pixel at row y, set pixels at y-1 and y+1
    - If y == 0 (top boundary): only expand downward (y+1)
    - If y == price_height-1 (bottom boundary): only expand upward (y-1)
    
    Parameters:
    -----------
    blue_matrix : np.ndarray
        Shape (price_height, IMAGE_SIZE), with 1px MA line drawn
    price_height : int
        Height of the price section
    
    Returns:
    --------
    widened : np.ndarray
        Same shape, with 3px wide MA line
    """
    
    widened = blue_matrix.copy()
    
    # Find all nonzero pixel locations (before flipud)
    ys, xs = np.where(blue_matrix > 0)
    
    for y, x in zip(ys, xs):
        # Expand upward (y - 1) if not at top
        if y > 0:
            widened[y - 1, x] = 1
        # Expand downward (y + 1) if not at bottom
        if y < price_height - 1:
            widened[y + 1, x] = 1
    
    return widened

def price_scaling(dataframe, size):
    
    """
    Scale price data to pixel range [0, size].
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Columns: ADJ_OPEN, ADJ_HIGH, ADJ_LOW, ADJ_CLOSE, MA20
    size : int
        Maximum pixel value (e.g., 159 for 160px height)
    """

    if np.nanmin(dataframe) == np.nanmax(dataframe):
        # All values same -> place at middle
        return np.round((dataframe) / (np.nanmax(dataframe)) * size // 2, 0)
    else:
        return np.round(
            (dataframe - np.nanmin(dataframe)) / 
            (np.nanmax(dataframe) - np.nanmin(dataframe)) * size, 0
        )
    
def vol_scaling(dataframe, size):

    """
    Scale volume data using ceiling division.
    Ensures any nonzero volume gets at least 1 pixel.
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Column: VOL
    size : int
        Maximum pixel height for volume (e.g., 64)
    """

    dataframe = dataframe.copy()
    if dataframe.max().values == 0:
        dataframe['VOL'] = [0] * len(dataframe)
        return dataframe
    scaled_vol = (np.ceil((dataframe / dataframe.max()) / (1 / size))).astype(int)
    return scaled_vol

# ============================================================
# Image Generation
# ============================================================

def generate_data_from_pickle(pickle_path, image_days=25):

    assert image_days in [20, 25, 30]

    layout = get_chart_layout(image_days)
    left_pad = layout['left_pad']
    px_per_day = layout['px_per_day']

    permno = int(pickle_path.split('/')[-1].split('.pkl')[0])

    stock = pd.read_pickle(pickle_path)
    stock.columns = ['LOW', 'HIGH', 'CLOSE', 'VOL', 'RET', 'OPEN']
    stock = stock[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'RET']].copy()

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

    stock = stock[['ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'MA20', 'VOL', 'COLOR', 'PRICE_CHANGE']].copy()
    stock['VOL'] = stock['VOL'].fillna(0)
    stock = stock[('1993-01-01') <= stock.index].copy()

    image_lst, label_lst, date_lst, permno_lst = [], [], [], []

    for i in range(image_days, len(stock) - HORIZON_DAYS + 1):
        date = str(stock.index[i - 1]).split(' ')[0]
        date_int = int(date.replace('-', ''))
        temp = stock.iloc[i - image_days:i].copy()

        start_price = stock.iloc[i - 1]['ADJ_CLOSE']
        end_price = stock.iloc[i + HORIZON_DAYS - 1]['ADJ_CLOSE']
        label = np.round(end_price / start_price - 1, 4)

        scaled_price = price_scaling(
            temp[['ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'MA20']],
            PRICE_SCALE      # 159
        ).astype(int)

        scaled_vol = vol_scaling(
            temp[['VOL']],
            VOL_SCALE         # 63 
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
        label_lst.append(label)
        permno_lst.append(permno)
        date_lst.append(date_int)

    return np.stack(image_lst), np.array(label_lst), np.array(permno_lst), np.array(date_lst)

def process_single_stock(pickle_path, output_dir, image_days=25):
    """
    Process one stock pickle → save as individual HDF5 file.
    
    Each per-stock HDF5 contains:
      - images:  (N, 3, 224, 224)  uint8
      - labels:  (N,)              float32
      - permnos: (N,)              int32
      - dates:   (N,)              int32
    
    Returns:
      (permno, n_samples) on success, (permno, 0) on failure
    """
    
    permno = int(pickle_path.split('/')[-1].split('.pkl')[0])
    out_path = os.path.join(output_dir, f'{permno}.h5')
    
    # --- Skip if already exists (resume-friendly) ---
    if os.path.exists(out_path):
        try:
            with h5py.File(out_path, 'r') as hf:
                n = hf['labels'].shape[0]
            return (permno, n)
        except:
            os.remove(out_path)  # corrupted, regenerate
    
    try:
        images, labels, permnos, dates = generate_data_from_pickle(pickle_path, image_days)
        
        if images.size == 0:
            return (permno, 0)
        
        n = len(labels)
        images_chw = np.transpose(images, (0, 3, 1, 2))  # (N,224,224,3) → (N,3,224,224)
        
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset('images',  data=images_chw, dtype=np.uint8,
                              chunks=(min(32, n), 3, IMAGE_SIZE, IMAGE_SIZE),
                              compression='gzip')
            hf.create_dataset('labels',  data=labels.astype(np.float32))
            hf.create_dataset('permnos', data=permnos.astype(np.int32))
            hf.create_dataset('dates',   data=dates.astype(np.int32))
        
        return (permno, n)
    
    except Exception as e:
        # Clean up partial file if it exists
        if os.path.exists(out_path):
            os.remove(out_path)
        return (permno, 0)
    
def build_per_stock_h5(stock_dir, output_dir, image_days=25, n_jobs=-1):
    """
    Step 1: Generate one HDF5 file per stock in parallel.
    
    Parameters:
    -----------
    stock_dir  : str  - directory containing .pkl files
    output_dir : str  - directory to save per-stock .h5 files
    image_days : int  - 20, 25, or 30
    n_jobs     : int  - number of parallel workers (-1 = all cores)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    stock_list = sorted(glob(f'{stock_dir}/*.pkl'))
    stock_list = [Path(path).as_posix() for path in stock_list]
    print(f"Found {len(stock_list)} stocks")
    
    # Check how many already done (for resume)
    existing = set(int(f.split('.h5')[0]) for f in os.listdir(output_dir) if f.endswith('.h5'))
    print(f"Already completed: {len(existing)} stocks")
    
    # --- Parallel processing ---
    print(f"\n[Step 1] Generating per-stock HDF5 files (n_jobs={n_jobs})...")
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_stock)(path, output_dir, image_days)
        for path in stock_list
    )
    
    # --- Summary ---
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

def merge_h5_files(chunk_dir, output_path, delete_chunks=False):
    """
    Step 2: Merge all per-stock HDF5 files into one final HDF5.
    
    Parameters:
    -----------
    chunk_dir     : str  - directory containing per-stock .h5 files
    output_path   : str  - path for merged output .h5 file
    delete_chunks : bool - if True, delete per-stock files after merging
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
                n = hf['labels'].shape[0]
                if n > 0:
                    valid_files.append((fpath, n))
                    total_samples += n
        except Exception as e:
            print(f"  Skipping corrupted file: {fpath} ({e})")
    
    print(f"Valid files: {len(valid_files)}, Total samples: {total_samples:,}")
    
    # --- Pass 2: Merge into single HDF5 ---
    print(f"\n[Pass 2] Merging into {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as hf_out:
        # Pre-allocate datasets with maxshape for safety
        img_ds = hf_out.create_dataset(
            'images',
            shape=(total_samples, 3, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
            chunks=(32, 3, IMAGE_SIZE, IMAGE_SIZE),
            compression='gzip',
            maxshape=(None, 3, IMAGE_SIZE, IMAGE_SIZE),
        )
        label_ds  = hf_out.create_dataset('labels',  shape=(total_samples,), dtype=np.float32, maxshape=(None,))
        permno_ds = hf_out.create_dataset('permnos', shape=(total_samples,), dtype=np.int32,   maxshape=(None,))
        date_ds   = hf_out.create_dataset('dates',   shape=(total_samples,), dtype=np.int32,   maxshape=(None,))
        
        current_idx = 0
        
        for fpath, n in tqdm(valid_files, desc="Merging"):
            try:
                with h5py.File(fpath, 'r') as hf_in:
                    img_ds[current_idx:current_idx + n]    = hf_in['images'][:]
                    label_ds[current_idx:current_idx + n]  = hf_in['labels'][:]
                    permno_ds[current_idx:current_idx + n] = hf_in['permnos'][:]
                    date_ds[current_idx:current_idx + n]   = hf_in['dates'][:]
                
                current_idx += n
                
            except Exception as e:
                print(f"\nFailed to read: {fpath} ({e})")
        
        # Trim if some files failed during merge
        if current_idx < total_samples:
            print(f"\nTrimming: expected {total_samples}, got {current_idx}")
            img_ds.resize(current_idx, axis=0)
            label_ds.resize(current_idx, axis=0)
            permno_ds.resize(current_idx, axis=0)
            date_ds.resize(current_idx, axis=0)
    
    print(f"\nDone! Saved {current_idx:,} samples to {output_path}")
    
    # --- Optional: delete per-stock files ---
    if delete_chunks:
        print("Deleting per-stock HDF5 files...")
        for fpath, _ in valid_files:
            os.remove(fpath)
        print("Deleted.")
    
    return current_idx

if __name__ == '__main__':
    
    # ============================================================
    # Configuration
    # ============================================================
    parser = argparse.ArgumentParser(description='Generate RGB chart images for ViT training')
    parser.add_argument('--image_days', type=int, default=30, choices=[20, 25, 30],
                        help='Number of trading days per image (20, 25, or 30)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel workers (-1 = all cores)')
    args = parser.parse_args()
    
    IMAGE_DAYS = args.image_days
    STOCK_DIR  = 'DB/train/stocks'
    CHUNK_DIR  = f'DB/train/chunks_h5/{IMAGE_DAYS}d'
    FINAL_H5   = f'DB/train/rgb_{IMAGE_DAYS}d_train.h5'
    
    # ============================================================
    # Step 1: Per-stock HDF5 (parallel)
    # ============================================================
    results = build_per_stock_h5(STOCK_DIR, CHUNK_DIR, image_days=IMAGE_DAYS, n_jobs=args.n_jobs)
    
    # ============================================================
    # Step 2: Merge into one file
    # ============================================================
    total = merge_h5_files(CHUNK_DIR, FINAL_H5, delete_chunks=False)