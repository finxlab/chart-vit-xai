import os
import cv2
import h5py
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

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
HORIZON_DAYS = 20
MA_WINDOW = 20

def price_scaling(dataframe, size):
    if np.nanmin(dataframe) == np.nanmax(dataframe): # when all values are same 
        return np.round((dataframe) / (np.nanmax(dataframe)) * size//2,0) #.astype(int) # return half of size -> middle of the image
    else:
        return np.round((dataframe - np.nanmin(dataframe)) / (np.nanmax(dataframe) - np.nanmin(dataframe)) * size,0) # .astype(int)
        
def vol_scaling(dataframe,size):
    dataframe = dataframe.copy()

    if dataframe.max().values == 0: # all values are 0
        dataframe['VOL'] = [0] * len(dataframe)
        return dataframe
    
    scaled_vol = (np.ceil((dataframe/dataframe.max())/(1/size))).astype(int) #.astype(int)
    
    return scaled_vol

def generate_data_from_pickle (pickle_path):
    
    permno = int(pickle_path.split('/')[-1].split('.pkl')[0])

    stock = pd.read_pickle(pickle_path)
    stock.columns = ['LOW','HIGH','CLOSE','VOL','RET','OPEN']
    stock = stock[['OPEN','HIGH','LOW','CLOSE','VOL','RET']].copy()

    stock['CLOSE'] = np.abs(stock['CLOSE'])
    stock['OPEN'] = np.abs(stock['OPEN'])
    stock['HIGH'] = np.abs(stock['HIGH'])
    stock['LOW'] = np.abs(stock['LOW'])
    
    stock['ADJ_CLOSE'] = (stock['RET']+1).cumprod() * stock['CLOSE'].iloc[0]
    stock['ADJ_OPEN'] = stock['ADJ_CLOSE'] / stock['CLOSE'] * stock['OPEN']
    stock['ADJ_HIGH'] = stock['ADJ_CLOSE'] / stock['CLOSE'] * stock['HIGH']
    stock['ADJ_LOW'] = stock['ADJ_CLOSE'] / stock['CLOSE'] * stock['LOW']

    stock["MA20"] = stock['ADJ_CLOSE'].rolling(window=MA_WINDOW,min_periods=1).mean()
    stock = stock.iloc[19:].copy()

    stock = stock[['ADJ_OPEN','ADJ_HIGH','ADJ_LOW','ADJ_CLOSE','MA20','VOL']].copy()
    stock['VOL'] = stock['VOL'].fillna(0)
    stock = stock[('1993-01-01')<= stock.index].copy()

    image_lst = []
    label_lst = []
    date_lst = []
    permno_lst = []

    for i in range(IMAGE_DAYS,len(stock) - HORIZON_DAYS+1):
        date = str(stock.index[i-1]).split(' ')[0]
        date_int = int(date.replace('-', ''))
        temp = stock.iloc[i-IMAGE_DAYS:i].copy()

        start_price = stock.iloc[i - 1]['ADJ_CLOSE']
        end_price = stock.iloc[i + HORIZON_DAYS - 1]['ADJ_CLOSE']

        label = np.round(end_price/start_price - 1,4)

        scaled_price = price_scaling(temp[['ADJ_OPEN','ADJ_HIGH','ADJ_LOW','ADJ_CLOSE','MA20']],50)
        scaled_vol = vol_scaling(temp[['VOL']],12)

        width = 60
        height = 12 + 1
        vol_matrix = np.zeros((height, width), dtype=np.uint8)

        for i in range(20):
            vol_matrix[0:scaled_vol['VOL'].iloc[i], 3*i + 1] = 1
        
        vol_matrix = np.flipud(vol_matrix)

        idx_lst = np.array([3*i+1 for i in range(20)])

        height = 51
        price_matrix = np.zeros((height, width), dtype=np.uint8)

        o = 50 - scaled_price['ADJ_OPEN']
        h = 50 - scaled_price['ADJ_HIGH']
        l = 50 - scaled_price['ADJ_LOW']
        c = 50 - scaled_price['ADJ_CLOSE']

        price_matrix[np.array(o[~o.isna()].astype(int)),idx_lst[~o.isna()]-1] = 1
        price_matrix[np.array(c[~c.isna()].astype(int)),idx_lst[~c.isna()]+1] = 1

        not_na_h = h[~(h.isna() | l.isna())].astype(int).values
        not_na_l = l[~(h.isna() | l.isna())].astype(int).values
        not_na_idx = idx_lst[~(h.isna() | l.isna())]

        for i in range(len(not_na_idx)):
            price_matrix[not_na_h[i]:not_na_l[i]+1, not_na_idx[i]] = 1

        ma = 50 - scaled_price['MA20'].astype(int).values

        for i in range(20):
            if i < 19:
                cv2.line(price_matrix, (idx_lst[i],ma[i]), (idx_lst[i+1],ma[i+1]), 1, 1)

        chart = np.vstack((price_matrix, vol_matrix)) * 255

        image_lst.append(chart)
        label_lst.append(label)
        permno_lst.append(permno)
        date_lst.append(date_int)

    images_np = np.stack(image_lst)   # Shape: (N, 80, 80)
    labels_np = np.array(label_lst)   # Shape: (N,)
    permno_np = np.array(permno_lst)   # Shape: (N,)
    dates_np = np.array(date_lst)     # Shape: (N,)

    return images_np, labels_np, permno_np, dates_np

def process_and_save_chunk(pickle_path, output_dir):
    """Process one stock → per-stock HDF5 file."""
    permno = int(pickle_path.split('/')[-1].split('.pkl')[0])
    out_path = os.path.join(output_dir, f'{permno}.h5')

    # Skip if already exists
    if os.path.exists(out_path):
        try:
            with h5py.File(out_path, 'r') as hf:
                n = hf['images'].shape[0]
            return (permno, n)
        except:
            os.remove(out_path)

    try:
        images_np, labels_np, permnos_np, dates_np = generate_data_from_pickle(pickle_path)

        if images_np.size == 0:
            return (permno, 0)

        n = len(labels_np)
        images_chw = images_np[:, np.newaxis, :, :]  # (N,64,60) → (N,1,64,60)

        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset('images',  data=images_chw, dtype=np.uint8,
                              chunks=(min(128, n), 1, IMAGE_H, IMAGE_W),
                              compression='gzip')
            hf.create_dataset('labels',  data=labels_np.astype(np.float32))
            hf.create_dataset('permnos', data=permnos_np.astype(np.int32))
            hf.create_dataset('dates',   data=dates_np.astype(np.int32))

        return (permno, n)

    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"Failed: {pickle_path} ({e})")
        return (permno, 0)

def combine_chunks_to_hdf5(chunk_dir, hdf5_path):
    """Merge all per-stock HDF5 files into a single HDF5."""

    h5_files = sorted(glob(f'{chunk_dir}/*.h5'))
    print(f"\n[Step 2] Merging {len(h5_files)} files...")

    # Pre-calculate total size
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

    # Merge
    with h5py.File(hdf5_path, 'w') as hf_out:
        img_ds    = hf_out.create_dataset('images',  shape=(total_samples, 1, IMAGE_H, IMAGE_W),
                                           dtype=np.uint8, chunks=(128, 1, IMAGE_H, IMAGE_W),
                                           compression='gzip', maxshape=(None, 1, IMAGE_H, IMAGE_W))
        label_ds  = hf_out.create_dataset('labels',  shape=(total_samples,), dtype=np.float32, maxshape=(None,))
        permno_ds = hf_out.create_dataset('permnos', shape=(total_samples,), dtype=np.int32, maxshape=(None,))
        date_ds   = hf_out.create_dataset('dates',   shape=(total_samples,), dtype=np.int32, maxshape=(None,))

        current_idx = 0
        for fpath, n in tqdm(valid_files, desc="  Merging"):
            try:
                with h5py.File(fpath, 'r') as hf_in:
                    img_ds[current_idx:current_idx + n]    = hf_in['images'][:]
                    label_ds[current_idx:current_idx + n]  = hf_in['labels'][:]
                    permno_ds[current_idx:current_idx + n] = hf_in['permnos'][:]
                    date_ds[current_idx:current_idx + n]   = hf_in['dates'][:]
                current_idx += n
            except Exception as e:
                print(f"\n  Failed: {fpath} ({e})")

        if current_idx < total_samples:
            print(f"\n  Trimming: expected {total_samples}, got {current_idx}")
            img_ds.resize(current_idx, axis=0)
            label_ds.resize(current_idx, axis=0)
            permno_ds.resize(current_idx, axis=0)
            date_ds.resize(current_idx, axis=0)

    print(f"\nDone! Saved {current_idx:,} samples to {hdf5_path}")
    return current_idx

if __name__ == '__main__':
    
    # ============================================================
    # Configuration
    # ============================================================
    STOCK_DIR = 'DB/train/stocks'
    CHUNK_DIR = 'DB/train/chunks_h5/gray'
    FINAL_H5  = 'DB/train/gray_train.h5'

    os.makedirs(CHUNK_DIR, exist_ok=True)

    stock_list = sorted(glob(f'{STOCK_DIR}/*.pkl'))
    stock_list = [Path(path).as_posix() for path in stock_list]
    print(f"Found {len(stock_list)} stocks")

    # ============================================================
    # Step 1: Per-stock HDF5 (parallel)
    # ============================================================
    print(f"\n[Step 1] Generating per-stock HDF5 (n_jobs=-1)...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_and_save_chunk)(path, CHUNK_DIR)
        for path in stock_list
    )

    total_samples = sum(n for _, n in results)
    success = sum(1 for _, n in results if n > 0)
    failed = sum(1 for _, n in results if n == 0)
    print(f"\nStep 1 done: {success} success, {failed} failed/empty, {total_samples:,} samples")

    # ============================================================
    # Step 2: Merge
    # ============================================================
    combine_chunks_to_hdf5(CHUNK_DIR, FINAL_H5)

    # ============================================================
    # Verification
    # ============================================================
    with h5py.File(FINAL_H5, 'r') as hf:
        print(f"\nVerification:")
        print(f"  images:  {hf['images'].shape}, dtype={hf['images'].dtype}")
        print(f"  labels:  {hf['labels'].shape}, dtype={hf['labels'].dtype}")
        print(f"  permnos: {hf['permnos'].shape}, dtype={hf['permnos'].dtype}")
        print(f"  dates:   {hf['dates'].shape}, dtype={hf['dates'].dtype}")