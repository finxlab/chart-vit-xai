"""
Test Data Preprocessing
================================

This script:
1. Loads CRSP data and filters for common stocks
2. Generates rebalance dates (every 20 trading days from 2000-12-29)
3. Saves per-stock pickle files for the test period (from 2000-10-01)

Directory structure after running:
DB/test/
├── stocks/            ← per-stock pickle files
rebalance_date.csv     ← rebalance + entry point dates

Note:
- Test data starts from 2000-10-01 to allow ~20 trading days of
  lookback (MA20) before the first rebalance date on 2000-12-29.
- The rough cut of 55 days ensures enough data for MA20 + image + horizon.
"""

import os
import pandas as pd
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

CRSP_PATH = 'DB/crsp_ver2_price_2603.csv'
TEST_STOCK_DIR = 'DB/test/stocks'
REBALANCE_CSV = 'DB/rebalance_date.csv'
TEST_START_DATE = '2000-10-01'           # lookback buffer before first rebalance
REBALANCE_START_DATE = '2000-12-29'      # first rebalance date
REBALANCE_INTERVAL = 20                  # every 20 trading days
MIN_OBS = 50                             # rough cut: 20 (MA) + 30 (image)

# ============================================================
# Step 1: Load and Filter CRSP VER 2
# ============================================================

print("Loading CRSP data...")
crsp = pd.read_csv(CRSP_PATH, low_memory=False)

crsp = crsp[
    (crsp['SecurityType'] == 'EQTY') &
    (crsp['SecuritySubType'] == 'COM') &
    (crsp['ShareType'] == 'NS') &
    (crsp['USIncFlg'] == 'Y') &
    (crsp['IssuerType'].isin(['ACOR', 'CORP'])) &
    (crsp['PrimaryExch'].isin(['N', 'A', 'Q'])) 
    # &
    # (crsp['TradingStatusFlg'] == 'A') &
    # (crsp['ConditionalType'] == 'RW')
].copy()

print(f"  Total rows after filter: {len(crsp):,}")

# # ============================================================
# # Step 2: Generate Rebalance Dates CRSP VER 2
# # ============================================================

all_date = crsp['DlyCalDt'].drop_duplicates().sort_values().reset_index(drop=True)

# Remove ghost dates (market closed but CRSP logged empty rows: e.g., 2001-09-11, 2012-10-29)
date_counts = crsp.dropna(subset=['DlyPrc']).groupby('DlyCalDt').size()
ghost_dates = set(all_date) - set(date_counts.index)
if ghost_dates:
    print(f"  Removing {len(ghost_dates)} ghost dates: {sorted(ghost_dates)}")
all_date = all_date[~all_date.isin(ghost_dates)].reset_index(drop=True)

rebalance_dates = list(all_date[REBALANCE_START_DATE <= all_date].reset_index(drop=True)[::REBALANCE_INTERVAL])
entry_point_dates = list(all_date[REBALANCE_START_DATE <= all_date].reset_index(drop=True)[1::REBALANCE_INTERVAL])

pd.DataFrame([rebalance_dates, entry_point_dates]).T.to_csv(REBALANCE_CSV)
print(f"  Rebalance dates: {len(rebalance_dates)}")
print(f"  Saved to: {REBALANCE_CSV}")

# ============================================================
# Step 3: Save Per-Stock Pickles (Test Period)
# ============================================================

os.makedirs(TEST_STOCK_DIR, exist_ok=True)

test_crsp = crsp[TEST_START_DATE <= crsp['DlyCalDt']].reset_index(drop=True).copy()
permno_lst = test_crsp['PERMNO'].unique().tolist()

ret_ck_lst = []
vol_ck_lst = []

print(f"\nProcessing {len(permno_lst)} stocks for test period...")

for permno in tqdm(permno_lst):
    temp = test_crsp[test_crsp['PERMNO'] == permno].copy()
    temp = temp.set_index('DlyCalDt').copy()
    temp.index.name = None
    temp.index = pd.to_datetime(temp.index)
    temp = temp[['DlyOpen','DlyHigh','DlyLow','DlyPrc','DlyVol','DlyRet','TradingStatusFlg']].copy()
    temp = temp.drop_duplicates().copy()

    if len(temp) < MIN_OBS:
        continue

    if temp['DlyRet'].isna().sum() > 0:
        temp['DlyRet'] = temp['DlyRet'].fillna(0)
        ret_ck_lst.append(permno)

    if temp['DlyVol'].isna().sum() > 0:
        temp['DlyVol'] = temp['DlyVol'].fillna(0)
        vol_ck_lst.append(permno)

    temp.loc[temp.index[0], 'DlyRet'] = 0
    temp.to_pickle(os.path.join(TEST_STOCK_DIR, f'{permno}.pkl'))

print(f'\nmissing return permno list: {ret_ck_lst}')
print(f'missing volume permno list: {vol_ck_lst}')
print(f'\nTotal stocks saved: {len(os.listdir(TEST_STOCK_DIR))}')