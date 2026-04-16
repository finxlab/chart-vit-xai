import os
import gc
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob

crsp_price = pd.read_csv('DB/crsp_ver2_price_2603.csv')

crsp_price = crsp_price[
    (crsp_price['SecurityType'] == 'EQTY') &
    (crsp_price['SecuritySubType'] == 'COM') &
    (crsp_price['ShareType'] == 'NS') &
    (crsp_price['USIncFlg'] == 'Y') &
    (crsp_price['IssuerType'].isin(['ACOR', 'CORP'])) &
    (crsp_price['PrimaryExch'].isin(['N', 'A', 'Q'])) 
].copy()

raw_prc = crsp_price[['PERMNO','DlyCalDt','DlyPrc']].drop_duplicates().pivot(index='DlyCalDt',columns = 'PERMNO',values='DlyPrc').copy()
raw_prc.index.name = None
raw_prc.index =pd.to_datetime(raw_prc.index)
raw_prc.columns.name = None
np.abs(raw_prc).to_pickle('DB/raw_prc.pkl')

vol = crsp_price[['PERMNO','DlyCalDt','DlyVol']].drop_duplicates().pivot(index='DlyCalDt',columns = 'PERMNO',values='DlyVol').copy()
vol.index.name = None
vol.index =pd.to_datetime(vol.index)
vol.columns.name = None
vol.to_pickle('DB/vol.pkl')

cap = crsp_price[['PERMNO','DlyCalDt','DlyCap']].drop_duplicates().pivot(index='DlyCalDt',columns = 'PERMNO',values='DlyCap').copy()
cap.index.name = None
cap.index =pd.to_datetime(cap.index)
cap.columns.name = None
cap.to_pickle('DB/cap.pkl')

ret = crsp_price[['PERMNO','DlyCalDt','DlyRet']].drop_duplicates().pivot(index='DlyCalDt',columns = 'PERMNO',values='DlyRet').copy()
ret.index.name = None
ret.index =pd.to_datetime(ret.index)
ret.columns.name = None
ret.to_pickle('DB/raw_ret.pkl')


crsp_delist = pd.read_csv('DB/crsp_ver2_delist_2603.csv')
crsp_delist = crsp_delist[crsp_delist.PERMNO.isin(ret.columns)].reset_index(drop=True)
crsp_delist['DelistingDt'] = pd.to_datetime(crsp_delist['DelistingDt'])

perf_mask = (
    crsp_delist['DelRet'].isna() & 
    crsp_delist['DelActionType'].isin(['GDR', 'GLI', 'GEX'])
)
crsp_delist.loc[perf_mask, 'DelRet'] = -0.3

# MER missing → 0.0
mer_mask = (
    crsp_delist['DelRet'].isna() & 
    (crsp_delist['DelActionType'] == 'MER')
)

crsp_delist.loc[mer_mask, 'DelRet'] = 0.0


last_indices = ret.apply(lambda x: x.dropna().index.max())
crsp_delist['last_ret_date'] = crsp_delist['PERMNO'].map(last_indices)
delret_check = crsp_delist[(crsp_delist['DelistingDt'] != crsp_delist['last_ret_date'])]


mask = (delret_check['DelistingDt'] - delret_check['last_ret_date']).map(lambda x: x.days) < 30


delret = pd.concat([crsp_delist[crsp_delist['DelistingDt'] == crsp_delist['last_ret_date']],delret_check.loc[mask[mask].index]])
delret= delret[['PERMNO','DelistingDt','DelRet']].reset_index(drop=True)


trading_dates = pd.DatetimeIndex(ret.index)

for _, row in delret.iterrows():
    permno = row['PERMNO']
    delist_dt = pd.Timestamp(row['DelistingDt'])
    del_ret = row['DelRet']

    if del_ret ==0:
        continue
    
    if permno not in ret.columns:
        continue

    future_dates = trading_dates[trading_dates > delist_dt]
    
    if len(future_dates) == 0:
        continue
    
    next_bd = future_dates[0]

    ret.loc[next_bd, permno] = del_ret

ret.to_pickle('DB/ret_w_delret.pkl')


final_ret_df = pd.read_pickle("DB/raw_ret.pkl")
not_na_mask = final_ret_df.notna()
first_valid_mask = not_na_mask & (not_na_mask.cumsum() == 1)
final_ret_df[first_valid_mask] = 0.0

cls = (final_ret_df + 1).cumprod()


rebalnace_date = pd.read_csv("DB/rebalance_date.csv",index_col=0)[['0']]


mom_pred = []
str_pred = []

os.makedirs('monthly_prediction/MOM', exist_ok=True)
os.makedirs('monthly_prediction/STR', exist_ok=True)

for date in tqdm(list(rebalnace_date['0'])):

    temp = cls[cls.index<=date].iloc[-241:].copy()
    month_2_12 = temp.iloc[-241:-20].dropna(axis=1,how='any').copy()
    month_1 = temp.iloc[-21:].dropna(axis=1,how='any').copy()

    mom = (month_2_12.iloc[-1]-month_2_12.iloc[0])/month_2_12.iloc[0] 
    str = (month_1.iloc[-1] - month_1.iloc[0])/month_1.iloc[0]

    mom_df = pd.DataFrame(mom).reset_index()
    str_df = pd.DataFrame(str).reset_index()

    mom_df.columns = ['PERMNO','avg']
    str_df.columns = ['PERMNO','avg']

    mom_df.to_csv("monthly_prediction/MOM/" + date + '.csv',index=False)
    str_df.to_csv("monthly_prediction/STR/" + date + '.csv',index=False)


prediction_files = sorted(glob("result/*_prediction.csv"))

for fpath in prediction_files:
    fname = os.path.basename(fpath).replace("_prediction.csv", "")
    model = pd.read_csv(fpath)
    
    seed_cols = [c for c in model.columns if c not in ['date', 'permno']]
    seed_names = [c.replace('prob_', 'seed') for c in seed_cols]
    
    for date in model['date'].unique():
        temp = model[model['date'] == date].reset_index(drop=True).copy()
        temp = temp.drop(columns=['date'])
        temp = temp.set_index('permno').copy()
        temp.index.name = None
        temp = temp.reset_index()
        temp.columns = ['PERMNO'] + seed_names
        temp['avg'] = temp.iloc[:,1:].mean(axis=1)
        temp
                
        date_str = pd.to_datetime(date, format="%Y%m%d").strftime("%Y-%m-%d")
        
        save_dir = os.path.join("monthly_prediction", fname)
        os.makedirs(save_dir, exist_ok=True)
        
        temp.to_csv(os.path.join(save_dir, f"{date_str}.csv"), index=False)