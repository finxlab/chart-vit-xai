import os
import pandas as pd
from tqdm import tqdm

crsp = pd.read_csv('DB/crsp_ver1_price_w_delist_2510.csv',low_memory=False)

crsp = crsp[crsp['SHRCD'].isin([10,11])].copy()
crsp = crsp[crsp['EXCHCD'].isin([1,2,3])].copy()
crsp = crsp.drop(columns=['SHRCD','EXCHCD','NUMTRD']).copy()
crsp['RET'] = crsp['RET'].replace('C',0).astype(float).copy()
crsp.reset_index(inplace=True,drop=True)

os.makedirs('DB/train/stocks', exist_ok=True)

permno_lst = crsp['PERMNO'].unique().tolist()

ret_ck_lst = []
vol_ck_lst = []

for permno in tqdm(permno_lst):
    temp = crsp[crsp['PERMNO'] == permno].copy()
    temp = temp.set_index('date').copy()
    temp.index.name = None
    temp.index = pd.to_datetime(temp.index)
    temp = temp.drop(columns = ['PERMNO','TICKER','COMNAM','HSICCD','DLAMT','DLPDT','NEXTDT','DLPRC','DLRET','SHROUT']).copy()
    temp = temp[('1992-12-03' <= temp.index) & (temp.index <= '2000-12-31')].copy()
    temp = temp[temp.isna().sum(axis=1)!=6].copy() # ['BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET', 'OPENPRC']

    if len(temp) < 55: # rough cut, 20 days for ma, 20 days for image, 20 days for prediction
        continue

    if temp['RET'].isna().sum() > 0:
        temp['RET'] = temp['RET'].fillna(0)
        ret_ck_lst.append(permno)

    if temp['VOL'].isna().sum() > 0:
        temp['VOL'] = temp['VOL'].fillna(0)
        vol_ck_lst.append(permno)

    temp.loc[temp.index[0], 'RET'] = 0
    temp.to_pickle('DB/train/stocks/' + str(permno) + '.pkl')

print(f'missing return permno list: {(ret_ck_lst)}')
print(f'missing volume permno list: {(vol_ck_lst)}')