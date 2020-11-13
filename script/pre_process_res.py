import os
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import argparse



parser = argparse.ArgumentParser(description='Code to preprocess data from the eICU database')
parser.add_argument('--path', help='Path to eICU database', required=True, type=str)
parser.add_argument( '--dump_hist' , type=int, default= 1 , help= 'Enable historgam plot ' )
args = parser.parse_args()

assert len(args.path) > 0, 'Empty path'
dump_hist = args.dump_hist

if dump_hist==1:
	tb = SummaryWriter(comment="respiratory")
# Read patients.csv

patients = pd.read_csv(os.path.join(args.path, 'patient.csv'))

# Remove patients having age <= 18 and age >= 89.
patients = patients.loc[patients['age'] != '> 89']
patients = patients.astype({'age': 'float'})
patients = patients.loc[(patients['age'] > 18) & (patients['age'] < 89)]

# Remove patients having more than one visit.
id_counts = patients['uniquepid'].value_counts(ascending = True)
single_visit_ids = id_counts[id_counts == 1].keys()
patients = patients.loc[patients['uniquepid'].isin(single_visit_ids)]

# Remove patients having invalid gender
patients = patients.loc[patients['gender'] != 'Unknown'] # Removes records having unknown gender
patients = patients.loc[patients['gender'].notnull()] # Removes records having NaN gender

# Remove patients having invalid discharge status
patients = patients.loc[patients['hospitaldischargestatus'].notnull()]
patients = patients.loc[patients['unitdischargestatus'].notnull()]

# Select unique stayids
stayids = patients['patientunitstayid'].unique()

patients.to_csv(os.path.join(args.path, 'filtered_patient.csv'))
del patients

# Read nursingChart.csv
nc = pd.read_csv(os.path.join(args.path, 'nurseCharting.csv'))
#
## Select features of interest and keys
filter_val=['Respiratory Rate']
#col=['nursingchartid', 'patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel','nursingchartcelltypevalname', 'nursingchartvalue']
col=['nursingchartid', 'patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel','nursingchartcelltypevalname', 'nursingchartvalue']   
new_nc=nc[col]

del nc
new_nc = new_nc[new_nc['nursingchartcelltypevallabel'].isin(filter_val)]
def check(x):
    try:
        x = float(str(x).strip())
    except:
        x = np.nan
    return x
def check_itemvalue(df):
    df['nursingchartvalue']=df['nursingchartvalue'].apply(lambda x: check(x))
    df['nursingchartvalue']=df['nursingchartvalue'].astype(float)
    return df


new_nc=check_itemvalue(new_nc)


def bin(df,x,label_groupby,col):
	df.dropna(how='all', subset=col, inplace=True)
	df[label_groupby] = (df[label_groupby] / x).astype(int)
	df = df.groupby(label_groupby).apply(lambda x: x.fillna(x.mean()))
	#df = df.unstack()
	df = df.droplevel(0, axis=0)
	df.drop_duplicates(subset=[label_groupby], keep='last', inplace=True)
	return df

new_nc=bin(new_nc,60,'nursingchartoffset',col)
#do imputation-
if dump_hist==1:
	tb.add_histogram('Respiratory Rate before imputation', torch.tensor(np.array(new_nc['nursingchartvalue'])))
print("Before Empty results are ",new_nc.isnull().sum())
#TODO:Need to check why it is 19
new_nc['nursingchartvalue'].fillna(value=19, inplace=True)
print("After Empty results are ",new_nc.isnull().sum())
if dump_hist==1:
	tb.add_histogram('Respiratory Rate After imputation', torch.tensor(np.array(new_nc['nursingchartvalue'])))

for stayid in stayids:
    # Get records corresponding to one unit stay ID
    df = new_nc.loc[new_nc['patientunitstayid'] == stayid]

    # Write to CSV file. nurseCharting is too large to handle in full
    df.to_csv(os.path.join('respiratory_rate', str(stayid) + '.csv'))

    del df

del new_nc
