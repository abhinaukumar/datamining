import os
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Code to preprocess data from the eICU database')
parser.add_argument('--path', help='Path to eICU database', required=True, type=str)
args = parser.parse_args()

assert len(args.path) > 0, 'Empty path'

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
#nursingchart = pd.read_csv(os.path.join(args.path, 'nurseCharting.csv'))
#
## Select features of interest and keys
#nursingchart = nursingchart.loc[:, ['nursingchartid', 'patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue']]
#nursingchart = nursingchart.loc[nursingchart['nursingchartcelltypevalname'].isin(['GCS Total', 'Verbal', 'Motor', 'Eyes'])]

lab = pd.read_csv(os.path.join(args.path,  'lab.csv'))
col = ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']
new_lab=lab[col]
f_lab= new_lab[new_lab['labname'].isin(['glucose','bedside glucose'])]


#Willrewrite this function
def check(x):
    try:
        x = float(str(x).strip())
    except:
        x = np.nan
    return x

def check_itemvalue(df):
    df['labresult']=df['labresult'].apply(lambda x: check(x))
    df['labresult']=df['labresult'].astype(float)
    return df


f_lab['labname'].replace({"bedside glucose": "glucose"}, inplace=True)
f_lab=check_itemvalue(f_lab)

#binningo
#df['glucose'] = df['glucose'].shift(-1)

def bin(df,x,label_groupby,col):
	df.dropna(how='all', subset=col, inplace=True)
	df[label_groupby] = (df[label_groupby] / x).astype(int)
	df = df.groupby(label_groupby).apply(lambda x: x.fillna(x.mean()))
	#df = df.unstack()
	df = df.droplevel(0, axis=0)
	df.drop_duplicates(subset=[label_groupby], keep='last', inplace=True)
	return df

f_lab=bin(f_lab,60,'labresultoffset',col)
	
	


print(f_lab['labresult'].describe())
#do imputation-
print("Before Empty results are ",f_lab.isnull().sum())
#TODO:Need to check why it is 36
f_lab['labresult'].fillna(value=128, inplace=True)
print("After Empty results are ",f_lab.isnull().sum())

for stayid in stayids:
    # Get records corresponding to one unit stay ID
    df = f_lab.loc[f_lab['patientunitstayid'] == stayid]

    # Write to CSV file. nurseCharting is too large to handle in full
    df.to_csv(os.path.join('glucose', str(stayid) + '.csv'))

    del df

