import os

import numpy as np
import pandas as pd

import argparse

# parser = argparse.ArgumentParser(description='Code to preprocess data from the eICU database')
# parser.add_argument('--path', help='Path to eICU database', required=True, type=str)
# args = parser.parse_args()


path = 'D:\Workspace\Pycharm\DataMiningProject'
assert len(path) > 0, 'Empty path'


# Read patients.csv
patients = pd.read_csv(os.path.join(path, 'patient.csv'))

# Remove patients having age <= 18 and age >= 89.
patients = patients.loc[patients['age'] != '> 89']
patients = patients.astype({'age': 'float'})
patients = patients.loc[(patients['age'] > 18) & (patients['age'] < 89)]

# Remove patients having more than one visit.:w

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

patients.to_csv(os.path.join(path, 'filtered_patient.csv'))
del patients

# # Read lab.csv
# labdata = pd.read_csv(os.path.join(path, 'lab.csv'))
#
# # Select features of interest and keys
# labdata = labdata.loc[labdata['labname'].isin(['pH','FiO2'])]
#
# for stayid in stayids:
#     # Get records corresponding to one unit stay ID
#     df = labdata.loc[labdata['patientunitstayid'] == stayid]
#     # Write to CSV file. nurseCharting is too large to handle in full
#     patient = patients.loc[patients['patientunitstayid'] == stayid]
#     height = patient['admissionheight'].values[0]
#     ad_weight = patient['admissionweight'].values[0]
#     dc_weight = patient['dischargeweight'].values[0]
#     admission_offset = patient['hospitaladmitoffset'].values[0]
#     discharge_offset = patient['hospitaldischargeoffset'].values[0]
#     df = df.append(
#         [{'patientunitstayid': stayid, 'labresultoffset': admission_offset, 'labname': 'height', 'labresult': height}],
#         ignore_index=True)
#     df = df.append(
#         [{'patientunitstayid': stayid, 'labresultoffset': admission_offset, 'labname': 'weight', 'labresult': ad_weight}],
#         ignore_index=True)
#     df = df.append(
#         [{'patientunitstayid': stayid, 'labresultoffset': discharge_offset, 'labname': 'weight','labresult': dc_weight}],
#         ignore_index=True)
#     df.to_csv(os.path.join(path, 'lab_split', str(stayid) + '.csv'))
#     del df
# del labdata

# Collecting features.
features = ['pH', 'FiO2','height','weight']
# Typical values for imputation, from Benchmarking ML algorithms paper.
impute_values = dict(zip(features, [7.4, 0.21, 170, 81]))

for i_stayid, stayid in enumerate(stayids):
    df = pd.read_csv(os.path.join(path, 'lab_split', str(stayid) + '.csv'))
    if df.shape[0] != 0:
        # Sort by offset, thereby arranging in time series order
        df = df.sort_values(by='labresultoffset')
        offsets = df['labresultoffset'].unique()

        out_df = pd.DataFrame(offsets, columns=['labresultoffset'])
        for feat in features:
            out_df.insert(out_df.shape[1], feat, np.nan, allow_duplicates=False)

        offset_groups = df.groupby('labresultoffset')
        i_offset = 0
        for offset, group in offset_groups:
            avail_feats = group['labname'].unique()
            for feat in features:
                if feat in avail_feats:
                    out_df.loc[i_offset, feat] = group.loc[group['labname'] == feat, 'labresult'].to_numpy()[0]
                    if feat == 'FiO2':
                        out_df.loc[i_offset, feat] = out_df.loc[i_offset, feat]/100
            i_offset += 1

        out_df['labresultoffset'] = (out_df['labresultoffset'] / 60).astype('int')

        # Impute values within offset by replacing NaN with mean over each column.
        out_df.groupby('labresultoffset').apply(lambda x: x.fillna(x.mean()))
        # For each offset, only choose last value.
        out_df.drop_duplicates('labresultoffset', keep='last', inplace=True)
        # Impute missing values with "typical values"


    else:
        out_df = pd.DataFrame(columns=['labresultoffset'])
    # out_df = out_df.loc[['labid','patientunitstayid','labresultoffset','labtypeid','labname','labresult']]
    out_df.insert(loc = 0, column = 'patientunitstayid', value = stayid,allow_duplicates = False)
    out_df['height'] = out_df['height'].mean()
    out_df['weight'] = out_df['weight'].mean()
    out_df.fillna(value=impute_values, inplace=True)
    out_df.to_csv(os.path.join(path, 'Features', str(stayid) + '.csv'), index=False)
    del df, out_df


