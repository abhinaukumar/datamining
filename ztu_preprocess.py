import os

import numpy as np
import pandas as pd

import argparse
import progressbar

parser = argparse.ArgumentParser(description='Code to preprocess data from the eICU database')
parser.add_argument('--path', default='/media/ztu/Data/eICU_Benchmark/eicu-collaborative-research-database-2.0',
                    help='Path to eICU database', type=str)
args = parser.parse_args()

assert len(args.path) > 0, 'Empty path'

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('StayID')
            ]

# Read patients.csv
patients = pd.read_csv(os.path.join(args.path, 'patient.csv'))  # (200859, 29)

# Filter patients 'Age' feature in patients.csv
patients.loc[patients['age'] == '> 89', 'age'] = 90
patients[['age']] = patients[['age']].fillna(-1)
patients = patients.astype({'age': 'int'})
patients = patients.loc[(patients['age'] >= 18) & (patients['age'] <= 89)]  # (193153, 29)

# Filter those having just one stay in unit
id_counts = patients['uniquepid'].value_counts(ascending = True)
single_visit_ids = id_counts[id_counts == 1].keys()
patients = patients.loc[patients['uniquepid'].isin(single_visit_ids)]  # (96212, 29)

# Filter atients 'Gender' feature in patients.csv, convert to numbers
gender_map = {'Female': 1, 'Male': 2, '': 0, 'NaN': 0, 'Unknown': 0, 'Other': 0}
def transform_gender(gender_series):
    global gender_map
    return {'gender': gender_series.fillna('').apply(lambda s: gender_map[s] if s in gender_map else gender_map[''])}
patients.update(transform_gender(patients.gender))  # (96212, 29) 

# Convert 'Ethnicity' to numbers
ethnicity_map = {'Asian': 1, 'African American': 2, 'Caucasian': 3, 'Hispanic': 4, 'Native American': 5, 'NaN': 0, '': 0}
def transform_ethnicity(ethnicity_series):
    global ethnicity_map
    return {'ethnicity': ethnicity_series.fillna('').apply(lambda s: ethnicity_map[s] if s in ethnicity_map else ethnicity_map[''])}
patients.update(transform_ethnicity(patients.ethnicity)) # (96212, 29) 

# Filter 'Admission Diagnosis' features into numbers
def transform_dx_into_id(df):
    df.apacheadmissiondx.fillna('nodx', inplace=True)
    dx_type = df.apacheadmissiondx.unique()
    dict_dx_key = pd.factorize(dx_type)[1]
    dict_dx_val = pd.factorize(dx_type)[0]
    dictionary = dict(zip(dict_dx_key, dict_dx_val))
    df['apacheadmissiondx'] = df['apacheadmissiondx'].map(dictionary)
    return df
patients = transform_dx_into_id(patients) # (96212, 29) 

# Remove patients having invalid discharge status
patients = patients.loc[patients['hospitaldischargestatus'].notnull()]
patients = patients.loc[patients['unitdischargestatus'].notnull()]  # (95333, 29)

# Select unique stayids
stayids = patients['patientunitstayid'].unique()

ztu_features = patients.loc[:, ['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx']]

ztu_features.to_csv(os.path.join(args.path, 'ztu_features.csv'))

patients.to_csv(os.path.join(args.path, 'filtered_patient.csv'))
del patients

# Commenting out to avoid running this. Loading nurseCharting may break computers
'''
# Read nursingChart.csv
nursingchart = pd.read_csv(os.path.join(args.path, 'nurseCharting.csv'))

# Select features of interest and keys
nursingchart = nursingchart.loc[:, ['nursingchartid', 'patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue']]
nursingchart = nursingchart.loc[nursingchart['nursingchartcelltypevalname'].isin(['GCS Total', 'Verbal', 'Motor', 'Eyes'])]

for stayid in stayids:
    # Get records corresponding to one unit stay ID
    df = nursingchart.loc[nursingchart['patientunitstayid'] == stayid]

    # Write to CSV file. nurseCharting is too large to handle in full
    df.to_csv(os.path.join(args.path, 'nurseChartingStays', str(stayid) + '.csv'))

    del df

del nursingchart
'''

# # Collecting features. 
# gcs_features = ['GCS Total', 'Verbal', 'Eyes', 'Motor']
# # Typical values for imputation, from Benchmarking ML algorithms paper.
# impute_values = dict(zip(gcs_features, [15, 5, 4, 6]))

# with progressbar.ProgressBar(max_value = len(stayids), widgets=widgets) as bar:
#     for i_stayid, stayid in enumerate(stayids):
#         df = pd.read_csv(os.path.join(args.path, 'nurseChartingStays', str(stayid) + '.csv'))

#         if df.shape[0] != 0:
#             # Sort by offset, thereby arranging in time series order
#             df = df.sort_values(by='nursingchartoffset')
#             offsets = df['nursingchartoffset'].unique()

#             out_df = pd.DataFrame(offsets, columns=['nursingchartoffset'])
#             for feat in gcs_features:
#                 out_df.insert(out_df.shape[1], feat, np.nan, allow_duplicates=False)

#             offset_groups = df.groupby('nursingchartoffset')
#             i_offset = 0
#             for offset, group in offset_groups:
#                 avail_feats = group['nursingchartcelltypevalname'].unique()
#                 for feat in gcs_features:
#                     if feat in avail_feats:
#                         out_df.loc[i_offset, feat] = group.loc[group['nursingchartcelltypevalname'] == feat, 'nursingchartvalue'].to_numpy()[0]
#                 i_offset += 1 

#             out_df['nursingchartoffset'] = (out_df['nursingchartoffset']/60).astype('int')

#             # Impute values within offset by replacing NaN with mean over each column.
#             out_df.groupby('nursingchartoffset').apply(lambda x: x.fillna(x.mean()))
#             # For each offset, only choose last value.
#             out_df.drop_duplicates('nursingchartoffset', keep='last', inplace=True)
#             # Impute missing values with "typical values"
#             out_df.fillna(value=impute_values, inplace=True)

#         else:
#             out_df = pd.DataFrame(columns=['nursingchartoffset'])

#         out_df.to_csv(os.path.join(args.path, 'gcsFeatures', str(stayid) + '.csv'), index=False)

#         del df, out_df
#         bar.update(i_stayid, StayID=stayid)

