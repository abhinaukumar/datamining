import os

import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Code to preprocess data from the eICU database')
parser.add_argument('--path', help='Path to eICU database', required=True, type=str)
args = parser.parse_arguments()

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

# Collecting features. (None of the code below this point has been tested)
gcs_features = ['GCS Total', 'Verbal', 'Eyes', 'Total']
for stayid in stayids:
    # Sort by offset, thereby arranging in time series order
    df = df.sort_values(by='nursingchartoffset')
    offsets = df['nursingchartoffset'].unique()
    out_df = df.copy()
    out_df[gcs_features] = np.nan

    # If value is given, set 
    for feat in gcs_features:
        out_df.loc[out['nursingchartcelltypevalname'] == feat, feat] = out['nursingchartvalue'][out['nursingchartcelltypevalname'] == feat].copy()


