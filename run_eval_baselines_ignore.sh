#!/bin/bash

MODELS=(
    'RIDGE'
    'LASSO'
    'SVR'
    'RFR'
    'XGB'
    'LGBM'
    'XGB'
)
PATH='data/eicu_features.csv'

for m in "${MODELS[@]}"
do
    log_file="logs/${m}_ignore_time_series.log"
    cmd="~/anaconda3/bin/python eval_baselines.py"
    cmd+=" --dataset_path ${PATH}"
    cmd+=" --model ${m}"
    cmd+=" --log_file ${log_file}"
    cmd+=" --ignore_time_series"

    echo "${cmd}"
    eval ${cmd}
done