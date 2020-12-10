#!/bin/bash

MODELS=(
    'RIDGE'
    'LASSO'
    'DT'
    'SVR'
    'RFR'
    'XGB'
    'LGBM'
)
PATH='data/eicu_features.csv'

for m in "${MODELS[@]}"
do
    log_file="logs/${m}_last.log"
    cmd="~/anaconda3/bin/python eval_baselines.py"
    cmd+=" --dataset_path ${PATH}"
    cmd+=" --model ${m}"
    cmd+=" --log_file ${log_file}"
    cmd+=" --ignore_time_series"
    cmd+=" --use_last_record"
    cmd+=" --n_jobs 1"
    echo "${cmd}"
    eval ${cmd}
done