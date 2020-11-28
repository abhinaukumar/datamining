#!/bin/bash
PATH='data/eicu_features.csv'
MODEL='lstm'


# log_file="logs/${m}.log"
cmd="~/anaconda3/bin/python run_test_rnn.py"
cmd+=" --path ${PATH}"
cmd+=" --model ${MODEL}"
cmd+=" --no_cuda"

echo "${cmd}"
eval ${cmd}