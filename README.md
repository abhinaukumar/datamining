# Data Mining - Analyzing Model Tradeoffs in Predicting Length of Stay (LOS) in eICU Patients

A detailed explanation of our experiments and results can be found in [this blog entry](https://abhinaukumar.github.io/post/eicu-tradeoffs). In this README file, we include instructions on how to run the code.

## Preprocessing the Data
```
python3 preprocess.py --path path-to-database-directory
```

`preprocess.py` creates a compressed csv file in the database directory called eicu_features.csv.gz. This file is used in all subsequent code.

## Training and Testing Decision Trees
```
python3 decision_trees.py
```

## Training and Testing other Baseline Models
Run baseline models on all the records for each patient:

```
bash run_eval_baselines.sh
```

Run baseline models on the first record plus number of offsets:

```
bash run_eval_baselines_first.sh
```

Run baseline models on the last record with time offsets:

```
bash run_eval_baselines_last.sh
```

> Note: if you cannot run it, change the `python` path in bash files

These two scripts respectively include and exclude the time-series nature of the data in the models. The naive model which ignores the time-series nature uses the offset as the 21st feature, while the model that includes time-series information uses the number of offsets.

## Training and Testing MLPs
```
python3 train_mlp_baseline.py --path path-to-database --model cnn_base --epochs number-of-training-epochs
```

```
python3 test_mlp_baseline.py --path path-to-database --model_path path-to-trained-model
```

Note that the MLPs are implemented using pointwise convolutions to effectively reuse models. The difference in training between an MLP and a CNN with kernel size 1 is in how the training examples are sampled.

## Training and Testing CNNs and RNNs
```
python3 train_nn.py --path path-to-database --model name-of-model-to-train [--epochs number-of-training-epochs] [--lr learning-rate] [--no_cuda] [--reverse_input]
```

```
python3 test_nn.py --path path-to-database --model_path path-to-trained-model [--no_cuda] [--reverse_input] [--tag tag-to-add-to-results-filename]
```

The arguments shown in square brackets are optional. `no_cuda` runs code on the CPU, while omitting the flag will default to using CUDA. `reverse_input` reverses the order of records when feeding examples to the model. This flag is only relevant when using RNNs.

The name of the model to train must be one of
1. cnn_k1 - CNN with kernel size = 1
2. cnn_k3 - CNN with kernel size = 3
3. cnn_k15 - CNN with kernel size = 15
4. cnn_kvar - CNN with variable kernel size
5. lstm - RNN (LSTM) model
6. retain - RETAIN model
7. bilstm - Bidirectional RNN (LSTM) model
8. biretain - Bidirectional RETAIN model

## Training and Testing Federated Learning Models
```
python3 federated_learning.py [--epochs number-of-training-epochs] [--lr learning-rate] [--no_cuda] [--reverse_input]
```
The optional flags here serve the same functions as in the case of `train_nn.py`.
