#!/usr/bin/env python3
import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats


class Logger:
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = open(log_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to dataset', type=str, default='data/eicu_features.csv')
    parser.add_argument('--model', help='Model to train', type=str, default='SVR')
    parser.add_argument('--log_file', help='log file name', type=str, default='logs/SVR.log')
    args = parser.parse_args()
    return args

class TargetEncoder():
    def __init__(self):
        self.category_maps = {}
        return

    def keys(self):
        return self.category_maps.keys()

    def fit(self, X, y, keys):
        if type(keys) != list:
            keys = [keys]

        for key in keys:
            print("Fitting column {}".format(key))
            category_map = {}
            for category, group in X.groupby(key, as_index=False):
                category_map[category] = y.loc[y.index.isin(group.index)].mean()
            category_map[''] = y.mean()
            self.category_maps[key] = category_map

    def transform(self, X):
        retX = X.copy()
        for key in retX.keys():
            if key in self.category_maps:
                retX[key] = retX[key].map(lambda x: self.category_maps[key][x] if x in self.category_maps[key] else self.category_maps[key][''])
        
        return retX

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    srcc = scipy.stats.spearmanr(y_true, y_pred)[0]
    plcc = scipy.stats.pearsonr(y_true, y_pred)[0]
    return [mae, mse, r2, srcc, plcc]

def formatted_print(metrics_train, metrics_test):
    print('======================================================')
    print(f'MAE_train :  {metrics_train[0]:6.6}')
    print(f'MSE_train :  {metrics_train[1]:6.6}')
    print(f'R2_train  :  {metrics_train[2]:6.6}')
    print(f'SRCC_train:  {metrics_train[3]:6.6}')
    print(f'PLCC_train:  {metrics_train[4]:6.6}')
    print('======================================================')
    print(f'MAE_test  :  {metrics_test[0]:6.6}')
    print(f'MSE_test  :  {metrics_test[1]:6.6}')
    print(f'R2_test   :  {metrics_test[2]:6.6}')
    print(f'SRCC_test :  {metrics_test[3]:6.6}')
    print(f'PLCC_test :  {metrics_test[4]:6.6}')
    print('======================================================')

def main(args):
    # read dataset
    print('Reading dataset...')
    df = pd.read_csv(args.dataset_path)
    df.drop(columns=['Unnamed: 0', 'unitdischargeoffset', 'uniquepid', 'hospitaldischargestatus', 'unitdischargestatus'], inplace=True)
    df.set_index('patientunitstayid', inplace=True)
    y = df['rlos']
    X = df.drop(columns=['rlos'])
    del df
    assert X.shape[0] == y.shape[0]
    
    # train test split
    stayids = X.index.unique()
    train_idx, test_idx = train_test_split(stayids, test_size=0.2, random_state=0)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # apply targetencoder to category features
    print('Applying targetencoder...')
    t_start = time.time()
    target_enc = TargetEncoder()
    cat_feature_keys = ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal']
    target_enc.fit(X_train, y_train, cat_feature_keys)
    X_train = target_enc.transform(X_train)
    X_test = target_enc.transform(X_test)

    # imputer 
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X_train)
    X_train = imp_mean.transform(X_train)
    X_test = imp_mean.transform(X_test)

    # scaler 
    print('Fitting MinMaxScaler...')
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Training using gridsearchCV
    if args.model == 'RIDGE':
        from sklearn.linear_model import Ridge
        param_grid = {'alpha': np.logspace(-3, 3, 10)}
        grid = RandomizedSearchCV(Ridge(), param_grid, n_jobs=1, cv=5, verbose=1)  # n_jobs=1 preventing OOM
    elif args.model == 'LASSO':
        from sklearn.linear_model import Lasso
        param_grid = {'alpha': np.logspace(-5, 0, 10)}
        grid = RandomizedSearchCV(Lasso(), param_grid, n_jobs=1, cv=5, verbose=1)
    elif args.model == 'SVR':
        from sklearn.svm import SVR
        param_grid = {'C': np.logspace(1, 10, 10, base=2),
                      'gamma': np.logspace(-8, 1, 10, base=2)}
        grid = RandomizedSearchCV(SVR(), param_grid, n_jobs=1, cv=5, verbose=1)
    elif args.model == 'RFR':
        from sklearn.ensemble import RandomForestRegressor
        param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [3, 4, 5, 6, 7, 9, 11, -1],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 5],
                'bootstrap': [True, False]}
        grid = RandomizedSearchCV(RandomForestRegressor(), param_grid, n_jobs=1, cv=5, verbose=1)
    elif args.model == 'XGB':
        from xgboost import XGBRegressor
        param_grid = {'max_depth': range(3,12),
                'min_child_weight': range(1,10),
                'gamma': list([i/10.0 for i in range(0,5)]),
                'subsample': list([i/10.0 for i in range(6,10)]),
                'colsample_bytree': list([i/10.0 for i in range(6,10)]),
                'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
        grid = RandomizedSearchCV(XGBRegressor(objective ='reg:squarederror'), param_grid, n_jobs=1, cv=5, verbose=1)
    elif args.model == 'LGBM': 
        from lightgbm import LGBMRegressor
        param_grid = {'num_leaves': [7, 15, 31, 61, 81, 127],
                    'max_depth': [3, 4, 5, 6, 7, 9, 11, -1],
                   'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                   'n_estimators': [100, 200, 300, 400, 500],
                   'boosting_type': ['gbdt', 'dart'],
                   'class_weight': [None, 'balanced'],
                   'min_child_samples': [10, 20, 40, 60, 80, 100, 200],
                #    'bagging_freq': [0, 3, 9, 11, 15, 17, 23, 31],
                   'subsample': [0.5, 0.7, 0.8, 0.9, 1.0],
                   'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10, 100],
                   'reg_lambda': [1e-5, 1e-2, 0.1, 1, 10, 100],
                #    'objective': [None, 'mse', 'mae', 'huber'],
                   }
        grid = RandomizedSearchCV(LGBMRegressor(), param_grid, n_jobs=1, verbose=1)
    
    # fit grid search CV
    print('Hyperparameter tuning on training set...')
    grid.fit(X_train, y_train)
    best_params = grid.best_params_

    # fit and test final model
    if args.model == 'RIDGE':
        regressor = Ridge(**best_params)
    elif args.model == 'LASSO':
        regressor = Lasso(**best_params)
    elif args.model == 'SVR':
        regressor = SVR(**best_params)
    elif args.model == 'RFR':
        regressor = RandomForestRegressor(**best_params)
    elif args.model == 'XGB':
        regressor = XGBRegressor(objective ='reg:squarederror', **best_params)
    elif args.model == 'LGBM': 
        regressor = LGBMRegressor(**best_params)
    print('Training final model with opt parameter...')
    t_start = time.time()
    regressor.fit(X_train, y_train)
    print(f"Training time: {time.time() - t_start} seconds.")
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    # compute metrics 
    metrics_train = compute_metrics(y_train, y_train_pred)
    metrics_test = compute_metrics(y_test, y_test_pred)

    formatted_print(metrics_train, metrics_test)

if __name__ == '__main__':
    args = arg_parser()
    sys.stdout = Logger(args.log_file)
    print(args)
    main(args)