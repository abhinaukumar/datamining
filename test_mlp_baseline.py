import torch
import numpy as np
import pandas as pd
import pickle as pkl
from nn_utils import TargetEncoder
import os
from scipy.io import savemat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import argparse
import progressbar

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Code to test RNN and intepretable RNN models')
parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--model_path', help='Path to model', type=str, required=True)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
assert os.path.exists(args.path), 'Path to dataset does not exist'
assert os.path.exists(args.model_path), 'Path to model does not exist'

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ]

batch_size = 100
args.batch_size = 100

df = pd.read_csv('eicu_features.csv')
df.drop(columns=['Unnamed: 0', 'unitdischargeoffset', 'uniquepid', 'hospitaldischargestatus', 'unitdischargestatus'], inplace=True)
df.set_index('patientunitstayid', inplace=True)
y = df['rlos']
print(y)
X = df.drop(columns=['rlos'])
del df

stayids = X.index.unique()
train_ids, test_ids = train_test_split(stayids, test_size=0.2, random_state=0)

X_train = X.loc[train_ids]
y_train = y.loc[train_ids]

X_test = X.loc[test_ids]
y_test = y.loc[test_ids]
print("Test is ")
print(y_test)

del X
del y

print('Fitting Target Encoder')
cat_columns = ['Eyes', 'GCS Total', 'ethnicity', 'gender', 'Verbal', 'Motor', 'apacheadmissiondx']
enc = TargetEncoder()

# Transform the datasets
enc.fit(X_train, y_train, cat_columns)
X_train_enc = enc.transform(X_train)
X_test_enc = enc.transform(X_test)

print('Fitting MinMaxScaler')
scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
scaler.fit(X_train_enc)
X_train_s = scaler.transform(X_train_enc)
X_test_s = scaler.transform(X_test_enc)

y_train = y_train.values
print("After")
print(y_train)

y_test = y_test.values
X_train_s = X_train_s.astype(float)
y_train = y_train.astype(float)

X_test_s = X_test_s.astype(float)
y_test = y_test.astype(float)
X_test_s = torch.tensor(X_test_s, dtype=torch.float32)
print("shape :", X_test_s.shape)
X_test_s_tensor = torch.Tensor(X_test_s).reshape(X_test_s.shape[0], X_test_s.shape[1], 1)
print("shape :", X_test_s_tensor.shape)
y_test_tensor = torch.tensor(y_test.reshape(y_test.shape[0], 1))

X_train_s = torch.tensor(X_train_s, dtype=torch.float32)
X_train_s_tensor = torch.Tensor(X_train_s).reshape(X_train_s.shape[0], X_train_s.shape[1], 1)
y_train_tensor = torch.tensor(y_train.reshape(y_train.shape[0], 1))

train_tensor = torch.utils.data.TensorDataset(X_train_s_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

test_tensor = torch.utils.data.TensorDataset(X_test_s_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

print("Successful")
model = pkl.load(open(args.model_path, 'rb'))

if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

with progressbar.ProgressBar(max_value=1000000, widgets=widgets) as bar:
    y_trues = []
    y_preds = []
    for i, (xs, ys) in enumerate(test_loader):
        y_trues.extend([y.squeeze().cpu().detach().numpy() for y in ys])
        x = xs
        y = ys
        y = y.cuda()
        x = x.cuda()
        print("y_shape is :", y.shape)
        print("x_shape is :", x.shape)
        y_hat = model.forward(x, mode='test')
        y_preds.append(y_hat.squeeze().cpu().detach().numpy())

        bar.update(i)

y_trues = np.hstack(y_trues)
y_preds = np.hstack(y_preds)

mae = np.mean(np.abs(y_trues - y_preds))
rmse = np.sqrt(np.mean((y_trues - y_preds)**2))
r2 = r2_score(y_trues, y_preds)

print("MAE: {}".format(mae))
print("RMSE: {}".format(rmse))
print("R2 score: {}".format(r2))

savemat(os.path.join('results', model.__class__.__name__ + '_results.mat'), {'y_trues': y_trues, 'y_preds': y_preds, 'mae': mae, 'rmse': rmse, 'r2_score': r2})
