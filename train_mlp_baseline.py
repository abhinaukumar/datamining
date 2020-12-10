import torch
import numpy as np
import pandas as pd
from nn_utils import TargetEncoder, save_model
from nn_models import models_dict
import os
import argparse
import progressbar
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Code to train RNN and intepretable RNN models')
parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--model', help='Model to train', type=str, required=True)
parser.add_argument('--epochs', help='Number of epochs for which to train the model', type=int, default=10)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
assert os.path.exists(args.path), 'Path to dataset does not exist'
assert args.model in models_dict, 'Invalid choice of model'

ModelClass = models_dict[args.model]

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('Error')
            ]

batch_size = 100
embedding_size = 40
hidden_size = 40

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

model = ModelClass(21)
if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    tot_loss = 0.0

    print("Epoch {}/{}".format(epoch+1, args.epochs))
    with progressbar.ProgressBar(max_value=2500000, widgets=widgets) as bar:
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            y_hat = model.forward(x)

            loss = torch.mean((y - y_hat)**2)  # MSE

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.detach().item()

            bar.update(i, Error=tot_loss/(i+1))

save_model(model, 'models', {'embedding_size': embedding_size, 'hidden_size': hidden_size, 'lr': args.lr, 'epochs': args.epochs, 'batch_size': batch_size})
