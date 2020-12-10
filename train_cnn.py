import torch
from rnn_utils import *
import os
import argparse
import progressbar
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Code to train RNN and intepretable RNN models')
parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--model', help='Model to train', type=str, required=True)
parser.add_argument('--epochs', help='Number of epochs for which to train the model', type=int, default=10)
# parser.add_argument('--batch_size', help='Number of examples to use per update', type=int, default=10)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.add_argument('--reverse_input', help='Flag to reverse input', action='store_true')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
assert os.path.exists(args.path), 'Path to dataset does not exist'
assert args.model in models_dict, 'Invalid choice of model'

batch_size = 100
ModelClass = models_dict[args.model]

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('Error')
            ]

args.batch_size = 100
embedding_size = 40
hidden_size = 40

def imputer(dataframe, strategy='zero'):
    normal_values = {'Eyes': 4, 'GCS Total': 15, 'Heart Rate': 86, 'Motor': 6, 'Invasive BP Diastolic': 56,
                     'Invasive BP Systolic': 118, 'O2 Saturation': 98, 'Respiratory Rate': 19,
                     'Verbal': 5, 'glucose': 128, 'admissionweight': 81, 'Temperature (C)': 36,
                     'admissionheight': 170, "MAP (mmHg)": 77, "pH": 7.4, "FiO2": 0.21}

    if strategy not in ['zero', 'back', 'forward', 'normal']:
        raise ValueError("impute strategy is invalid")
    df = dataframe
    if strategy in ['zero', 'back', 'forward', 'normal']:
        if strategy == 'zero':
            df.fillna(value=0, inplace=True)
        elif strategy == 'back':
            df.fillna(method='bfill', inplace=True)
        elif strategy == 'forward':
            df.fillna(method='ffill', inplace=True)
        elif strategy == 'normal':
            df.fillna(value=normal_values, inplace=True)
        if df.isna().sum().any():
            df.fillna(value=normal_values, inplace=True)
        return df



df = pd.read_csv('eicu_features.csv')
#df = imputer(df, strategy='normal')
df.drop(columns=['Unnamed: 0', 'unitdischargeoffset', 'uniquepid', 'hospitaldischargestatus', 'unitdischargestatus'], inplace=True)
df.set_index('patientunitstayid', inplace=True)
y = df['rlos']
print(y)
X = df.drop(columns=['rlos'])
del df

stayids = X.index.unique()
train_ids, test_ids = train_test_split(stayids, test_size=0.2, random_state=0)
#stayids = train_ids if mode == 'train' else test_ids
#n_ids = len(stayids)

X_train = X.loc[train_ids]
y_train = y.loc[train_ids]

X_test = X.loc[test_ids]
y_test = y.loc[test_ids]
print("Test is ")
print(y_test)

del X
del y

#if not os.path.exists('models'):
#    os.mkdir('models')

#encoder_path = os.path.join('models', 'targetencoder.pkl')
#if os.path.exists(encoder_path):
#    encoder = pkl.load(open(encoder_path, 'rb'))
    
print('Fitting Target Encoder')
cat_colums = ['Eyes','GCS Total','ethnicity','gender','Verbal','Motor','apacheadmissiondx']
enc = TargetEncoder()
# transform the datasets

enc.fit(X_train, y_train, [ 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
X_train_enc = enc.transform(X_train )
X_test_enc = enc.transform(X_test)
print('Fitting MinMaxScaler')
scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
scaler.fit(X_train_enc)
X_train_s = scaler.transform(X_train_enc)
X_test_s = scaler.transform(X_test_enc)
y_train=y_train.values
print("After")
print(y_train)

y_test=y_test.values
X_train_s=X_train_s.astype(float)
y_train=y_train.astype(float)

X_test_s=X_test_s.astype(float)
y_test=y_test.astype(float)
X_test_s=torch.tensor(X_test_s,dtype=torch.float32)
print("shape :",X_test_s.shape)
X_test_s_tensor = torch.Tensor(X_test_s).reshape(X_test_s.shape[0],X_test_s.shape[1], 1)
print("shape :",X_test_s_tensor.shape)
y_test_tensor=torch.tensor(y_test.reshape(y_test.shape[0], 1))

X_train_s=torch.tensor(X_train_s,dtype=torch.float32)
X_train_s_tensor = torch.Tensor(X_train_s).reshape(X_train_s.shape[0],X_train_s.shape[1], 1)
#y_train_tensor=torch.tensor(y_train)
y_train_tensor=torch.tensor(y_train.reshape(y_train.shape[0], 1))





#Y
train_tensor = torch.utils.data.TensorDataset(X_train_s_tensor, y_train_tensor) 
train_loader = torch.utils.data.DataLoader(train_tensor,batch_size=batch_size,shuffle=True)

test_tensor = torch.utils.data.TensorDataset(X_test_s_tensor, y_test_tensor) 
test_loader = torch.utils.data.DataLoader(test_tensor,batch_size=batch_size,shuffle=False)

print("SuccessfuL")
data_generator = DataGenerator(args.path, args.batch_size, mode='train', use_cuda=args.use_cuda)
#model = ModelClass(21, embedding_size, hidden_size)
model = ModelClass(21 )
if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    tot_loss = 0.0

    print("Epoch {}/{}".format(epoch+1, args.epochs))
    with progressbar.ProgressBar(max_value = 2500000, widgets=widgets) as bar:
        #for i in range(data_generator.steps_per_epoch):
        for i, (xs, ys) in enumerate(train_loader):
            #xs, ys = data_generator.next()
            #print("xs_shape is :" ,len(xs))
            #print("ys_shape is :",len(ys))
            #print("xs_shape is :" ,xs)
            #print("ys_shape is :",ys)
            #print("xs_shape is :" ,np.asarray(xs).shape)
            #print("ys_shape is :" ,np.asarray(ys).shape)
            
            y_preds = []
            loss = 0.0
            for x1,y1 in zip(xs, ys):
                x=xs
                y=ys
                y=y.cuda()
                x=x.cuda()
                #print("y_shape is :" ,y.shape)
                #print("x_shape is :" ,x.shape)
                if args.reverse_input:
                    x = torch.flip(xs, (1,))
                y_hat = model.forward(x)
                if args.reverse_input:
                    y_hat = torch.flip(y_hat, (1,))

                loss += torch.mean((y - y_hat)**2) # MSE
                y_preds.append(y_hat)
            loss /= batch_size

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.detach().item()

            bar.update(i, Error=tot_loss/(batch_size*(i+1)))

save_model(model, 'models', {'embedding_size': embedding_size, 'hidden_size': hidden_size, 'lr': args.lr, 'epochs':args.epochs, 'batch_size':batch_size, 'reversed':args.reverse_input})

