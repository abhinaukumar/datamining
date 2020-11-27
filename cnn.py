import os
import torch
import pandas as pd
import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
#from sklearn.base import BaseEstimator

#from sklearn.preprocessing import CategoricalEncoder
#from category_encoders import TargetEncoder
from rnn_utils import *

import datetime
VERBOSE=1
# argument parser
parser = argparse.ArgumentParser(description= 'DATA MINING - RLOS' )
parser.add_argument( '--batch_size' , type=int, default= 1 , help= 'Number of samples per mini-batch' )
parser.add_argument( '--epochs' , type=int, default= 20 , help= 'Number of epoch to train' )
parser.add_argument( '--lr' , type=float, default= 0.02 , help= 'Learning rate' )
parser.add_argument( '--enable_cuda' , type=int, default= 1 , help= 'Enable Training on GPU ' )
parser.add_argument( '--kernel_sz' , type=int, default= 1 , help= 'Size of Kernel' )
#parser.add_argument( '--enable_cuda' , action="store_true" , help= 'Run Training on GPU ' )
args = parser.parse_args()


if args.enable_cuda:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#print(torch.cuda.device_count())
	#print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
	device="cpu"

print("device_type:",device)

tb = SummaryWriter(comment="cnn_model")

num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
kernel_sz = args.kernel_sz

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
#enc = ORIG_TargetEncoder(cols=cat_colums).fit(X_train,y_train)

# transform the datasets

#enc = TargetEncoder()
enc.fit(X_train, y_train, [ 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
#enc.fit(X_train, y_train, ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
X_train_enc = enc.transform(X_train )
X_test_enc = enc.transform(X_test)
#pkl.dump(encoder, open(encoder_path, 'wb'))
#scaler_path = os.path.join('models', 'minmaxscaler.pkl')

print('Fitting MinMaxScaler')
scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
scaler.fit(X_train_enc)
X_train_s = scaler.transform(X_train_enc)
X_test_s = scaler.transform(X_test_enc)
#pkl.dump(scaler, open(scaler_path, 'wb'))

#X_train[X.keys()] = scaler.transform(X)
#X[X.keys()] = scaler.transform(X)
#steps_per_epoch = n_ids//batch_size

#pk df = pd.read_csv('eicu_features.csv')
#pk print(df.columns)
#pk 
#pk cat_colums = ['Eyes','GCS Total','ethnicity','gender','Verbal','Motor','apacheadmissiondx']
#pk 
#pk X=df.loc[:, df.columns != 'rlos']
#pk Y=df.loc[:, df.columns == 'rlos']
#pk 
#pk 
#pk #Y=  df['rlos']
#pk ##print(y.columns)
#pk #X=  df.drop(['rlos'],axis=1)
#pk 
#pk for col in cat_colums:
#pk 	X[col].astype('category')
#pk 
#pk #one hot encoding - creating instance of one-hot-encoder
#pk #enc = OneHotEncoder(handle_unknown='ignore')
#pk #for col in cat_colums:
#pk #	enc_df = pd.DataFrame(enc.fit_transform(X[[col]]).toarray())
#pk #	X = X.join(dum_df)
#pk 
#pk for col in cat_colums:
#pk 	dum_df = pd.get_dummies(X, columns=[col], prefix=[str(col)+"_Type_is"] )
#pk 	X=  X.drop([col],axis=1)
#pk 	#print(dum_df.columns)
#pk 	#X = X.join(dum_df)
#pk 	X = (dum_df)
#pk #ifor col in cat_colums:
#pk #	X=  X.drop([col],axis=1)
#pk if VERBOSE==1:
#pk 	print("FINAL:")
#pk X=  X.drop(['offset'],axis=1)
#pk X=  X.drop(['Unnamed: 0'],axis=1)
#pk X=  X.drop(['patientunitstayid'],axis=1)
#pk X=  X.drop(['unitdischargeoffset'],axis=1)
#pk X=  X.drop(['uniquepid'],axis=1)
#pk X=  X.drop(['hospitaldischargestatus'],axis=1)
#pk X=  X.drop(['unitdischargestatus'],axis=1)
#pk #print(X.columns)
#pk #print(X.head)
#pk if VERBOSE==1:
#pk 	for col in X.columns:
#pk 		print(col)
#pk #
#pk #
#pk #X.to_csv( 'out.csv')
#pk X=X[0:].values
#pk Y=Y[0:].values
#pk print(type(X))
#pk print(type(Y))
#pk 
#pk 
#pk #Split the data
#pk X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state=42)
#pk 
#pk 
#pk #Min Max scaling 
#pk min_max_scaler = MinMaxScaler()
#pk X_train_s = min_max_scaler.fit_transform(X_train)
#pk X_test_s = min_max_scaler.transform(X_test)

#X_train_s=X_train_s[0:].values
print(y_train)
y_train=y_train.values
print("After")
print(y_train)

#X_test_s=X_test_s[0:].values
y_test=y_test.values

X_train_s=X_train_s.astype(float)
y_train=y_train.astype(float)

X_test_s=X_test_s.astype(float)
y_test=y_test.astype(float)

X_test_s=torch.tensor(X_test_s,dtype=torch.float32)
X_test_s_tensor = torch.Tensor(X_test_s).reshape(X_test_s.shape[0],X_test_s.shape[1], 1)
y_test_tensor=torch.tensor(y_test)

X_train_s=torch.tensor(X_train_s,dtype=torch.float32)
X_train_s_tensor = torch.Tensor(X_train_s).reshape(X_train_s.shape[0],X_train_s.shape[1], 1)
y_train_tensor=torch.tensor(y_train)





#Y
train_tensor = torch.utils.data.TensorDataset(X_train_s_tensor, y_train_tensor) 
train_loader = torch.utils.data.DataLoader(train_tensor,batch_size=batch_size,shuffle=True)

test_tensor = torch.utils.data.TensorDataset(X_test_s_tensor, y_test_tensor) 
test_loader = torch.utils.data.DataLoader(test_tensor,batch_size=batch_size,shuffle=False)

print("SuccessfuL")
#
#
class MyConvNet (nn.Module):
	def __init__ (self):
		super(MyConvNet, self).__init__()
		#cnn1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
		self.conv1 = nn.Conv1d( 21, 32 , kernel_size= 1 , stride= 1 )
		#self.bnorm1 .add_module( "bnorm1" , nn.BatchNorm2d(4))
		#self.bn1 = nn.BatchNorm1d(1 )
		self.act1 = nn.ReLU(inplace= True )
		self.conv2 = nn.Conv1d( 32 , 64 , kernel_size= 1 , stride= 1 )
		#self.bn2 = nn.BatchNorm1d(1 )
		self.act2 = nn.ReLU(inplace= True )
		self.lin1 = nn.Linear( 64 , 32 )
		self.lin1_relu = nn.ReLU(inplace=False)
		self.lin2 = nn.Linear( 32 , 1 )
	def forward (self, x):
		c1 = self.conv1(x)
		#print("c1 shape is ",c1.shape)
		#b1 = self.bn1(c1)
		a1 = self.act1(c1)
		#print("a1 shape is ",a1.shape)
		#c2 = self.conv2(a1)
		c2 = self.conv2(c1)
		#print("c2 shape is ",c2.shape)
		#b2 = self.bn1(c2)
		a2 = self.act2(c2)
		#print("a2 shape is ",a2.shape)
		#flt = a2.view(a2.size( 0 ), -1 )
		flt = a2.view(c2.size( 0 ), -1 )
		#print("flt shape is ",flt.shape)
		lin1  = self.lin1(flt)
		#print("lin1 shape is ",lin1.shape)
		lin1_relu  = self.lin1_relu(lin1)
		#out  = self.lin2(lin1_relu)
		out  = self.lin2(lin1)
		return out

print("SuccessfuL")
model=MyConvNet() 
model.to(device)
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 


print("SuccessfuL")
# Training the Model
#model.train()
# Training the Model
model.train()
total=0
correct=0;
iteration=0;
for epoch in range(num_epochs):
	total_loss=0;
	for i, (p_rec, act_out) in enumerate(train_loader):
		print("SuccessfuL")
#pk		p_rec = Variable(p_rec.view( -1 , 28 * 28 ))
		p_rec = p_rec.to(device)
		act_out = act_out.to(device)
		
#		act_out = Variable(act_out)
# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = model(p_rec)
		print("Model:done")
		loss = criterion(outputs, act_out.float())
#the penalty will go here as it should be done before back propagating the gradient
		loss.backward()
		total_loss = total_loss + loss.data.item()
# (2)
		optimizer.step()
# (3)
		if ((( i+1)%100) == 0):
			print( 'Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'% (epoch + 1 , num_epochs, i + 1 ,len(train_tensor) // batch_size, loss.data.item()))
		tb.add_scalar('Train loss', loss.item(),iteration)
		iteration=iteration+1
	tb.add_scalar('Total Train loss per epoch', total_loss/(len(train_tensor)),epoch)

model.eval()
correct = 0
total = 0
total_test_loss = 0
for p_rec, act_out in test_loader:
#	p_rec = Variable(p_rec.view( -1 , 28 * 28 ))
	p_rec = p_rec.to(device)
	act_out = act_out.to(device)
	outputs = model(p_rec)	
	loss = criterion(outputs, act_out.float())
	total_test_loss +=loss
	print('Test Loss per batch',loss.data.item())
#print( 'Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'% (epoch + 1 , num_epochs, i + 1 ,len(train_tensor) // batch_size, loss.data.item()))
print('Total Test  Loss  {}'.format(total_test_loss/len(test_tensor)))
