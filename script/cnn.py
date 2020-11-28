import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from collections import namedtuple


import datetime
# argument parser
parser = argparse.ArgumentParser(description= 'Data Mining' )
parser.add_argument( '--batch-size' , type=int, default= 50 , help= 'Number of samples per mini-batch' )
parser.add_argument( '--epochs' , type=int, default= 12, help= 'Number of epoch to train' )
parser.add_argument( '--lr' , type=float, default= 0.02 , help= 'Learning rate' )
parser.add_argument( '--enable_cuda' , type=int, default= 1 , help= 'Enable Training on GPU ' )
parser.add_argument( '--loss_func' , type=int, default= 0 , help= 'Select Loss function 0-Crossentropy 1-Adam ' )
parser.add_argument( '--kernel_sz' , type=int, default= 3 , help= 'Size of Kernel' )
args = parser.parse_args()



if args.enable_cuda:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#print(torch.cuda.device_count())
	#print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
	device="cpu"

print("device_type:",device)


# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
loss_func = args.loss_func
reg_lambda = args.reg_lambda
kernel_sz = args.kernel_sz
select_dset = args.select_dset
test_num = args.test_num
num_bits = args.num_bits
str_loss = "ADAM"

print("Batch_size:",batch_size);
print("num_epochs:",num_epochs);
print("learning_rate:",learning_rate);
print("Loss_Func:",loss_func);
print("reg_lambda:",reg_lambda);

current_time = str(datetime.datetime.now().timestamp())
#log_file=test_type+"bs:"+str(batch_size)+"n_ep:"+str(num_epochs)+"lr:"+str(learning_rate)+"Regc:"+str(reg_lambda)+"time:"+current_time
#log_file="test_num:"+str(test_num)+"dset:"+str(select_dset)
log_file="batch_size:"+str(batch_size)+"learning_rate:"+str(learning_rate)+"2std"
print(log_file)
tb = SummaryWriter(comment=log_file)

def read_data(file_path):

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan
 


# MNIST Dataset (Images and Labels)
if select_dset==0:
	train_dataset = dsets.MNIST(root = './data' ,
	train = True ,
	transform = transforms.ToTensor(),
	download = True )
	
	test_dataset = dsets.MNIST(root = './data' ,
	train = False ,
	transform = transforms.ToTensor())
else:

	train_dataset = dsets.FashionMNIST(root ='./data',
	        train = True,
	        transform = transforms.ToTensor(),
	        download = True)
	
	test_dataset = dsets.FashionMNIST(root ='./data',
	        train = False,
	        transform = transforms.ToTensor())
# Dataset Loader (Input Pipeline)
dataset_size = (len(train_dataset))
#indices = list(range(dataset_size))
#ft_split = int(np.floor(0.5 * dataset_size))
#train_indice,ft_indice=indices[:ft_split],indices[ft_split:]
#print(train_indice)
#print(ft_indice)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
batch_size = batch_size,
shuffle = True )

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
batch_size = batch_size,
shuffle = False )

print("Lenght of train dataset is ", len(train_dataset))
print("Lenght of test dataset is ", len(test_dataset))
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

class MyConvNet (nn.Module):
	def __init__ (self):
		super(MyConvNet, self).__init__()

		#cnn1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
		self.conv1 = nn.Conv1d( 20 , 32 , kernel_size= 3 , stride= 1 )
#		self.bn1 = nn.BatchNorm2d( 16 )
		self.act1 = nn.ReLU(inplace= True )
		self.pool1 = nn.MaxPool2d(kernel_size= 2 )
		self.conv2 = nn.Conv1d( 32 , 64 , kernel_size= 3 , stride= 1 ,padding= 1 )
	#	self.bn2 = nn.BatchNorm2d( 64 )
		self.act2 = nn.ReLU(inplace= True )
		self.pool2 = nn.MaxPool2d(kernel_size= 2 )
		self.lin1 = nn.Linear( 64 , 1 )
	def forward (self, x):
		c1 = self.conv1(x)
		b1 = self.bn1(c1)
		a1 = self.act1(b1)
		p1 = self.pool1(a1)
		c2 = self.conv2(p1)
		b2 = self.bn2(c2)
		a2 = self.act2(b2)
		p2 = self.pool2(a2)
		flt = p2.view(p2.size( 0 ), -1 )
		l1  = self.lin1(flt)
		out = self.lin2(l1)
		return out
model=MyConvNet();

###Below modifiy
model.to(device)

def get_num_correct(pred,label):
  return pred.argmax(dim=1).eq(labels).sum().item()

#Hooks for conv2 layer weights
conv2_layer_wt_list=[]
count=0
def norm_weights(module,input,output):

  global count;
  norm_weight=0;
  for wt in module.weight:
    norm_weight +=torch.norm(abs(wt))
    
  if module.training:
    conv2_layer_wt_list.append(norm_weight)
    tb.add_scalar('Conv2 weight Norm vs Epoch', norm_weight, count)
    count+=1
    #print("norm_weight :%d\n",count)
  #conv2_layer_wt_list.append(norm_weight)
  #tb.add_scalar('Conv2 weight Norm vs Epoch', norm_weight, count)
  #count+=1;
  #print("norm_weight :%d\n",count)


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True)
    def close(self):
        self.hook.remove()

#Hooks
#2
#conv2_handle = model.features.conv2.register_forward_hook(norm_weights)
#3
#activation_conv2 = SaveFeatures(model.features.conv2)
#activation_relu2 = SaveFeatures(model.features.Relu2)
#activation_bnorm2 = SaveFeatures(model.features.bnorm2)
# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
if loss_func==0:
	print("Loss func is SGD")
	optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
else:
	print("Loss func is ADAM")
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

##
train_target = torch.tensor(train['Target'].values.astype(np.float32))
train = torch.tensor(train.drop('Target', axis = 1).values.astype(np.float32)) 
train_tensor = data_utils.TensorDataset(train, train_target) 
train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)









# Training the Model
#model.train()
total=0
correct=0;
# Training the Model
total=0
correct=0;
#pk for epoch in range(num_epochs):
#pk 	total_loss=0;
#pk 	model.train()
#pk 
#pk 	for i, (images, labels) in enumerate(train_loader):
#pk #pk		images = Variable(images.view( -1 , 28 * 28 ))
#pk 		images = images.to(device)
#pk 		labels = labels.to(device)
#pk 		
#pk 		labels = Variable(labels)
#pk # Forward + Backward + Optimize
#pk 		optimizer.zero_grad()
#pk 		outputs = model(images)
#pk 		loss = criterion(outputs, labels)
#pk 		total_loss+= loss.item()
#pk 
#pk 		loss_batch_size = labels.size( 0 )
#pk # (1)
#pk #the penalty will go here as it should be done before back propagating the gradient
#pk 		l1_norm=0
#pk 		a=[]
#pk 		for name,param in model.named_parameters():
#pk 			if 'weight' in name:
#pk 				#print(name)
#pk 				l1_norm += torch.sum(abs(param))
#pk 		loss = loss + reg_lambda*(l1_norm )
#pk 		loss.backward()
#pk # (2)
#pk 		optimizer.step()
#pk # (3)
#pk 		l11_norm=0
#pk 		#average_loss=(float (loss.item()/loss_batch_size))
#pk 		num_image_processed=((epoch*(len(train_dataset)//batch_size))+(i+1))
#pk 	#	print("loss_batch_size",loss_batch_size,"i is",i,"num_image_processed",num_image_processed);
#pk 		tb.add_scalar('Average Train loss per batch',loss.item(),num_image_processed )
#pk 		if ((( i+1)%100) == 0):
#pk 			print( 'Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'% (epoch + 1 , num_epochs, i + 1 ,len(train_dataset) // batch_size, loss.data.item()))
#pk 			
#pk #	tb.add_scalar('Train loss', loss.item(), epoch)
#pk 	#tb.add_histogram('conv2_activation', activation_conv2.features, epoch)
#pk 	#tb.add_histogram('conv2_weight', model.features.conv2.weight, epoch)
#pk 	#tb.add_histogram('Relu2_activation', activation_relu2.features, epoch)
#pk 	#tb.add_histogram('bnorm2_activation', activation_bnorm2.features, epoch)
#pk 
#pk 
#pk # Test the Model
#pk #tb.add_histogram('After Training conv2_activation', activation_conv2.features)
#pk #tb.add_histogram('After Training Relu2_activation', activation_relu2.features)
#pk #tb.add_histogram('After Training bnorm2_activation', activation_bnorm2.features)
#pk 	correct = 0
#pk 	total = 0
#pk 	total_loss = 0
#pk 	for images, labels in train_loader:
#pk 	#	images = Variable(images.view( -1 , 28 * 28 ))
#pk 		images = images.to(device)
#pk 		labels = labels.to(device)
#pk 		outputs = model(images)	
#pk 		loss = criterion(outputs, labels)
#pk 		total_loss+= (loss.item()*images.shape[0])
#pk 		loss_batch_size = labels.size( 0 )
#pk 		_, predicted = torch.max(outputs.data, 1 )
#pk 		total += labels.size( 0 )
#pk 		correct += (predicted == labels).sum()
#pk 		accuracy = ( 100 * (correct.to(dtype=torch.float))/total)
#pk 	print( 'Accuracy of the model on the train images: % d %%' % ( 100 * (correct.to(dtype=torch.float))/total))
#pk 	#print( '0 Accuracy of the model on the 10000 test images: % d %%' % accuracy)
#pk 	tb.add_scalar('Train total_loss', (total_loss/len(train_dataset)), epoch)
#pk 	tb.add_scalar('Train accuracy', accuracy, epoch)
#pk 
#pk 	model.eval()
#pk 	correct = 0
#pk 	total = 0
#pk 	test_loss = 0
#pk 	total_loss = 0
#pk 	total_test_loss = 0
#pk 	for images, labels in test_loader:
#pk 	#	images = Variable(images.view( -1 , 28 * 28 ))
#pk 		images = images.to(device)
#pk 		labels = labels.to(device)
#pk 		outputs = model(images)	
#pk 		test_loss = criterion(outputs, labels)
#pk 		total_loss+= (test_loss.item()*images.shape[0])
#pk 		_, predicted = torch.max(outputs.data, 1 )
#pk 		total += labels.size( 0 )
#pk 		correct += (predicted == labels).sum()
#pk 		accuracy = ( 100 * (correct.to(dtype=torch.float))/total)
#pk 	print( 'Accuracy of the model on the 10000 test images: % d %%' % ( 100 * (correct.to(dtype=torch.float))/total))
#pk 	#print( '0 Accuracy of the model on the 10000 test images: % d %%' % accuracy)
#pk 	tb.add_scalar("Test Accuracy",accuracy,epoch)
#pk 	tb.add_scalar('Test total_loss', (total_loss/len(test_dataset)), epoch)
#pk 
#pk test_accuracy = accuracy
#pk summary(model, input_size=(1, 28,28 ))
#pk print("Training ends here and now fine tuning will start")
#pk PATH = "save_model/best_save_model_4.pt"
#pk torch.save(model, PATH)
#pk #model.train()


#q_model=quantizeModel(model,num_bits);

#pk for num_bits in range(1, 20):
#pk 	model = torch.load(PATH)
#pk 	model.eval()
#pk 	q_model=quantizeModel(model,num_bits);
#pk 	correct = 0
#pk 	total = 0
#pk 	test_loss = 0
#pk 	total_loss = 0
#pk 	total_test_loss = 0
#pk 	for images, labels in test_loader:
#pk 	#	images = Variable(images.view( -1 , 28 * 28 ))
#pk 		images = images.to(device)
#pk 		labels = labels.to(device)
#pk 		outputs = q_model(images)	
#pk 		test_loss = criterion(outputs, labels)
#pk 		total_loss+= (test_loss.item()*images.shape[0])
#pk 		_, predicted = torch.max(outputs.data, 1 )
#pk 		total += labels.size( 0 )
#pk 		correct += (predicted == labels).sum()
#pk 		accuracy = ( 100 * (correct.to(dtype=torch.float))/total)
#pk 	tb.add_scalar("Test Accuracy vs quant_bits",accuracy,num_bits)
#pk 	print( 'Accuracy of the model on the 10000 test images: % d %%' % ( 100 * (correct.to(dtype=torch.float))/total))
#pk 	test_accuracy_aq = accuracy
#pk 	print( ' Accuracy of the model on the 10000 test images before quantization: % d %%' % test_accuracy)
#pk 	print( ' Accuracy of the model on the 10000 test images after quantization: % d %%' % test_accuracy_aq)


def min_max(x,cnt=2): return (x.mean()+(cnt*x.std())), (x.mean()-(cnt*x.std()))
#part 2 draw the histograms
PATH = "save_model/best_save_model_1.pt"
model = torch.load(PATH)
model.eval()
tb.add_histogram('Before clamping conv1_weight', model.conv1.weight,bins=256)
tb.add_histogram('Before clamping conv2_weight', model.conv2.weight,bins=256)
tb.add_histogram('Before clamping lin1_weight', model.lin1.weight,bins=256)
tb.add_histogram('Before clamping lin2_weight', model.lin2.weight,bins=256)
for name, layer in enumerate(model.modules()):
	if  isinstance(layer, nn.Conv2d):
		print(layer)
		print('----- model conv Layer max and min',min_max(layer.weight.data))
		clamp_max,clamp_min = min_max(layer.weight.data)
		layer.weight.data = layer.weight.data.clamp(min=clamp_min,max=clamp_max)
	elif  isinstance(layer, nn.Linear):
		print(layer)
		print('----- model linear Layer max and min',min_max(layer.weight.data))
		clamp_max,clamp_min = min_max(layer.weight.data)
		layer.weight.data = layer.weight.data.clamp(min=clamp_min,max=clamp_max)
#
tb.add_histogram('After clamping conv1_weight', model.conv1.weight,bins=256)
tb.add_histogram('After clamping conv2_weight', model.conv2.weight,bins=256)
tb.add_histogram('After clamping lin1_weight', model.lin1.weight,bins=256)
tb.add_histogram('After clamping lin2_weight', model.lin2.weight,bins=256)

q_model=quantizeModel(model,8);
tb.add_histogram('After quantization conv1_weight', q_model.conv1.weight,bins=256)
tb.add_histogram('After quantization conv2_weight', q_model.conv2.weight,bins=256)
tb.add_histogram('After quantization lin1_weight', q_model.lin1.weight,bins=256)
tb.add_histogram('After quantization lin2_weight', q_model.lin2.weight,bins=256)
tb.close()

#for name, layer in enumerate(q_model.modules()):
#	if  isinstance(layer, nn.Conv2d):
#		print(layer)
#		print('-----q model Layer max and min',min_max(layer.weight.data))
#		clamp_max,clamp_min = min_max(layer.weight.data)
#	elif  isinstance(layer, nn.Linear):
#		print(layer)
#		print('----- model Layer Mean and std',min_max(layer.weight.data))
#		clamp_max,clamp_min = min_max(layer.weight.data)

correct = 0
total = 0
test_loss = 0
total_loss = 0
total_test_loss = 0
for images, labels in test_loader:
#	images = Variable(images.view( -1 , 28 * 28 ))
	images = images.to(device)
	labels = labels.to(device)
	outputs = q_model(images)	
	test_loss = criterion(outputs, labels)
	total_loss+= (test_loss.item()*images.shape[0])
	_, predicted = torch.max(outputs.data, 1 )
	total += labels.size( 0 )
	correct += (predicted == labels).sum()
	accuracy = ( 100 * (correct.to(dtype=torch.float))/total)
#tb.add_scalar("Test Accuracy vs quant_bits",accuracy,num_bits)
print( 'Accuracy of the model on the 10000 test images: % .4f %%' % ( 100 * (correct.to(dtype=torch.float))/total))
test_accuracy_aq = accuracy
#print( ' Accuracy of the model on the 10000 test images before quantization: % d %%' % test_accuracy)
print( ' Accuracy of the model on the 10000 test images after quantization: % .4f %%' % test_accuracy_aq)



class ShelterOutcomeDataset(Dataset):
    def __init__(self, X, Y, emb_cols):
        X = X.copy()
        self.X1 = X.loc[:,emb_cols].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=emb_cols).copy().values.astype(np.float32) #numerical columns
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

train_ds = ShelterOutcomeDataset(X_train, y_train, emb_cols)
valid_ds = ShelterOutcomeDataset(X_val, y_val, emb_cols)

batch_size = 1000
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)

#categorical embedding for columns having more than two values
emb_c = {n: len(col.cat.categories) for n,col in X.items() if len(col.cat.categories) > 2}
emb_cols = emb_c.keys() # names of columns chosen for embedding
emb_szs = [(c, min(50, (c+1)//2)) for _,c in emb_c.items()] #embedding sizes for the chosen columns


class ShelterOutcomeModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 5)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)
        

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x