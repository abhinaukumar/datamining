from rnn_utils import *
import os
import argparse
import progressbar
import time
from Update import *
import copy
from Fed import *
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Code to train RNN and intepretable RNN models')
parser.add_argument('--epochs', help='Number of epochs for which to train the model', type=int, default=10)
parser.add_argument('--batch_size', help='Number of examples to use per update', type=int, default=10)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.set_defaults(use_cuda=True)
args = parser.parse_args()

# widgets = [
#             progressbar.ETA(),
#             progressbar.Bar(),
#             ' ', progressbar.DynamicMessage('Error')
#             ]

args.batch_size = 1
embedding_size = 40
hidden_size = 40

path1 = 'mini_eicu_features_height_1.csv'
path2 = 'mini_eicu_features_height_2.csv'
path3 = 'mini_eicu_features_height_3.csv'
path4 = 'mini_eicu_features_height_4.csv'
path5 = 'mini_eicu_features_height_5.csv'
data_generator1 = DataGenerator(path1, args.batch_size, mode='train', use_cuda=args.use_cuda)
data_generator2 = DataGenerator(path2, args.batch_size, mode='train', use_cuda=args.use_cuda)
data_generator3 = DataGenerator(path3, args.batch_size, mode='train', use_cuda=args.use_cuda)
data_generator4 = DataGenerator(path4, args.batch_size, mode='train', use_cuda=args.use_cuda)
data_generator5 = DataGenerator(path5, args.batch_size, mode='train', use_cuda=args.use_cuda)


data_generator = [data_generator1,data_generator2, data_generator3, data_generator4,data_generator5]

model_glob = LSTMModel(21, embedding_size, hidden_size)
if args.use_cuda:
    model_glob = model_glob.cuda()
    model_glob.tensors_to_cuda()

model_glob.train()
w_glob = model_glob.state_dict()

# training
loss_train = []
lr_base_t0 = time.time()
print(time.localtime(lr_base_t0))
for iter in range(args.epochs):
    loss_locals = []
    w_locals = []
    m = 2
    client = np.random.choice(5, m, replace=False)
    for idx in client:
        dataset_train = data_generator[idx]
        local = LocalUpdate(args=args, dataset=dataset_train)
        w, loss = local.train(net=copy.deepcopy(model_glob))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(loss)

    w_glob = FedAvg(w_locals)
    model_glob.load_state_dict(w_glob)

    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)

lr_base_t1 = time.time()
print(time.localtime(lr_base_t1))

path = 'mini_eicu_features.csv'
data_generator = DataGenerator(path, args.batch_size, mode='train', use_cuda=args.use_cuda)
batch_loss = []
loss = 0.0
for iter in range(data_generator.steps_per_epoch):
    xs, ys = data_generator.next()
    y_preds = []
    for x, y in zip(xs, ys):
        y_hat = model_glob.forward(x)
        loss += torch.mean((y - y_hat) ** 2)  # MSE
        y_preds.append(y_hat)
loss /= data_generator.steps_per_epoch

print('loss: ', loss)

