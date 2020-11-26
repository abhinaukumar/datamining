import torch
from rnn_utils import *
import os
import argparse
import progressbar

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Code to train RNN and intepretable RNN models')
parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--model_path', help='Path to model', type=str, required=True)
# parser.add_argument('--batch_size', help='Number of examples to use per update', type=int, default=10)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
assert os.path.exists(args.path), 'Path to dataset does not exist'
assert os.path.exists(args.model_path), 'Path to model does not exist'

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ]

args.batch_size = 1

data_generator = DataGenerator(args.path, args.batch_size, mode='test', use_cuda=args.use_cuda)

model = pkl.load(open(args.model_path, 'rb'))

if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

with progressbar.ProgressBar(max_value = data_generator.steps_per_epoch, widgets=widgets) as bar:
    y_trues = []
    y_preds = []
    for i in range(data_generator.steps_per_epoch):
        xs, ys = data_generator.next()
        y_trues.extend([y.squeeze().cpu().detach().numpy() for y in ys]) 
        for x,y in zip(xs, ys):
            y_hat = model.forward(x)
            y_preds.append(y_hat.squeeze().cpu().detach().numpy())

        bar.update(i)

