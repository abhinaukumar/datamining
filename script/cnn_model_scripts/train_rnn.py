import torch
from rnn_utils import *
import os
import argparse
import progressbar
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

tb = SummaryWriter(comment="plot")
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

ModelClass = models_dict[args.model]

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('Error')
            ]

args.batch_size = 1
embedding_size = 40
hidden_size = 40

data_generator = DataGenerator(args.path, args.batch_size, mode='train', use_cuda=args.use_cuda)
#model = ModelClass(21, embedding_size, hidden_size)
model = ModelClass(21 )

activation_conv1 = SaveFeatures(model.conv1)
activation_conv2 = SaveFeatures(model.conv2)
activation_conv3 = SaveFeatures(model.conv3)
activation_lin1 = SaveFeatures(model.lin1)
activation_lin2 = SaveFeatures(model.lin2)

if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    tot_loss = 0.0

    print("Epoch {}/{}".format(epoch+1, args.epochs))
    with progressbar.ProgressBar(max_value = data_generator.steps_per_epoch, widgets=widgets) as bar:
        for i in range(data_generator.steps_per_epoch):
            xs, ys = data_generator.next()
            
            y_preds = []
            loss = 0.0
            for x,y in zip(xs, ys):
                if args.reverse_input:
                    x = torch.flip(x, (1,))
                y_hat = model.forward(x)
                if args.reverse_input:
                    y_hat = torch.flip(y_hat, (1,))

                loss += torch.mean((y - y_hat)**2) # MSE
                y_preds.append(y_hat)
            loss /= args.batch_size

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.detach().item()

            bar.update(i, Error=tot_loss/(i+1))
    tb.add_histogram('conv1_activation', activation_conv1.features, epoch)
    tb.add_histogram('conv2_activation', activation_conv2.features, epoch)
    tb.add_histogram('conv3_activation', activation_conv3.features, epoch)
    tb.add_histogram('lin1_activation', activation_lin1.features, epoch)
    tb.add_histogram('lin2_activation', activation_lin2.features, epoch)

save_model(model, 'models', {'embedding_size': embedding_size, 'hidden_size': hidden_size, 'lr': args.lr, 'epochs':args.epochs, 'batch_size':args.batch_size, 'reversed':args.reverse_input})

