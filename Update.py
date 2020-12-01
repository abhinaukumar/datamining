from rnn_utils import *


class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.steps_per_epoch = dataset.steps_per_epoch
        self.dataset = dataset


    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        batch_loss = []
        for iter in range(self.steps_per_epoch):
            xs, ys = self.dataset.next()
            y_preds = []
            loss = 0.0
            for x, y in zip(xs, ys):
                y_hat = net.forward(x)
                loss += torch.mean((y - y_hat) ** 2)  # MSE
                y_preds.append(y_hat)
            loss /= self.args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss)
        return net.state_dict(), sum(batch_loss)/len(batch_loss)

