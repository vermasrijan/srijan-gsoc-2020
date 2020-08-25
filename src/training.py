import syft as sy
from syft.grid.public_grid import PublicGridNetwork
import torch as th
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from syft.federated.floptimizer import Optims
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(18420, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 6)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


class Decentralized:

    def epoch_total_size(self, data):
        total = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                total += data[i][j].shape[0]

        return total

    def train_distributed(self, GRID_ADDRESS='0.0.0.0', GRID_PORT='5000', N_EPOCS=20, CLIENTS=None,SAVE_MODEL=False, SAVE_MODEL_PATH='./models'):
        hook = sy.TorchHook(th)

        my_grid = PublicGridNetwork(hook, "http://" + GRID_ADDRESS + ":" + GRID_PORT)
        data = my_grid.search("#X", "#gtex_v8", "#dataset")
        time.sleep(2)
        target = my_grid.search("#Y", "#gtex_v8", "#dataset")
        time.sleep(2)
        data = list(data.values())
        target = list(target.values())

        print(data)
        print("======")
        print(target)

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        if (th.cuda.is_available()):
            th.set_default_tensor_type(th.cuda.FloatTensor)

        model = Net()
        model.to(device)
        workers = ['h{}'.format(i+1) for i in range(CLIENTS)]
        print('WORKERS: ', workers)
        print(1)
        optims = Optims(workers, optim=optim.Adam(params=model.parameters(), lr=0.003))
        # criterion = nn.CrossEntropyLoss()

        for epoch in range(N_EPOCS):
            print(2)
            model.train()
            epoch_total = self.epoch_total_size(data)
            current_epoch_size = 0
            for i in range(len(data)):
                print(3)
                correct = 0
                for j in range(len(data[i])):
                    epoch_loss = 0.0
                    epoch_acc = 0.0

                    current_epoch_size += len(data[i][j])
                    worker = data[i][j].location
                    model.send(worker)
                    time.sleep(4)
                    print(4)

                    # Call the optimizer for the worker using get_optim
                    opt = optims.get_optim(data[i][j].location.id)
                    time.sleep(5)

                    opt.zero_grad()
                    pred = model(data[i][j])
                    time.sleep(5)
                    loss = F.cross_entropy(pred, target[i][j])
                    loss.backward()
                    opt.step()
                    print(5)

                    # statistics
                    # prob = F.softmax(pred, dim=1)
                    top1 = torch.argmax(pred, dim=1)
                    ncorrect = torch.sum(top1 == target[i][j])
                    print(6)

                    # Get back loss
                    loss = loss.get()
                    time.sleep(5)
                    ncorrect = ncorrect.get()
                    time.sleep(5)

                    epoch_loss += loss.item()
                    epoch_acc += ncorrect.item()

                    epoch_loss /= target[i][j].shape[0]
                    epoch_acc /= target[i][j].shape[0]

                    model.get()
                    time.sleep(5)

                    print(
                        'Train Epoch: {} | With {} data |: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f} | Train Acc: {:.3f}'.format(
                            epoch, worker.id, current_epoch_size, epoch_total,
                            100. * current_epoch_size / epoch_total, epoch_loss, epoch_acc))


class Centralized:
    def preprocess(self, datasets, labels):
        # Concatenate
        X, Y = datasets[0], labels[0]
        for i in range(1, len(datasets), 1):
            X = torch.cat((X, datasets[i]), dim=0)
            Y = torch.cat((Y, labels[i]), dim=0)

        return X, Y

    def train_centralized(self, N_EPOCHS=20, datasets=None, labels=None):
        # Create the network, define the criterion and optimizer
        model = Net()
        epochs = N_EPOCHS
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        X, Y = self.preprocess(datasets, labels)

        for e in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            optimizer.zero_grad()

            pred = model(X)
            loss = F.cross_entropy(pred, Y)

            loss.backward()
            optimizer.step()

            # statistics
            # prob = F.softmax(pred, dim=1)
            top1 = torch.argmax(pred, dim=1)
            ncorrect = torch.sum(top1 == Y)

            epoch_loss += loss.item()

            epoch_acc += ncorrect.item()

            epoch_loss /= Y.shape[0]
            epoch_acc /= Y.shape[0]

            print(f"Epoch: {e}", f"Training loss: {epoch_loss}", f" | Training Accuracy: {epoch_acc}")

