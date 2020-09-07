import syft as sy
from syft.grid.public_grid import PublicGridNetwork
import torch as th
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from syft.federated.floptimizer import Optims
import time
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient
import numpy as np

DATA_SEND_TIME = 2
DATA_SEARCH_TIME = 1

class Net(nn.Module):
    '''
    Hyperparameters for centralized / decentralized models
    '''

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
    '''
    Class for decentralized training
    '''


    def epoch_total_size(self, data):
        total = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                total += data[i][j].shape[0]

        return total

    def train_distributed(self, _ports, datasets, labels, GRID_ADDRESS='0.0.0.0', GRID_PORT='5000', N_EPOCHS=20, CLIENTS=None,SAVE_MODEL=False, SAVE_MODEL_PATH='./models'):

        # Send Data
        hook = sy.TorchHook(th)

        # Connect directly to grid nodes
        nodes = ["ws://0.0.0.0:{}".format(i) for i in _ports]

        compute_nodes = []
        for node in nodes:
            # For syft 0.2.8 --> replace DynamicFLClient with DataCentricFLClient
            compute_nodes.append(DataCentricFLClient(hook, node))

        tag_input = []
        tag_label = []

        for i in range(len(compute_nodes)):
            tag_input.append(datasets[i].tag("#X", "#gtex_v8", "#dataset").describe(
                "The input datapoints to the GTEx_V8 dataset."))
            tag_label.append(labels[i].tag("#Y", "#gtex_v8", "#dataset").describe(
                "The input labels to the GTEx_V8 dataset."))

        x_dataset = []
        y_dataset = []

        for i in range(len(compute_nodes)):
            x_dataset.append(tag_input[i].send(compute_nodes[i]))  # First chunk of dataset to h1
            y_dataset.append(tag_label[i].send(compute_nodes[i]))  # First chunk of labels to h1
            time.sleep(DATA_SEND_TIME)

        for i in range(len(compute_nodes)):
            compute_nodes[i].close()

        ###############<Press Enter to continue...>####################
        host_address = ["http://0.0.0.0:{}".format(GRID_PORT), "http://0.0.0.0:{}/connected-nodes".format(GRID_PORT), "http://0.0.0.0:{}/search-available-tags".format(GRID_PORT)] + ["http://0.0.0.0:{}".format(i) for i in _ports]
        print('Go to the following addresses: {}'.format(host_address))
        input('Press Enter to continue...')
        ###############################################################

        my_grid = PublicGridNetwork(hook, "http://" + GRID_ADDRESS + ":" + GRID_PORT)
        data = my_grid.search("#X", "#gtex_v8", "#dataset")
        time.sleep(DATA_SEARCH_TIME)

        target = my_grid.search("#Y", "#gtex_v8", "#dataset")
        time.sleep(DATA_SEARCH_TIME)

        data = list(data.values())
        target = list(target.values())

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        if (th.cuda.is_available()):
            th.set_default_tensor_type(th.cuda.FloatTensor)

        model = Net()
        model.to(device)
        workers = ['h{}'.format(i+1) for i in range(CLIENTS)]
        print('WORKERS: ', workers)
        optims = Optims(workers, optim=optim.Adam(params=model.parameters(), lr=0.003))
        # criterion = nn.CrossEntropyLoss()

        # Metrics dict
        glob_mod_metadata = dict()

        for epoch in range(N_EPOCHS):
            model.train()
            epoch_total = self.epoch_total_size(data)
            current_epoch_size = 0

            for i in range(len(data)):
                correct = 0
                for j in range(len(data[i])):
                    epoch_loss = 0.0
                    epoch_acc = 0.0

                    current_epoch_size += len(data[i][j])
                    worker = data[i][j].location
                    model.send(worker)

                    # Call the optimizer for the worker using get_optim
                    opt = optims.get_optim(data[i][j].location.id)

                    opt.zero_grad()
                    pred = model(data[i][j])

                    loss = F.cross_entropy(pred, target[i][j])
                    loss.backward()
                    opt.step()

                    # statistics
                    # prob = F.softmax(pred, dim=1)
                    top1 = torch.argmax(pred, dim=1)
                    ncorrect = torch.sum(top1 == target[i][j])

                    # Get back loss
                    loss = loss.get()
                    ncorrect = ncorrect.get()

                    epoch_loss += loss.item()
                    epoch_acc += ncorrect.item()

                    epoch_loss /= target[i][j].shape[0]
                    epoch_acc /= target[i][j].shape[0]

                    model.get()

                    print(
                        'Train Epoch: {} | With {} data |: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f} | Train Acc: {:.3f}'.format(
                            epoch, worker.id, current_epoch_size, epoch_total,
                            100. * current_epoch_size / epoch_total, epoch_loss, epoch_acc))

                    # Save metrics in dict
                    glob_mod_metadata['round_{}_{}_results'.format(epoch, worker.id)] = {'accuracy': round(epoch_acc, 4),
                                                                                'loss': round(epoch_loss, 4)}


        return glob_mod_metadata

class Centralized:
    '''
    Class for centralized training
    '''

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

        # Metrics dict
        glob_mod_metadata = dict()

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

            glob_mod_metadata['epoch_{}'.format(e)] = {'accuracy': np.round(epoch_acc, 4), 'loss': np.round(epoch_loss, 4)}

        return glob_mod_metadata

