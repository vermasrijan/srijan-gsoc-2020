#dependencies for helper functions/classes
import pandas as pd
import pyarrow.parquet as pq
from typing import NamedTuple
import os.path as path
import os
# import requests
import numpy as np
import random
import yaml
import subprocess

# syft dependencies
import syft as sy

#########<syft==0.2.8>#######################
# # Dynamic FL -->
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

# #Static FL -->
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient

import torch
import pickle
import time
import numpy as np
import torchvision
from torchvision import datasets, transforms

#sklearn for preprocessing the data and train-test split
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error

seed = 7



# import click
#
# @click.command()
# @click.option('--input', default="./data", help='Input for dataset')
# @click.option('--train_type', default="centralized", help='Either centralized or decentralized fashion')
# @click.option("--dataset_size", default="huber", help="")
# @click.option('--split_type', default=5, help='')
# @click.option('--split_size', default=1, help='')
# @click.option('--nodes', default=1, help="")
# @click.option('--ports', default=3, help="species_in_validation")
class Labels(NamedTuple):
    '''
    One-hot labeled data
    '''
    tissue: np.ndarray
    sex: np.ndarray
    age: np.ndarray
    death: np.ndarray


class Genes:
    '''
    Class to load GTEX samples and gene expressions data
    '''

    def __init__(self, samples_path: str = '', expressions_path: str = '', problem_type: str = "classification"):
        self.__set_samples(samples_path)
        self.__set_labels(problem_type)
        if expressions_path != '':
            self.expressions = self.get_expressions(expressions_path)

    def __set_samples(self, sample_path: str) -> pd.DataFrame:
        self.samples: pd.DataFrame = pq.read_table(sample_path).to_pandas()
        self.samples["Death"].fillna(-1.0, inplace=True)
        self.samples: pd.DataFrame = self.samples.set_index("Name")
        self.samples["Sex"].replace([1, 2], ['male', 'female'], inplace=True)
        self.samples["Death"].replace([-1, 0, 1, 2, 3, 4],
                                      ['alive/NA', 'ventilator case', '<10 min.', '<1 hr', '1-24 hr.', '>1 day'],
                                      inplace=True)

        return self.samples

    def __set_labels(self, problem_type: str = "classification") -> Labels:
        self.labels_list = ["Tissue", "Sex", "Age", "Death"]
        self.labels: pd.DataFrame = self.samples[self.labels_list]
        self.drop_list = self.labels_list + ["Subtissue", "Avg_age"]

        if problem_type == "classification":
            dummies_df = pd.get_dummies(self.labels["Age"])
            print(dummies_df.columns.tolist())
            self.Y = dummies_df.values

        if problem_type == "regression":
            self.Y = self.samples["Avg_age"].values

        return self.Y

    def delete_particular_age_examples(self):
        df_series = pd.DataFrame(self.labels["Age"])
        indexes_of_50 = np.where(df_series["Age"] == '50-59')[0].tolist()[300:]
        indexes_of_60 = np.where(df_series["Age"] == '60-69')[0].tolist()[300:]
        indexes_of_20 = np.where(df_series["Age"] == '20-29')[0].tolist()[300:]
        indexes_of_30 = np.where(df_series["Age"] == '30-39')[0].tolist()[300:]
        indexes_of_40 = np.where(df_series["Age"] == '40-49')[0].tolist()[300:]
        indexes_to_delete = indexes_of_50 + indexes_of_60 + indexes_of_20 + indexes_of_30 + indexes_of_40

        return indexes_to_delete

    def sex_output(self, model):
        return Dense(units=self.Y.sex.shape[1], activation='softmax', name='sex_output')(model)

    def tissue_output(self, model):
        return Dense(units=self.Y.tissue.shape[1], activation='softmax', name='tissue_output')(model)

    def death_output(self, model):
        return Dense(units=self.Y.death.shape[1], activation='softmax', name='death_output')(model)

    def age_output(self, model):
        '''
        Created an output layer for the keras mode
        :param model: keras model
        :return: keras Dense layer
        '''
        return Dense(units=self.Y.age.shape[1], activation='softmax', name='age_output')(model)

    def get_expressions(self, expressions_path: str) -> pd.DataFrame:
        '''
        load gene expressions DataFrame
        :param expressions_path: path to file with expressions
        :return: pandas dataframe with expression
        '''
        if expressions_path.endswith(".parquet"):
            return pq.read_table(expressions_path).to_pandas().set_index("Name")
        else:
            separator = "," if expressions_path.endswith(".csv") else "\t"
            return pd.read_csv(expressions_path, sep=separator).set_index("Name")

    def prepare_data(self, normalize_expressions: bool = True) -> np.ndarray:
        '''
        :param normalize_expressions: if keras should normalize gene expressions
        :return: X array to be used as input data by keras
        '''
        data = self.samples.join(self.expressions, on="Name", how="inner")
        ji = data.columns.drop(self.drop_list)
        x = data[ji]

        # adding one-hot-encoded tissues and sex
        x = pd.concat([x, pd.get_dummies(data['Tissue'], prefix='tissue'), pd.get_dummies(data['Sex'], prefix='sex')],
                      axis=1)
        x = x.values

        return normalize(x, axis=0) if normalize_expressions else x

    def get_features_dataframe(self, add_tissues=True):
        data = self.samples.join(self.expressions, on="Name", how="inner")
        ji = data.columns.drop(self.drop_list)
        df = data[ji]
        if add_tissues:
            df = pd.concat(
                [df, pd.get_dummies(data['Tissue'], prefix='tissue'), pd.get_dummies(data['Sex'], prefix='sex')],
                axis=1)
        x = df.values
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_normalized = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)
        return df_normalized

samples_path = 'data/gtex/v8_samples.parquet'
expressions_path = 'data/gtex/v8_expressions.parquet'


def Huber(yHat, y, delta=1.):
    return np.where(np.abs(y - yHat) < delta, .5 * (y - yHat) ** 2, delta * (np.abs(y - yHat) - 0.5 * delta))


def transform_to_probas(age_intervals):
    class_names = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    res = []
    for a in age_intervals:
        non_zero_index = class_names.index(a)
        res.append([0 if i != non_zero_index else 1 for i in range(len(class_names))])
    return np.array(res)


def transform_to_interval(age_probas):
    class_names = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    return np.array(list(map(lambda p: class_names[np.argmax(p)], age_probas)))

genes = Genes(samples_path, expressions_path, problem_type="classification")
X = genes.get_features_dataframe().values
Y = genes.Y

a = transform_to_interval(Y)
unique, counts = np.unique(a, return_counts=True)

b = [np.where(r==1)[0][0] for r in Y]
unique, counts = np.unique(b, return_counts=True)

df_pie = pd.DataFrame(columns = ['age_group','label'])
df_pie['age_group'] = a
df_pie['label'] = b

import numpy as np
def balanced_sample_maker(X, y, sample_size, random_seed=None):
    """ return a balanced data set by sampling all classes with sample_size
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    return (X[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx)


no_samples_to_take = 200
res = balanced_sample_maker(X,np.asarray(df_pie['age_group'].values), no_samples_to_take)

unique, counts = np.unique(res[1], return_counts=True)
print(dict(zip(unique, counts)))

le = LabelEncoder()
le.fit(res[1])
y = le.transform(res[1])


def create_clients(image_list, label_list, num_clients=5, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}

a = create_clients(res[0],y, 2)

datasets = []
labels = []
for key, val in a.items():
    data, label = zip(*a[key])
    label = np.array(label)
    data = np.array(data)
    data, label = np.vstack(data).astype(np.uint8), np.vstack(label).astype(np.uint8)
    label = label.reshape(label.shape[0])

    # Convert numpy array to torch tensors -->
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)

    data = torch.tensor(data, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.int64)

    datasets.append(data)
    labels.append(label)

doc = {'version': '3',
 'services': {'network': {'image': 'openmined/grid-network:v028',
   'environment': ['PORT=5000',
    'SECRET_KEY=ineedtoputasecrethere',
    'DATABASE_URL=sqlite:///databasenetwork.db'],
   'ports': ['5000:5000']}}}

n = 2
_ports = [str(i) for i in range(3000, 3000+n, 1)]

for i in range(n):
    doc['services'].update(
        {'h{}'.format(i+1): {'image': 'openmined/grid-node:v028',
        'environment': ['NODE_ID=h{}'.format(i+1),
        'ADDRESS=http://h{0}:{1}/'.format(i+1, _ports[i]),
        'PORT={}'.format(_ports[i]),
        'NETWORK=http://network:5000',
        'DATABASE_URL=sqlite:///databasenode.db'],
        'depends_on': ["network"],
        'ports': ['{0}:{0}'.format(_ports[i])]}
        })

with open('docker-compose.yml', 'w') as f:
    yaml.dump(doc, f)

cmd = ['docker-compose', 'up', '-d']
print('===========')
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('==<STARTING DOCKER IMAGE>==')
out, error = p.communicate()
print(out)
print('===========')
print(error)
print('===========')
print('DONE!')
time.sleep(15)


hook = sy.TorchHook(torch)

# Connect directly to grid nodes
nodes = ["ws://0.0.0.0:{}".format(i) for i in _ports]

compute_nodes = []
for node in nodes:
    # For syft 0.2.8 --> replace DynamicFLClient with DataCentricFLClient
    compute_nodes.append( DataCentricFLClient(hook, node) )

print(compute_nodes)

tag_input = []
tag_label = []

for i in range(len(compute_nodes)):
    tag_input.append(datasets[i].tag("#X", "#gtex_v8", "#dataset","#balanced").describe("The input datapoints to the GTEx_V8 dataset."))
    tag_label.append(labels[i].tag("#Y", "#gtex_v8", "#dataset","#balanced").describe("The input labels to the GTEx_V8 dataset."))

for i in range(len(compute_nodes)):
    shared_x = tag_input[i].send(compute_nodes[i]) # First chunk of dataset to h1
    shared_y = tag_label[i].send(compute_nodes[i]) # First chunk of labels to h1
    print("X tensor pointer: ", shared_x)
    print("Y tensor pointer: ", shared_y)

time.sleep(40)

for i in range(len(compute_nodes)):
    compute_nodes[i].close()

print("NODES CLOSED!!!!")

import syft as sy
from syft.grid.public_grid import PublicGridNetwork
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from syft.federated.floptimizer import Optims

hook = sy.TorchHook(th)

GRID_ADDRESS = '0.0.0.0'
GRID_PORT = '5000'

my_grid = PublicGridNetwork(hook,"http://" + GRID_ADDRESS + ":" + GRID_PORT)

data = my_grid.search("#X", "#gtex_v8", "#dataset")
target = my_grid.search("#Y", "#gtex_v8", "#dataset")

print(data)
print("======")
print(target)