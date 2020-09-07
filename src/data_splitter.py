#dependencies for helper functions/classes
import pandas as pd
import pyarrow.parquet as pq
from typing import NamedTuple
import random
import numpy as np

#sklearn for preprocessing the data and train-test split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

seed = 7

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
            # print(dummies_df.columns.tolist())
            self.Y = dummies_df.values

        if problem_type == "regression":
            self.Y = self.samples["Avg_age"].values

        return self.Y

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

    def Huber(self, yHat, y, delta=1.):
        return np.where(np.abs(y - yHat) < delta, .5 * (y - yHat) ** 2, delta * (np.abs(y - yHat) - 0.5 * delta))

    def transform_to_probas(self, age_intervals):
        class_names = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
        res = []
        for a in age_intervals:
            non_zero_index = class_names.index(a)
            res.append([0 if i != non_zero_index else 1 for i in range(len(class_names))])
        return np.array(res)

    def transform_to_interval(self, age_probas):
        class_names = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
        return np.array(list(map(lambda p: class_names[np.argmax(p)], age_probas)))




class ClientGenerator:
    '''
    Class which will generate clients.
    Return: a dict with client names as keys, and values in the form of a tuple (X, Y)
    '''

    def balanced_sample_maker(self, X, y, sample_size, random_seed=None):
        """ return a balanced data set by sampling all classes with sample_size
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
            balanced_copy_idx += over_sample_idx
        np.random.shuffle(balanced_copy_idx)

        return (X[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx)

    def label_encode(self, y):
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        return y

    def create_clients(self, x_list, label_list, num_clients=5, initial='client'):
        ''' return: a dictionary with keys clients' names and value as
                    data shards - tuple of input and label lists.
            args:
                x_list: a list of numpy arrays of training samples
                label_list:a list of binarized labels for each class
                num_client: number of fedrated members (clients)
                initials: the clients'name prefix, e.g, clients_1

        '''

        # create a list of client names
        client_names = ['{}_h{}'.format(initial, i + 1) for i in range(num_clients)]

        # randomize the data
        data = list(zip(x_list, label_list))
        random.shuffle(data)

        # shard data and place at each client
        size = len(data) // num_clients
        shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

        # number of clients must equal number of shards
        assert (len(shards) == len(client_names))

        return {client_names[i]: shards[i] for i in range(len(client_names))}


    def non_iid_x(self, x_list, label_list, x=1, num_intraclass_clients=10):
        ''' creates x non_IID clients
        args:
            x_list: python list of data points
            label_list: python list of labels
            x: none IID severity, 1 means each client will only have one class of data
            num_intraclass_client: number of sub-client to be created from each none IID class,
            e.g for x=1, we could create 10 further clients by splitting each class into 10

        return - dictionary
            keys - clients's name,
            value - client's non iid 1 data shard (as tuple list of images and labels) '''

        non_iid_x_clients = dict()

        # create unique label list and shuffle
        unique_labels = np.unique(np.array(label_list))
        random.shuffle(unique_labels)

        # create sub label lists based on x
        sub_lab_list = [unique_labels[i:i + x] for i in range(0, len(unique_labels), x)]

        for item in sub_lab_list:
            class_data = [(image, label) for (image, label) in zip(x_list, label_list) if label in item]

            # decouple tuple list into seperate input and label lists
            images, labels = zip(*class_data)

            # create formated client initials
            initial = ''
            for lab in item:
                initial = initial + lab + '_'

            # create num_intraclass_clients clients from the class
            intraclass_clients = self.iid_clients(list(images), list(labels), num_intraclass_clients, initial)

            # append intraclass clients to main clients'dict
            non_iid_x_clients.update(intraclass_clients)

        return non_iid_x_clients