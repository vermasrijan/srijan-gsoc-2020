from data_splitter import Genes, ClientGenerator
import time
from data_sender import Preprocess
from training import Centralized, Decentralized
import warnings
import numpy as np
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

import click
@click.command()
@click.option('--samples_path', default="data/gtex/v8_samples.parquet", help='Input path for samples')
@click.option('--expressions_path', default="data/gtex/v8_expressions.parquet", help='Input for expressions')
@click.option('--train_type', default="centralized", help='Either centralized or decentralized fashion')
@click.option('--dataset_size', default=600, help="Size of data for training")
@click.option('--split_type', default='balanced', help='balanced / unbalanced / iid / non_iid')
@click.option('--split_size', default=0.8, help='Train / Test Split')
@click.option('--n_epochs', default=2, help='No. of Epochs / Rounds')
@click.option('--metrics_path', default='data/metrics', help="Path to save metrics")
@click.option('--no_of_clients', default=3, help="Clients / Nodes for decentralized training")
@click.option('--tags', default=None, help="Give tags for the data, which is to be sent to the nodes")
@click.option('--node_start_port', default='3000', help="Start port No. for a node")
@click.option('--grid_address', default='0.0.0.0', help="grid address for network")
@click.option('--grid_port', default='5000', help="grid port for network")


def main(samples_path, expressions_path, train_type, dataset_size, no_of_clients, grid_port, grid_address, metrics_path, n_epochs, split_type, split_size, node_start_port, tags):

    print('='*60)
    GRID_ADDRESS = grid_address
    GRID_PORT = grid_port

    if split_type == 'balanced':
        no_samples_to_take = dataset_size // 6

    # Calculate start time
    start_time = time.time()

    print('----<DATA PREPROCESSING STARTED..>----')
    genes = Genes(samples_path, expressions_path, problem_type="classification")
    X = genes.get_features_dataframe().values
    Y = genes.Y

    transformed_genes = genes.transform_to_interval(Y)

    res = ClientGenerator().balanced_sample_maker(X, np.asarray(transformed_genes), no_samples_to_take)

    unique, counts = np.unique(res[1], return_counts=True)

    y = ClientGenerator().label_encode(res[1])

    client_dict = ClientGenerator().create_clients(res[0], y, no_of_clients)

    datasets, labels = Preprocess().tensor_converter(client_dict)

    print('----<STARTED TRAINING IN A {} FASHION..>----'.format(train_type))
    print('DATASET SIZE: {}'.format(dataset_size))

    if train_type == 'decentralized':

        print('TOTAL CLIENTS: {}'.format(no_of_clients))
        print('DATAPOINTS WITH EACH CLIENT: ')

        for key, val in client_dict.items():
            data, label = zip(*client_dict[key])
            unique, counts = np.unique(label, return_counts=True)
            print('{}: {} ; Label Count: {}'.format(key, len(label), dict(zip(unique, counts))))

        _ports = Preprocess().docker_compose_generator(no_of_clients)

        Preprocess().docker_initializer()

        metrics_dict = Decentralized().train_distributed(_ports, datasets, labels, CLIENTS=no_of_clients, GRID_ADDRESS=GRID_ADDRESS, GRID_PORT=GRID_PORT, N_EPOCHS=n_epochs)

        Preprocess().docker_kill()

    elif train_type == 'centralized':
        metrics_dict = Centralized().train_centralized(datasets=datasets, labels=labels, N_EPOCHS=n_epochs)

    #Save metrics
    Preprocess().save_metrics(metrics_dict, metrics_path, train_type)

    # Calculating end time
    end_minus_start_time = ((time.time() - start_time))
    print('=' * 60)
    print("OVERALL RUNTIME: {:.3f} seconds".format(end_minus_start_time))  # Calculating end time

if __name__ == "__main__":
    main()