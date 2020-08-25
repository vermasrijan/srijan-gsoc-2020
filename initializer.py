from data_splitter import Genes, ClientGenerator
from time import time
import numpy as np
from data_sender import Preprocess, DataSender
from training import Decentralized, Centralized
import warnings
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
@click.option('--no_of_clients', default=3, help="Clients / Nodes for decentralized training")
@click.option('--node_start_port', default='3000', help="Start port No. for a node")
@click.option('--grid_address', default='0.0.0.0', help="grid address for network")
@click.option('--grid_port', default='5000', help="grid port for network")

def main(samples_path, expressions_path, train_type, dataset_size, no_of_clients, grid_port, grid_address, split_type, split_size, node_start_port):

    print('='*60)
    GRID_ADDRESS = grid_address
    GRID_PORT = grid_port

    no_samples_to_take = dataset_size // no_of_clients
    # Calculate start time
    start_time = time()

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
    if train_type == 'decentralized':

        _ports = Preprocess().docker_compose_generator(no_of_clients)

        Preprocess().docker_initializer()

        # try:
        DataSender().send_client_data(_ports, datasets, labels)

        Decentralized().train_distributed(CLIENTS=no_of_clients)
        # except:
        print('DONE!')

    elif train_type == 'centralized':
        Centralized().train_centralized(datasets=datasets, labels=labels)

    # Calculating end time
    end_minus_start_time = ((time() - start_time))
    print('=' * 60)
    print("OVERALL RUNTIME: {:.3f} seconds".format(end_minus_start_time))  # Calculating end time

if __name__ == "__main__":
    main()