from data_splitter import Genes, ClientGenerator
import time
from data_sender import Preprocess
from training import Centralized, Decentralized
import warnings
import numpy as np
import sys
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
@click.option('--n_epochs', default=10, help='No. of Epochs / Rounds')
@click.option('--metrics_path', default='data/metrics', help="Path to save metrics")
@click.option('--model_save_path', default='data/models', help="Path to save trained models")
@click.option('--metrics_file_name', default=None, help="Custom name for metrics file")
@click.option('--no_of_clients', default=2, help="Clients / Nodes for decentralized training")
@click.option('--swarm', default='no', help="Option for switching between docker compose vs docker stack")
@click.option('--no_cuda', default=True, help="no_cuda = True means not to use CUDA. Default --> use CPU")
@click.option('--tags', default=None, help="Give tags for the data, which is to be sent to the nodes")
@click.option('--node_start_port', default='3000', help="Start port No. for a node")
@click.option('--grid_address', default='0.0.0.0', help="grid address for network")
@click.option('--grid_port', default='5000', help="grid port for network")


def main(samples_path, expressions_path, train_type, dataset_size, no_of_clients, grid_port, grid_address, metrics_path, n_epochs, split_type, metrics_file_name, swarm, no_cuda, model_save_path, split_size, node_start_port, tags):

    print('='*60)

    try:
        GRID_ADDRESS = grid_address
        GRID_PORT = grid_port

        if split_type == 'balanced':
            no_samples_to_take = dataset_size // 6

        # Calculate start time
        start_time = time.time()

        print('----<DATA PREPROCESSING STARTED..>----')
        # Using class Genes
        genes = Genes(samples_path, expressions_path, problem_type="classification")
        X = genes.get_features_dataframe().values
        Y = genes.Y
        transformed_genes = genes.transform_to_interval(Y)

        # Using class ClientGenerator
        client_generator = ClientGenerator()
        res = client_generator.balanced_sample_maker(X, np.asarray(transformed_genes), no_samples_to_take)
        y = client_generator.label_encode(res[1])
        client_dict = client_generator.create_clients(res[0], y, no_of_clients)

        # Using class Preprocess
        prepro = Preprocess()
        datasets, labels = prepro.tensor_converter(client_dict)

    except Exception as e:
        print("ERROR: DATA PREPROCESS ERROR!")
        print('FULL ERROR: {}'.format(e))
        sys.exit(1)

    print('----<STARTED TRAINING IN A {} FASHION..>----'.format(train_type))
    print('DATASET SIZE: {}'.format(dataset_size))

    if train_type == 'decentralized':

        try:
            print('TOTAL CLIENTS: {}'.format(no_of_clients))
            print('DATAPOINTS WITH EACH CLIENT: ')

            for key, val in client_dict.items():
                data, label = zip(*client_dict[key])
                unique, counts = np.unique(label, return_counts=True)
                print('{}: {} ; Label Count: {}'.format(key, len(label), dict(zip(unique, counts))))

            # Initializing Docker
            _ports = prepro.docker_compose_generator(no_of_clients)
            prepro.docker_initializer(SWARM=swarm)

            decentralized_ob = Decentralized()
            metrics_dict = decentralized_ob.train_distributed(_ports, datasets, labels, CLIENTS=no_of_clients, GRID_ADDRESS=GRID_ADDRESS, GRID_PORT=GRID_PORT, N_EPOCHS=n_epochs, NO_CUDA=no_cuda)

            prepro.docker_kill()

        except Exception as e:

            # Cleaning Docker containers
            prepro.docker_kill()

            print("ERROR: DECENTRALIZED TRAIN ERROR!")
            print('FULL ERROR: {}'.format(e))
            sys.exit(1)

    elif train_type == 'centralized':

        try:
            centralized_ob = Centralized()
            metrics_dict = centralized_ob.train_centralized(datasets=datasets, labels=labels, N_EPOCHS=n_epochs)

        except Exception as e:
            print("ERROR: CENTRALIZED TRAIN ERROR!")
            print('FULL ERROR: {}'.format(e))
            sys.exit(1)

    #Save metrics
    prepro.save_metrics(metrics_dict, metrics_path, train_type, metrics_file_name)

    # Calculating end time
    end_minus_start_time = ((time.time() - start_time))
    print('=' * 60)
    print("OVERALL RUNTIME: {:.3f} seconds".format(end_minus_start_time))  # Calculating end time

if __name__ == "__main__":
    main()