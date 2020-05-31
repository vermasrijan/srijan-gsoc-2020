import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import backend as K
import os
import json
import pickle
from tensorflow import keras
import time
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

#TODO -
# 1. 'g1' to be mapped with 'l1' where 'g1' is global model
# directory and 'l1' has local model trained for 'g1' and 'l1' lies in <c1,c2,c3...>
# 2. 'g2' to be mapped with 'l2'
# 3. take input as 'g1', and if input = 'g1', then local mod training
# happens for 'l1', for all clients

#Load test dataset
def load_data(data_dir):

    with open(data_dir+'/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # process and batch the test set
    test_batched = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1])).batch(len(test_data[1]))

    return test_batched

# Load Global Model
def load_model(model_dir, glob_mod_name='g1'):
    model = keras.models.load_model(model_dir)
    return model

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clients
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in
                        client_names]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return local_count / global_count

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad

#Testing metrics for global model
def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('Results_after_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, round(float(loss), 3)))
    return round(float(acc), 3), round(float(loss), 3)


def modify_search_args(args: Namespace):
    """
    Modifies and validates searching arguments in place.
    :param args: Arguments.
    """
    # TODO : Take glob_mod_name as input argument
    glob_mod_name = 'g1'
    local_mod_no = glob_mod_name[-1]

    #Load Global Model
    print("--------< LOADING GLOBAL MODEL >----------")
    global_model = load_model(os.path.join(args.global_model_path + '/'))
    print('Global model loaded successfully!')
    print("-" * 62)

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    print("--------< LOADING CLIENT MODELS >----------")
    for subdir, dirs, files in os.walk(args.client_dir):
        if str(subdir)[-2:] == 'l{}'.format(local_mod_no):
            # TODO - sampleNo/Total sample - write code below, using metadataof each client
            # for file in files:
            #     if 'metadata' in str(file):
            #         with open(os.path.join(subdir, file)) as f:
            #             m_data = json.load(f)
            #     print(os.path.join(subdir, file))
    # scale the model weights and add to list
    #TODO scaling factor is hard coded at the moment
    # scaling_factor = weight_scalling_factor(clients_batched, client)

            #Loading local model
            local_model = load_model(os.path.join(subdir+'/'))

            # scale the model weights and add to list
            scaling_factor = 0.2
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

    # clear session to free memory after each communication round
    K.clear_session()

    print("--------< RUNNING FEDERATED AVERAGING ALGORITHM... >----------")
    # to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    # update global model
    global_model.set_weights(average_weights)

    #Load test data
    test_batched = load_data(args.test_dataset_path)

    #NOTE: no. of keys in global model metadata file = rounds done till now
    try:
        with open(args.metadata_path) as f:
            glob_mod_metadata = json.load(f)
        comm_round = len(glob_mod_metadata.keys()) + 1

    except:
        glob_mod_metadata = dict()
        comm_round = 1

    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

    #Save metadata in dict
    glob_mod_metadata['round_{}_results'.format(comm_round)] = {'global_acc': global_acc, 'global_loss': global_loss}

    #Save metrics in metadata
    with open(args.metadata_path, 'w') as f:
        json.dump(glob_mod_metadata, f, indent=4)

def add_search_args(parser: ArgumentParser):
    """
    Adds searching arguments to an ArgumentParser.
    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--global_model_path', type=str, default=None,
                        help='Global model directory')
    parser.add_argument('--test_dataset_path', type=str, default=None,
                        help='Path to test dataset')
    parser.add_argument('--client_dir', type=str, default=None,
                        help='Directory where client models are saved')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to save metadata files')

def parse_search_args() -> Namespace:
    """
    Parses arguments for training (includes preprocessing/validating arguments).
    :return: A Namespace containing the parsed, modified, and validated args.
    """
    example_text = '''example:

    ->$ python fed_av_algo.py --global_model_path <Global_model_path> --test_dataset_path <Path_to_test_data> --client_dir <Clients Dir>  --metadata_path <Global model Metadata_file>
    ->$ python fed_av_algo.py --global_model_path ./coordinator/g1/ --test_dataset_path coordinator/g1/ --client_dir ./clients --metadata_path coordinator/g1/global_mod_metadata.json
     For more help, use help argument. Example: python fed_av_algo.py -h
     '''

    descript = '''
    ->fed_av_algo.py is a script for averaging local model weights. 
    ->Provide a valid path to global_model, client_dir,  metadata_path and test_dataset_path
    ->Global Model will be updated'''

    parser = ArgumentParser(prog='fed_av_algo',
                            description=descript,
                            epilog=example_text,
                            formatter_class=RawDescriptionHelpFormatter)

    # parser = ArgumentParser()
    add_search_args(parser)
    args = parser.parse_args()
    if args.global_model_path is None or args.test_dataset_path is None or args.client_dir is None\
            or args.metadata_path is None:
        print('Please provide arguments.')
        print(
            'Example command: $ python fed_av_algo.py --global_model_path <Global_model_path> --test_dataset_path <Path_to_test_data> --client_dir <Clients Dir>  --metadata_path <Global model Metadata_file>')
        print('Example: $ python fed_av_algo.py --global_model_path ./coordinator/g1/ --test_dataset_path coordinator/g1/ --client_dir ./clients --metadata_path coordinator/g1/global_mod_metadata.json')
        print('For more help, use help argument. Example: python fed_av_algo.py -h')
    else:
        modify_search_args(args)

    return args

if __name__ == '__main__':
    # Calculate start time
    start_time = time.time()

    print("=" * 62)

    args = parse_search_args()

    # Calculating end time
    end_minus_start_time = ((time.time() - start_time) / 60)
    print("TOTAL SCRIPT RUNTIME: {:.3f} minutes".format(end_minus_start_time))  # Calculating end time
    print("=" * 62)
    # print(args)

'''
Example code:
$python fed_av_algo.py --global_model_path <Global_model_path> --test_dataset_path <Path_to_test_data> --client_dir <Clients Dir>  --metadata_path <Global model Metadata_file>
$python fed_av_algo.py --global_model_path ./coordinator/g1/ --test_dataset_path coordinator/g1/ --client_dir ./clients --metadata_path coordinator/g1/global_mod_metadata.json
'''