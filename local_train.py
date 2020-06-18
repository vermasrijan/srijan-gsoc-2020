import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import pickle
import json
import time
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter


#For initializing / compiling local model
class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

#Configurations of local model -
#<TODO> Decay needs to changed!! because there are no 'comms_round' now
lr = 0.01
comms_round = 10
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr,
                decay=lr / comms_round,
                momentum=0.9
               )


#Load Global Model
def load_model(gobal_model_dir):
    global_model = keras.models.load_model(gobal_model_dir)
    return global_model


#Load dataset
def load_data(data_path):
    # Data path currently is a 'pickle file'
    with open(data_path, 'rb') as f:
        tr_data = pickle.load(f)

    return tr_data

#Batch data
def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


#Train local model
def local_train(local_data ,global_model, local_epoch=1):

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    smlp_local = SimpleMLP()
    local_model = smlp_local.build(784, 10)
    local_model.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=metrics)

    # set local model weight to the weight of the global model
    local_model.set_weights(global_weights)

    # fit local model with client's data
    local_model.fit(local_data, epochs=local_epoch, verbose=0)

    return local_model


#Save local model file
def save_model(save_dir, local_model):
    local_model.save(save_dir)


#Save metadata in json format
def save_metadata(raw_data, metadata_path):
    meta_dict = dict()

    meta_dict['total_samples'] = len(raw_data)

    with open(metadata_path, 'w') as f:
        json.dump(meta_dict, f, indent=4)

def modify_search_args(args: Namespace):
    """
    Modifies and validates searching arguments in place.
    :param args: Arguments.
    """

    # LOADING DATA
    print("--------< LOADING GLOBAL MODEL >----------")
    # print("-" * 62)
    global_model = load_model(args.global_model_path)
    print('Global model loaded successfully!')
    print("-" * 62)

    # DATA PRE-PROCESSING
    print("--------< LOADING LOCAL DATA >----------")
    print('Please wait for ~2 seconds  :) ')
    data_shard = load_data(args.local_dataset_path)
    local_data = batch_data(data_shard, bs=32)

    # LOCAL MODEL TRAINING
    print("--------< LOCAL MODEL TRAINING >----------")
    print('Please wait for ~1 seconds  :) ')
    local_model = local_train(local_data, global_model, local_epoch=1)

    # save local model
    print("--------< SAVE LOCAL MODEL WEIGHTS >----------")
    save_model(args.local_model_path, local_model)

    # SAVE METADATA FILE
    save_metadata(data_shard, args.metadata_path)



    client_index = args.local_model_path.find('clients') + len('clients/c')
    print("----< CLIENT {}: LOCAL MODEL TRAINING COMPLETED! >----".
          format(args.local_model_path[client_index]))

def add_search_args(parser: ArgumentParser):
    """
    Adds searching arguments to an ArgumentParser.
    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--global_model_path', type=str, default=None,
                        help='Global model directory')
    parser.add_argument('--local_dataset_path', type=str, default=None,
                        help='Path to clients dataset')
    parser.add_argument('--local_model_path', type=str, default=None,
                        help='Path to save clients local model')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to save metadata files')


def parse_search_args() -> Namespace:
    """
    Parses arguments for training (includes preprocessing/validating arguments).
    :return: A Namespace containing the parsed, modified, and validated args.
    """
    example_text = '''example:

    ->$ python local_train.py --global_model_path <Global_model_path> --local_model_path <Path_to_save_client_model> --local_dataset_path <Client Dataset>  --metadata_path <Metadata_file>
    ->$ python local_train.py --global_model_path ./coordinator/g1/ --local_model_path ./clients/c1/l1/ --local_dataset_path ./clients/c1/l1/train_data.pkl --metadata_path ./clients/c1/l1/metadata.json

     For more help, use help argument. Example: python local_train.py -h
     '''

    descript = '''
    ->local_train.py is a script for training local models at client side. 
    ->Provide a valid path to global_model, client_data_path,  metadata_path and local_model_save_directory
    ->Local data will be trained and weights will be saved for Fed. averaging algorithm'''

    parser = ArgumentParser(prog='local_train',
                            description=descript,
                            epilog=example_text,
                            formatter_class=RawDescriptionHelpFormatter)

    # parser = ArgumentParser()
    add_search_args(parser)
    args = parser.parse_args()
    if args.global_model_path is None or args.local_model_path is None or args.local_dataset_path is None\
            or args.metadata_path is None:
        print('Please provide arguments.')
        print(
            'Example command: $ python local_train.py --global_model_path <Global_model_path> --local_model_path <Path_to_save_client_model> --local_dataset_path <Client Dataset>  --metadata_path <Metadata_file>')
        print('Example: $ python local_train.py --global_model_path ./coordinator/g1/ --local_model_path ./clients/c1/l1/ --local_dataset_path ./clients/c1/l1/train_data.pkl --metadata_path ./clients/c1/l1/metadata.json')
        print('For more help, use help argument. Example: python local_train.py -h')
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
$python local_train.py --global_model_path <Global_model_path> --local_model_path <Path_to_save_client_model> --local_dataset_path <Client Dataset>  --metadata_path <Metadata_file>
$python local_train.py --global_model_path ./coordinator/g1/ --local_model_path ./clients/c1/l1/ --local_dataset_path ./clients/c1/l1/train_data.pkl --metadata_path ./clients/c1/l1/metadata.json
'''