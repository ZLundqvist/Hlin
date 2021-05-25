import argparse
from models.random_forest import RandomForestModel
from pre_processors.frequency_vector import FrequencyVectorPreProcessor
from pre_processors.sliding_window import SlidingWindowPreProcessor
from pre_processors.n_gram import NGramPreProcessor
from models.knn import KNNModel
from models.isolation_forest import IsolationForestModel
from models.one_class_svm import OneClassSVMModel
from .filesystem import resolve_abs_path

def get_args():
    parser = argparse.ArgumentParser(description='Hlin')
    parser.add_argument('-p', dest='pre_processor', required=True, choices=['frequency_vector', 'sliding_window', 'n_gram'])
    parser.add_argument('-m', dest='model', required=True, choices=['knn', 'isolation_forest', 'random_forest', 'one_class_svm'])
    parser.add_argument('--dd', dest='drop_duplicates_mode', default='none', choices=['first', 'label', 'none'])
    parser.add_argument('--skip-training', dest='skip_training', default=False, action='store_true')
    parser.add_argument('--iteration', dest='iteration', type=int, required=True)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', dest='input')
    input_group.add_argument('--input-dir', dest='input_dir')
    args = parser.parse_known_args()

    # Append preprocessor-specific args 
    if args[0].pre_processor == 'frequency_vector':
        FrequencyVectorPreProcessor.append_args(argparser=parser)
    elif args[0].pre_processor == 'sliding_window':
        SlidingWindowPreProcessor.append_args(argparser=parser)
    elif args[0].pre_processor == 'n_gram':
        NGramPreProcessor.append_args(argparser=parser)

    # Append model-specific args 
    if args[0].model == 'knn':
        KNNModel.append_args(argparser=parser)
    elif args[0].model == 'isolation_forest':
        IsolationForestModel.append_args(argparser=parser)
    elif args[0].model == 'random_forest':
        RandomForestModel.append_args(argparser=parser)
    elif args[0].model == 'one_class_svm':
        OneClassSVMModel.append_args(argparser=parser)
    else:
        raise Exception('Unsupported model', args[0].model)

    # Re-parse
    args = parser.parse_args()
    return args

def pretty_print_args(args):
    kwargs = args._get_kwargs()

    print('------------ Settings ------------')
    for name, value in kwargs:
        if value is not None:
            print(f'{(name + ":"):<{22}}{value}')

