import argparse

from pre_processors.frequency_vector import FrequencyVectorPreProcessor
from pre_processors.sliding_window import SlidingWindowPreProcessor
from pre_processors.n_gram import NGramPreProcessor
from models.knn import KNNModel
from .filesystem import resolve_abs_path

def get_args():
    parser = argparse.ArgumentParser(description='Hlin')
    parser.add_argument('-i', dest='input', required=True)
    parser.add_argument('-p', dest='pre_processor', required=True, choices=['frequency_vector', 'sliding_window', 'n_gram'])
    parser.add_argument('-m', dest='model', required=True, choices=['knn'])
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

    # Re-parse
    args = parser.parse_args()

    # Convert input to absolute path
    args.input = resolve_abs_path(args.input)

    return args

def pretty_print_args(args):
    kwargs = args._get_kwargs()

    print('------------ Settings ------------')
    for name, value in kwargs:
        print(f'{(name + ":"):<{20}}{value}')

