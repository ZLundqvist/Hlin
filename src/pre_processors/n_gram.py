import argparse


class NGramPreProcessor:
    def __init__(self, args):
        self.input = args.input
        self.n = args.ngram_size
        print('NGramPreProcessor')


    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--ngram-size', dest='ngram_size', type=int, required=True)

