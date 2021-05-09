import argparse


class SlidingWindowPreProcessor:
    
    def __init__(self, args):
        self.input = args.input
        print('SlidingWindowPreProcessor')


    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--window-size', dest='window_size', type=int, required=True)
        argparser.add_argument('--window-step-size', dest='window_step_size', type=int, required=True)

