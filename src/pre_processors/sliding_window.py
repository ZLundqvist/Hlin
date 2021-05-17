import argparse
import os
import pandas

from util.filesystem import ensure_file, write_cache_pickle, read_cache_pickle
from util.pre_processing import get_calls_metadata, system_calls_iterator, drop_duplicates

class SlidingWindowPreProcessor:
    
    def __init__(self, input_file: str, args):
        self.input = input_file
        self.input_filename = os.path.basename(self.input).split('.')[0] # Get file name without extension
        self.window_size = args.window_size
        self.window_step_size = args.window_step_size
        self.drop_duplicates_mode = args.drop_duplicates_mode
        self.id = f'{self.get_static_id(args)}_{self.input_filename}'

    def pre_process(self):
        cached_df = read_cache_pickle(self.id)
        if cached_df is not None:
            print(f'[+] Bags: {len(cached_df.index)}')
            return cached_df

        if not ensure_file(self.input):
            raise Exception(f'Input file does not exist: {self.input}')

        df = self.create_dataframe()
        df, duplicates_dropped = drop_duplicates(df=df, mode=self.drop_duplicates_mode)
        print(f'[+] DataFrame created (rows={len(df)}, duplicates_dropped={duplicates_dropped})')

        write_cache_pickle(self.id, df=df)

        return df

    def create_dataframe(self):
        num_calls, unique_calls = get_calls_metadata(self.input)

        print(f'[+] System calls: {num_calls}')
        print(f'[+] Unique calls: {len(unique_calls)}')

        bags = self.create_bags(unique_syscalls=unique_calls)

        print(f'[+] Bags: {len(bags)}')

        return pandas.DataFrame(bags)

    def create_bags(self, unique_syscalls: list):
        bags = []
        counter = 0
        current_bag = None
        for system_call in system_calls_iterator(self.input):

            if counter == 0:
                current_bag = {
                    'label': 'N',
                    'timestamp': system_call['timestamp']
                }

                # Fill bag
                for unique_call in unique_syscalls:
                    current_bag[unique_call] = 0


            counter = counter + 1

            if counter < self.window_step_size:
                if system_call['label'] == 'A':
                    current_bag['label'] = 'A'

                system_call_name = system_call['name']
                current_bag[system_call_name] = current_bag[system_call_name] + 1
            else:
                bags.append(current_bag)
                counter = 0
                current_bag = None
        
        # If we iterated through all system calls and one bag was not completed (not filled)
        if current_bag is not None:
            bags.append(current_bag)

        return bags

    # Returns the static portion of the pre-processor id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'sliding_window_{args.window_size}_{args.window_step_size}_{args.drop_duplicates_mode}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--window-size', dest='window_size', type=int, default=11)
        argparser.add_argument('--window-step-size', dest='window_step_size', type=int, default=6)

