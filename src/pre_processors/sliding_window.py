import argparse
import os

from util.filesystem import ensure_file, write_cache_json, read_cache_json
from util.pre_processing import read_log_file, get_unique_calls

class SlidingWindowPreProcessor:
    
    def __init__(self, input_file: str, args):
        self.input = input_file
        self.input_filename = os.path.basename(self.input).split('.')[0] # Get file name without extension
        self.window_size = args.window_size
        self.window_step_size = args.window_step_size
        self.id = f'{self.get_static_id(args)}_{self.input_filename}'

    def pre_process(self):
        cached_bags = read_cache_json(self.id)
        if cached_bags:
            print(f'[+] Bags: {len(cached_bags)}')
            return cached_bags

        if not ensure_file(self.input):
            raise Exception(f'Input file does not exist: {self.input}')

        system_calls = read_log_file(self.input)
        print(f'[+] System calls: {len(system_calls)}')

        unique_calls = get_unique_calls(system_calls)
        print(f'[+] Unique calls: {len(unique_calls)}')

        bags = self.create_bags(system_calls=system_calls, unique_calls=unique_calls)
        print(f'[+] Bags: {len(bags)}')

        write_cache_json(self.id, bags)

        return bags

    def create_bags(self, system_calls: list, unique_calls: set):
        bags = []

        for index in range(0, len(system_calls), self.window_step_size):
            bag = {}
            label = 'N'

            for unique_call in unique_calls:
                bag[unique_call] = 0

            for system_call in system_calls[index:index+self.window_size]:
                if system_call['label'] == 'A':
                    label = 'A'
                
                system_call_name = system_call['name']
                bag[system_call_name] = bag[system_call_name] + 1

            bags.append({
                'timestamp': system_calls[index]['timestamp'],
                'label': label,
                'bag': bag
            })

        return bags

    # Returns the static portion of the pre-processor id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'sliding_window_{args.window_size}_{args.window_step_size}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--window-size', dest='window_size', type=int, default=11)
        argparser.add_argument('--window-step-size', dest='window_step_size', type=int, default=6)

