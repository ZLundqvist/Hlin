import argparse
import os
import pandas as pd

from util.filesystem import ensure_file, write_cache_json, read_cache_json
from util.pre_processing import read_log_file, get_unique_calls, get_calls_metadata

class FrequencyVectorPreProcessor:
    def __init__(self, input_file: str, args):
        self.input = input_file
        self.input_filename = os.path.basename(self.input).split('.')[0] # Get file name without extension
        self.delta_t = args.delta_t / 1000 # convert to ms
        
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

        frequency_vectors = self.create_frequency_vectors(system_calls=system_calls)

        bags = self.create_bags(frequency_vectors=frequency_vectors, unique_calls=unique_calls)
        print(f'[+] Bags: {len(bags)}')

        write_cache_json(self.id, bags)

        return bags

    def create_dataframe(self) -> pd.DataFrame:
        num_calls, unique_calls = get_calls_metadata(self.input)

        print(f'[+] System calls: {num_calls}')
        print(f'[+] Unique calls: {len(unique_calls)}')


        return None


    def create_frequency_vectors(self, system_calls: list) -> list:
        frequency_vectors = []

        current_timestamp = None
        for system_call in system_calls:
            if current_timestamp is None:
                frequency_vectors.append({
                    'timestamp': system_call['timestamp'],
                    'calls': [system_call]
                })
                current_timestamp = system_call['timestamp']
                continue
            
            if system_call['timestamp'] >= current_timestamp + self.delta_t:
                # If system_call belongs to the next timestamp
                frequency_vectors.append({
                    'timestamp': system_call['timestamp'],
                    'calls': [system_call]
                })
                current_timestamp = system_call['timestamp']
                continue
            else:
                # If system_call belongs to current_timestamp
                frequency_vectors[-1]['calls'].append(system_call)

        return frequency_vectors

    def create_bags(self, frequency_vectors: list, unique_calls: list) -> list:
        bags = []

        for frequency_vector in frequency_vectors:
            label = 'N'
            bag = {}

            for unique_call in unique_calls:
                bag[unique_call] = 0

            for system_call in frequency_vector['calls']:
                # Mark bag as anomalous if any of its system calls is labeled anomalous
                if system_call['label'] == 'A':
                    label = 'A'

                system_call_name = system_call['name']
                bag[system_call_name] = bag[system_call_name] + 1


            bags.append({
                'label': label,
                'timestamp': frequency_vector['timestamp'],
                'bag': bag
            })

        return bags
    
    # Returns the static portion of the pre-processor id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'frequency_vector_{args.delta_t}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--delta-t', dest='delta_t', help="delta_t in milliseconds", default=100, type=int)

