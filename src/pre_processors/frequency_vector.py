import argparse
import os
import pandas as pd
from decimal import Decimal

from util.filesystem import ensure_file, read_cache_pickle, write_cache_pickle
from util.pre_processing import get_system_calls_metadata, drop_duplicates, get_system_calls

class FrequencyVectorPreProcessor:
    def __init__(self, input_file: str, args):
        self.input = input_file
        self.input_filename = os.path.basename(self.input).split('.')[0] # Get file name without extension
        self.delta_t = Decimal(args.delta_t) / Decimal(1000) # convert to ms
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

    def create_dataframe(self) -> pd.DataFrame:
        num_calls, unique_calls = get_system_calls_metadata(self.input)

        print(f'[+] System calls: {num_calls}')
        print(f'[+] Unique calls: {len(unique_calls)}')

        bags = self.create_bags(unique_syscalls=unique_calls)

        print(f'[+] Bags: {len(bags)}')

        return pd.DataFrame(bags)

    def create_bags(self, unique_syscalls: list):

        def create_new_bag(first_system_call):
            new_bag = {
                'label': 'N',
                'timestamp': first_system_call['timestamp']
            }

            # Fill bag
            for unique_call in unique_syscalls:
                new_bag[unique_call] = 0

            # Add syscall to new bag
            system_call_name = first_system_call['name']
            new_bag[system_call_name] = new_bag[system_call_name] + 1

            # Update label if needed
            if first_system_call['label'] == 'A':
                new_bag['label'] = 'A'

            return new_bag


        bags = []
        current_bag = None
        for system_call in get_system_calls(self.input):
            # For first syscall => create new bag and add syscall to it
            if current_bag is None:
                current_bag = create_new_bag(system_call)
                continue

            # For all but first bag            
            if system_call['timestamp'] >= current_bag['timestamp'] + self.delta_t:
                # If system_call belongs to the next bag
                # Add current_bag to list of bags
                bags.append(current_bag)

                # Create new current_bag
                current_bag = create_new_bag(system_call)
            else:
                # If system_call belongs to current bag
                system_call_name = system_call['name']
                current_bag[system_call_name] = current_bag[system_call_name] + 1
                # Set label
                if system_call['label'] == 'A':
                    current_bag['label'] = 'A'

        # Last bag was not added to list of bags, so add it here
        bags.append(current_bag)

        return bags
    
    # Returns the static portion of the pre-processor id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'fv_{args.delta_t}_{args.drop_duplicates_mode}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--delta-t', dest='delta_t', help="delta_t in milliseconds", default=100, type=int)

