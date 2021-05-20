import argparse
import os
import pandas as pd
import sys
import math

from util.filesystem import ensure_file, write_cache_pickle, read_cache_pickle
from util.pre_processing import get_system_calls_metadata, get_system_calls, drop_duplicates

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

        
        print(f'[+] Creating DataFrame...')
        df = self.create_dataframe()
        df, duplicates_dropped = drop_duplicates(df=df, mode=self.drop_duplicates_mode)
        print(f'\n[+] DataFrame created (rows={len(df)}, duplicates_dropped={duplicates_dropped})')

        write_cache_pickle(self.id, df=df)

        return df

    # Creates an empty pandas DataFrame which has been memory optimized
    def create_empty_dataframe(self, unique_syscalls: list):
        # Create dataframe with a column for each system call
        columns = unique_syscalls.copy()
        df = pd.DataFrame(columns=columns)

        # Convert all columns to type int16
        df = df.astype('uint16')

        # Insert label column and convert it to categorical type
        df.insert(loc=0, column='label', value='N')
        df['label'] = df['label'].astype('category')

        return df

    def create_dataframe(self):
        num_calls, unique_calls = get_system_calls_metadata(self.input)
        print(f'[+] System calls: {num_calls}')
        print(f'[+] Unique calls: {len(unique_calls)}')

        df = self.create_empty_dataframe(unique_syscalls=unique_calls)
        bags = []
        system_call_history = []

        # This flag is used to indicate if system calls were added to history AFTER last bag was created
        # Basically, if it is True after the for-in loop system calls were added to history but not bag was created from them
        # If that is the case, create one last bag (size of bag will be smaller than window_size though)
        history_has_new_syscalls = False

        def create_new_bag():
            new_bag = {
                'label': 'N'
            }

            # Fill bag with zeroes
            for unique_call in unique_calls:
                new_bag[unique_call] = 0

            # Add history to new_bag
            for syscall in system_call_history:
                syscall_name = syscall['name']
                new_bag[syscall_name] = new_bag[syscall_name] + 1

                if syscall['label'] == 'A':
                    new_bag['label'] = 'A'

            return new_bag

        current_syscall_number = 0
        for system_call in get_system_calls(self.input):
            # Add syscall to start of list
            system_call_history.insert(0, system_call)
            current_syscall_number = current_syscall_number + 1
            history_has_new_syscalls = True

            assert(len(system_call_history) <= self.window_size) # Length should never exceed window size

            if len(system_call_history) == self.window_size:
                # The history is large enough to create a new bag, nice!
                new_bag = create_new_bag()
                bags.append(new_bag)

                # Remove <window_step_size> last items in history
                del system_call_history[-self.window_step_size:]

                history_has_new_syscalls = False

            
            FLUSH_AT = 100000
            # Periodically flush bags to dataframe
            if len(bags) > FLUSH_AT: # arbitrary
                df = df.append(bags)
                bags = []

                progress = round((current_syscall_number / num_calls) * 100, 5)
                sys.stdout.write(f'\r[+] {progress}%')
                sys.stdout.flush()

        # We might need to create one last bag if system calls were added after the last bag was created
        if history_has_new_syscalls:
            last_bag = create_new_bag()
            bags.append(last_bag)

        df = df.append(bags)
        
        # Check that we generated correct amount of bags
        expected_rows = math.ceil(((num_calls - self.window_size) / self.window_step_size)) + 1 # Number of rows expected is number of steps needed to be taken + 1 (because a bag is created before first step is taken)
        assert(len(df) == expected_rows)

        return df

    # Returns the static portion of the pre-processor id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'sw_{args.window_size}_{args.window_step_size}_{args.drop_duplicates_mode}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--window-size', dest='window_size', type=int, default=11)
        argparser.add_argument('--window-step-size', dest='window_step_size', type=int, default=6)

