import argparse
import os
import pandas as pd
import numpy as np
import sys
import math
from util.filesystem import ensure_file, write_cache_pickle, read_cache_pickle
from util.pre_processing import get_system_calls, get_system_calls_metadata, drop_duplicates
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class NGramPreProcessor:
    def __init__(self, input_file: str, args):
        self.input = input_file
        self.input_filename = os.path.basename(self.input).split('.')[0] # Get file name without extension
        self.ngram_size = args.ngram_size
        self.drop_duplicates_mode = args.drop_duplicates_mode
        
        self.id = f'{self.get_static_id(args)}_{self.input_filename}'

    def pre_process(self):
        cached_df = read_cache_pickle(self.id)
        if cached_df is not None:
            print(f'[+] N-grams: {len(cached_df.index)}')
            return cached_df

        if not ensure_file(self.input):
            raise Exception(f'Input file does not exist: {self.input}')

        
        print(f'[+] Creating DataFrame...')
        df = self.create_dataframe()
        df, duplicates_dropped = drop_duplicates(df=df, mode=self.drop_duplicates_mode)
        print(f'\n[+] DataFrame created (rows={len(df)}, duplicates_dropped={duplicates_dropped})')

        write_cache_pickle(self.id, df=df)

        return df


    def create_dataframe(self):
        num_calls, unique_calls = get_system_calls_metadata(self.input)
        
        X = []
        for i in range(self.ngram_size):
            X.append(unique_calls.copy())
        X = np.transpose(X)
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(X)
        
        print(f'[+] System calls: {num_calls}')
        print(f'[+] Unique calls: {len(unique_calls)}')

        n_grams = []
        labels = []
        dataframes = []
        system_call_history = []
        current_syscall_number = 0

        def flush_to_df():
            if not len(n_grams) > 0:
                return
            encoded_n_grams = encoder.transform(n_grams)
            dataframes.append(pd.DataFrame(encoded_n_grams, dtype='bool'))
            n_grams.clear()

            progress = round((current_syscall_number / num_calls) * 100, 3)
            sys.stdout.write(f'\r[+] {progress}%')
            sys.stdout.flush()
            

        def create_new_n_gram():
            sequence = []
            # Add history to new_n_gram
            labeltoset = 'N'
            for syscall in system_call_history:
                system_call_name = syscall['name']
                sequence.append(system_call_name)
                
                if syscall['label'] == 'A':
                    labeltoset = 'A'              
            labels.append(labeltoset)
            
            # Make sure sequence is of length=self.ngram_size
            while len(sequence) < self.ngram_size:
                sequence.append(None)
            n_grams.append(sequence)

        
        for system_call in get_system_calls(self.input):
            # Add syscall to start of list
            system_call_history.append(system_call)
            current_syscall_number = current_syscall_number + 1
            history_has_new_syscalls = True

            assert(len(system_call_history) <= self.ngram_size) # Length should never exceed window size

            if len(system_call_history) == self.ngram_size:
                # The history is large enough to create a new bag, nice!
                create_new_n_gram()

                #  Clear History
                system_call_history.clear()
                history_has_new_syscalls = False

            
            FLUSH_AT = 100000
            # Periodically flush ngrams to dataframe
            if len(n_grams) > FLUSH_AT: # arbitrary
                flush_to_df()

        # We might need to create one last bag if system calls were added after the last bag was created
        if history_has_new_syscalls:
            create_new_n_gram()

        flush_to_df()

        df_combined = pd.concat(dataframes)
        assert len(df_combined) == len(labels)

        # Add labels to df
        df_combined['label'] = labels
        df_combined['label'] = df_combined['label'].astype('category')
        
        # # Check that we generated correct amount of bags
        expected_rows = math.ceil((num_calls / self.ngram_size))  # Number of rows expected is number of steps needed to be taken + 1 (because a bag is created before first step is taken)
        assert(len(df_combined) == expected_rows)
        return df_combined

    @staticmethod
    def get_static_id(args):
        return f'n_gram_{args.ngram_size}_{args.drop_duplicates_mode}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--ngram-size', dest='ngram_size', type=int, default=5)

        
