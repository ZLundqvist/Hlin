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

    # Creates an empty pandas DataFrame which has been memory optimized
    def create_empty_dataframe(self, unique_syscalls: list):
        # Create dataframe with a column for each system call
        # columns = unique_syscalls.copy()
        df = pd.DataFrame()

        # # Convert all columns to type int16
        # df = df.astype('uint16')

        # # Insert label column and convert it to categorical type
        # df.insert(loc=0, column='label', value='N')
        # df['label'] = df['label'].astype('category')

        # ['N', 0, 1, 0, 0, 1, 0, 1, 0,]

        return df

    def create_dataframe(self):
        num_calls, unique_calls = get_system_calls_metadata(self.input)
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        X = []
        for i in range(self.ngram_size):
            X.append(unique_calls.copy())
        X = np.transpose(X)

        encoder = encoder.fit(X)
        
        print(f'[+] System calls: {num_calls}')
        print(f'[+] Unique calls: {len(unique_calls)}')

        # df = self.create_empty_dataframe(unique_syscalls=unique_calls)
        n_grams = []
        labels = []
        system_call_history = []

        def create_new_n_gram():
            sequence = []

            # new_n_gram = {
            #     'label': 'N',
            # }

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
                # print(len(sequence))
                sequence.append(None)

            # new_n_gram['seq'] = encoder.transform([sequence])

            n_grams.append(sequence)

            # n_grams.append(encoder.transform([sequence1])[0])
            # print(encoder.transform([sequence]))
            #encoder.transform([sequence])
            # return new_n_gram

        current_syscall_number = 0
        for system_call in get_system_calls(self.input):
            # Add syscall to start of list
            system_call_history.append(system_call)
            current_syscall_number = current_syscall_number + 1
            history_has_new_syscalls = True

            assert(len(system_call_history) <= self.ngram_size) # Length should never exceed window size

            if len(system_call_history) == self.ngram_size:
                # The history is large enough to create a new bag, nice!
                create_new_n_gram()

                # Remove <window_step_size> last items in history
                system_call_history =[]
                history_has_new_syscalls = False

            
            FLUSH_AT = 100000
            #Periodically flush bags to dataframe
            # if len(n_grams) > FLUSH_AT: # arbitrary
                # df = df.append(n_grams)
                
                # n_grams = []
                # labels=[]
            if current_syscall_number % 50000 == 0:

                progress = round((current_syscall_number / num_calls) * 100, 5)
                sys.stdout.write(f'\r[+] {progress}%')
                sys.stdout.flush()

        print('')

        # We might need to create one last bag if system calls were added after the last bag was created
        if history_has_new_syscalls:
            create_new_n_gram()

        # first_ngram = n_grams[0]
        # first_encoded_ngram= encoder.transform([first_ngram])
        # df = pd.DataFrame(first_encoded_ngram, dtype='int8')
        # print(df)

        
        print(f'\n[+] One-Hot encoding...')
        encoded_n_grams = encoder.transform(n_grams)
        print(f'[+] Encoding done')

        df2 = pd.DataFrame(encoded_n_grams, dtype='bool')

        print(df2)
        # df_final = pd.concat([df, df2], ignore_index=True, axis=0)

        # print(df_final)

        # df = df.append(encoded_n_grams)
           


        # df = pd.DataFrame()
        # df = df.append(encoded_n_grams)
        # df = df.astype('uint16') # Convert to uint datatype
        df2['label'] = labels
        print(df2)
        # # Check that we generated correct amount of bags
        # expected_rows = math.ceil((num_calls / self.ngram_size))  # Number of rows expected is number of steps needed to be taken + 1 (because a bag is created before first step is taken)
        # assert(len(df_final) == expected_rows)
        return df2   

    @staticmethod
    def get_static_id(args):
        return f'n_gram_{args.ngram_size}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--ngram-size', dest='ngram_size', type=int, default=5)

        
