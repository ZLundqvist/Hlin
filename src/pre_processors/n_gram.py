import argparse
import os
import pandas
import numpy as np
from util.filesystem import ensure_file, write_cache_pickle, read_cache_pickle
from util.pre_processing import get_calls_metadata, system_calls_iterator, drop_duplicates
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
            print(f'[+] n-grams: {len(cached_df.index)}')
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

        n_grams = self.create_n_grams(unique_syscalls=unique_calls)

        print(f'[+] Bags: {len(n_grams)}')

        return pandas.DataFrame(n_grams)

    def create_n_grams(self, unique_syscalls: list):
        ngrams = []
        counter = 0
        current_n_gram = None
        
        for system_call in system_calls_iterator(self.input):

            if counter == 0:
                current_n_gram = {
                    'label': 'N',
                    'timestamp': system_call['timestamp'],
                    'seq' : []
                }


            counter = counter + 1

            if counter <= self.ngram_size:
                if system_call['label'] == 'A':
                    current_n_gram['label'] = 'A'
                system_call_name = system_call['name']
                current_n_gram['seq'].append(system_call_name)
                
            else:
                # One Hot Encoding

                # encoder = OneHotEncoder()
                # X = unique_system_calls * [len(unique_system_calls)]
                # encoder.fix(X)
                # encoder.transform(ngrams)
                data = current_n_gram['seq']
                values = np.array(data)
                values = values.reshape(len(unique_syscalls), 1)
                # integer encode

                encoder = OneHotEncoder(sparse=False)
                onehot_encoded = encoder.fit(values)
                print(onehot_encoded)

                # add
                ngrams.append(current_n_gram)
                print(current_n_gram)
                counter = 0
                current_n_gram = None

        # If we iterated through all system calls and one bag was not completed (not filled)
        if current_n_gram is not None:
            ngrams.append(current_n_gram)

        return ngrams
    # def create_n_grams(self, system_calls: list, unique_calls: list):
    #     ngrams = []

    #     # encoder = OneHotEncoder()
    #     # X = unique_system_calls * [len(unique_system_calls)]
    #     # encoder.fix(X)
    #     # encoder.transform(ngrams)

    #     words_count = len(unique_calls)
    #     arrange = np.arange(words_count)
        
    #     for index in range(0, len(system_calls), self.ngram_size):
    #         ngram = []
    #         label = 'N'
    #         for system_call in system_calls[index:index+self.ngram_size]:
    #             ngram.append(system_call['name'])
    #             if system_call['label'] == 'A':
    #                 label = 'A'
                
    #         if(len(ngram) != self.ngram_size):
    #             print(f"{system_calls[index]['timestamp']}: {len(ngram)}")
            

    #         ngram = self.get_one_hot(arrange, words_count)

    #         # print(ngram)
    #         ngrams.append({
    #             'timestamp': system_calls[index]['timestamp'],
    #             'label': label,
    #             'ngrams': ngram
    #         })

    #     return ngrams

    @staticmethod
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
        
    @staticmethod
    def get_static_id(args):
        return f'n_gram_{args.ngram_size}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--ngram-size', dest='ngram_size', type=int, default=10)

        
