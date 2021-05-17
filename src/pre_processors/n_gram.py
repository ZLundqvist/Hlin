import argparse
import os
import numpy as np
from util.filesystem import ensure_file, write_cache_json, read_cache_json
from util.pre_processing import read_log_file, get_unique_calls

class NGramPreProcessor:
    def __init__(self, input_file: str, args):
        self.input = input_file
        self.input_filename = os.path.basename(self.input).split('.')[0] # Get file name without extension
        self.ngram_size = args.ngram_size
        
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

        n_grams = self.create_n_grams(system_calls=system_calls, unique_calls=unique_calls)

        write_cache_json(self.id, n_grams)

        return n_grams
    
    def create_n_grams(self, system_calls: list, unique_calls: list):
        ngrams = []

        # encoder = OneHotEncoder()
        # X = unique_system_calls * [len(unique_system_calls)]
        # encoder.fix(X)
        # encoder.transform(ngrams)

        words_count = len(unique_calls)
        arrange = np.arange(words_count)
        
        for index in range(0, len(system_calls), self.ngram_size):
            ngram = []
            label = 'N'
            for system_call in system_calls[index:index+self.ngram_size]:
                ngram.append(system_call['name'])
                if system_call['label'] == 'A':
                    label = 'A'
                
            if(len(ngram) != self.ngram_size):
                print(f"{system_calls[index]['timestamp']}: {len(ngram)}")
            

            ngram = self.get_one_hot(arrange, words_count)

            # print(ngram)
            ngrams.append({
                'timestamp': system_calls[index]['timestamp'],
                'label': label,
                'ngrams': ngram
            })

        return ngrams

    @staticmethod
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
        
    @staticmethod
    def get_static_id(args):
        return f'n_gram_{args.ngram_size}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--ngram-size', dest='ngram_size', type=int, default=1)

        
