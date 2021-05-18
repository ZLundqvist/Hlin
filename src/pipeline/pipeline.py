import time

from pre_processors.frequency_vector import FrequencyVectorPreProcessor
from pre_processors.sliding_window import SlidingWindowPreProcessor
from pre_processors.n_gram import NGramPreProcessor
from models.knn import KNNModel
from models.isolation_forest import IsolationForestModel
from util.args import pretty_print_args
from util.filesystem import get_dir_files_abs, resolve_abs_path, write_eval_results_to_csv
from util.data_transformers import n_gram_knn, n_gram_isolation_forest

from models.random_forest import RandomForestModel

import data_transformers.knn as KNNTransformers
import data_transformers.random_forest as RandomForestTransformers
import data_transformers.isolation_forest as IsolationForestTransformers


class Pipeline:
    def __init__(self, args):
        self.args = args
        pretty_print_args(self.args)
        self.setup()

    def setup(self):
        if self.args.pre_processor == 'frequency_vector':
            self.pre_processor_class = FrequencyVectorPreProcessor
            
            if self.args.model == 'knn':
                self.data_transformer = KNNTransformers.from_frequency_vector
                self.model_class = KNNModel
            elif self.args.model == 'isolation_forest':
                self.data_transformer = IsolationForestTransformers.from_frequency_vector
                self.model_class = IsolationForestModel
            elif self.args.model == 'random_forest':
                self.data_transformer = RandomForestTransformers.from_frequency_vector
                self.model_class = RandomForestModel

        elif self.args.pre_processor == 'sliding_window':
            self.pre_processor_class = SlidingWindowPreProcessor

            if self.args.model == 'knn':
                self.data_transformer = KNNTransformers.from_sliding_window
                self.model_class = KNNModel
            elif self.args.model == 'isolation_forest':
                self.data_transformer = IsolationForestTransformers.from_sliding_window
                self.model_class = IsolationForestModel
            elif self.args.model == 'random_forest':
                self.data_transformer = RandomForestTransformers.from_sliding_window
                self.model_class = RandomForestModel

        elif self.args.pre_processor == 'n_gram':
            self.pre_processor_class = NGramPreProcessor

            if self.args.model == 'knn':
                self.data_transformer = KNNTransformers.from_n_gram
                self.model_class = KNNModel
            elif self.args.model == 'isolation_forest':
                self.data_transformer = RandomForestTransformers.from_n_gram
                self.model_class = IsolationForestModel
            elif self.args.model == 'random_forest':
                self.data_transformer = RandomForestTransformers.from_n_gram
                self.model_class = RandomForestModel

        # The run id is unique per combination of pp and model (including their args)
        self.run_id = f'{self.model_class.get_static_id(self.args)}_{self.pre_processor_class.get_static_id(self.args)}'

    def execute(self):
        results = []

        if self.args.input:
            input_file = resolve_abs_path(self.args.input)
            result = self.execute_file_input(input_file)
            results.append(result)
        elif self.args.input_dir:
            input_files = get_dir_files_abs(self.args.input_dir)
            assert len(input_files) > 0, 'Directory is empty'

            for f in input_files:
                input_file = resolve_abs_path(f)
                result = self.execute_file_input(f)
                results.append(result)

        write_eval_results_to_csv(run_id=self.run_id, eval_results=results)

    def execute_file_input(self, input_file):
        print('\n---------- Pre-processing ----------')
        t0 = time.time()
        pp = self.pre_processor_class(args=self.args, input_file=input_file)
        df = pp.pre_process()
        print(f'[+] Pre-process: {round(time.time() - t0, 2)}s')

        print('\n------- Data Transformation -------')
        t0 = time.time()
        transformed_data_set = self.data_transformer(df)
        print(f'[+] Transform: {round(time.time() - t0, 2)}s')

        print('\n------- Training/Validation -------')
        t0 = time.time()
        model = self.model_class(args=self.args, input_file=input_file, data_set=transformed_data_set)
        eval_result = model.train_validate()
        print(f'[+] Train/Validate: {round(time.time() - t0, 2)}s')

        return eval_result
        