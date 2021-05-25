import time

from pre_processors.frequency_vector import FrequencyVectorPreProcessor
from pre_processors.sliding_window import SlidingWindowPreProcessor
from pre_processors.n_gram import NGramPreProcessor
from models.knn import KNNModel
from models.isolation_forest import IsolationForestModel
from models.one_class_svm import OneClassSVMModel
from util.args import pretty_print_args
from util.filesystem import does_eval_results_file_exist, get_dir_files_abs, resolve_abs_path, write_eval_results_to_csv

from models.random_forest import RandomForestModel

import data_transformers.knn as KNNTransformers
import data_transformers.random_forest as RandomForestTransformers
import data_transformers.isolation_forest as IsolationForestTransformers
import data_transformers.one_class_svm as OneClassSVMTransformers

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
            elif self.args.model == 'one_class_svm':
                self.data_transformer = OneClassSVMTransformers.from_frequency_vector
                self.model_class = OneClassSVMModel

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
            elif self.args.model == 'one_class_svm':
                self.data_transformer = OneClassSVMTransformers.from_sliding_window
                self.model_class = OneClassSVMModel

        elif self.args.pre_processor == 'n_gram':
            self.pre_processor_class = NGramPreProcessor

            if self.args.model == 'knn':
                self.data_transformer = KNNTransformers.from_n_gram
                self.model_class = KNNModel
            elif self.args.model == 'isolation_forest':
                self.data_transformer = IsolationForestTransformers.from_n_gram
                self.model_class = IsolationForestModel
            elif self.args.model == 'random_forest':
                self.data_transformer = RandomForestTransformers.from_n_gram
                self.model_class = RandomForestModel
            elif self.args.model == 'one_class_svm':
                self.data_transformer = OneClassSVMTransformers.from_n_gram
                self.model_class = OneClassSVMModel

        # The run id is unique per combination of pp and model (including their args)
        self.run_id = f'{self.model_class.get_static_id(self.args)}_{self.pre_processor_class.get_static_id(self.args)}'

        # Ensure output file with same run_id does not exist
        assert not does_eval_results_file_exist(self.run_id), f'Evaluation results file already exists: {self.run_id}'


    def execute(self):
        results = []

        t0 = time.time()

        if self.args.input:
            input_file = resolve_abs_path(self.args.input)
            result = self.execute_file_input(input_file, progress='[1/1]')
            results.append(result)
        elif self.args.input_dir:
            input_files = get_dir_files_abs(self.args.input_dir)
            assert len(input_files) > 0, 'Directory is empty'

            for idx, file in enumerate(input_files):
                result = self.execute_file_input(file, progress=f'[{idx}/{len(input_files)}]')
                results.append(result)


        if not self.args.skip_training:
            write_eval_results_to_csv(run_id=self.run_id, eval_results=results)

        print(f'[+] Run complete: {round(time.time() - t0, 2)}s')

    def execute_file_input(self, input_file, progress: str):
        print(f'\n---------- Pre-processing {progress} ----------')
        t0 = time.time()
        pp = self.pre_processor_class(args=self.args, input_file=input_file)
        df = pp.pre_process()
        print(f'[+] Pre-process: {round(time.time() - t0, 2)}s')

        print(f'\n------- Data Transformation {progress} -------')
        t0 = time.time()
        transformed_data_set = self.data_transformer(df)
        print(f'[+] Transform: {round(time.time() - t0, 2)}s')


        print(f'\n------- Training/Validation {progress} -------')
        if self.args.skip_training:
            print('[+] Skipped')
            return None

        t0 = time.time()
        model = self.model_class(args=self.args, input_file=input_file, data_set=transformed_data_set)
        eval_result = model.train_validate()
        eval_result.set_iteration(self.args.iteration)
        print(f'[+] Train/Validate: {round(time.time() - t0, 2)}s')

        return eval_result
        