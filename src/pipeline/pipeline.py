import time

from pre_processors.frequency_vector import FrequencyVectorPreProcessor
from pre_processors.sliding_window import SlidingWindowPreProcessor
from models.knn import KNNModel
from models.isolation_forest import IsolationForestModel
from util.args import pretty_print_args
from util.data_transformers import frequency_vector_knn, frequency_vector_isolation_forest, sliding_window_knn, sliding_window_isolation_forest

class Pipeline:
    def __init__(self, args):
        self.args = args
        pretty_print_args(self.args)
        self.setup()

    def setup(self):
        if self.args.pre_processor == 'frequency_vector':
            self.pre_processor_class = FrequencyVectorPreProcessor
            
            if self.args.model == 'knn':
                self.data_transformer = frequency_vector_knn
                self.model_class = KNNModel
            elif self.args.model == 'isolation_forest':
                self.data_transformer = frequency_vector_isolation_forest
                self.model_class = IsolationForestModel

        elif self.args.pre_processor == 'sliding_window':
            self.pre_processor_class = SlidingWindowPreProcessor

            if self.args.model == 'knn':
                self.data_transformer = sliding_window_knn
                self.model_class = KNNModel
            elif self.args.model == 'isolation_forest':
                self.data_transformer = sliding_window_isolation_forest
                self.model_class = IsolationForestModel


    def execute(self):
        print('\n---------- Pre-processing ----------')
        t0 = time.time()
        pp = self.pre_processor_class(self.args)
        pp_data = pp.pre_process()
        print(f'[+] Pre-process: {round(time.time() - t0, 2)}s')

        print('\n------- Data Transformation -------')
        t0 = time.time()
        transformed_data = self.data_transformer(pp_data)
        print(f'[+] Transform: {round(time.time() - t0, 2)}s')

        print('\n------- Training/Validation -------')
        t0 = time.time()
        model = self.model_class(self.args, data_set=transformed_data)
        model.train_validate()
        print(f'[+] Train/Validate: {round(time.time() - t0, 2)}s')
