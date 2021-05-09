import time

from pre_processors.frequency_vector import FrequencyVectorPreProcessor
from models.knn import KNNModel
from util.args import pretty_print_args
from util.data_transformers import frequency_vector_knn

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
