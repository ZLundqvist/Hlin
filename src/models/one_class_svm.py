import argparse
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from util.eval_result import EvalResult

class OneClassSVMModel:
    def __init__(self, args, input_file: str, data_set: pd.DataFrame):
        self.kernel = args.kernel
        self.input_file = input_file

        self.X = data_set.drop(['label'], axis=1).values
        self.y = data_set['label'].values

    def train_validate(self): 
        clf = OneClassSVM(kernel=self.kernel, gamma='auto', cache_size=20000).fit(self.X)

        y_pred = clf.predict(self.X)

        y_pred = np.where(y_pred == 1, 'N', 'A')

        eval_results = EvalResult(input_file=self.input_file, true_y=self.y, pred_y=y_pred)
        eval_results.pretty_print()
        return eval_results

    # Returns the static portion of the model id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'ocsvm_{args.kernel}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--kernel', dest='kernel', default='rbf', choices=['rbf', 'linear', 'poly', 'sigmoid'])


