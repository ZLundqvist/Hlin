import argparse
import numpy as np
from sklearn.ensemble import IsolationForest
import multiprocessing
from scipy import sparse
import pandas as pd
import math
from util.eval_result import EvalResult


class IsolationForestModel:
    def __init__(self, args, input_file: str, data_set):
        self.n_estimators = args.n_estimators
        self.contamination = data_set['contamination']
        self.input_file = input_file
        self.cont_mode = args.cont_mode
        if self.cont_mode == 'custom':
            self.c_floor = args.c_floor
            self.c_ceiling = args.c_ceiling

        # Convert matrix to sparse (way faster)
        df = data_set['df'].drop(['label'], axis=1)
        sparse_matrix = sparse.csr_matrix(df)
        self.X_sparse = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
        self.y = data_set['df']['label'].values

        self.n_jobs = self.calculate_njobs(df=df)

    def calculate_njobs(self, df):
        cpus = multiprocessing.cpu_count()
        # Scale down with increasing data set
        # Not needed when using sparse matrices
        # million_rows = max(math.floor(len(df) / 10**6), 1)
        # cpus = max(math.floor(cpus / million_rows), 3)

        print(f'[+] Using cores: {cpus}')

        return cpus

    def train_validate(self):
        print('[+] Fitting')
        if self.cont_mode == 'custom':
            self.contamination = max(min(self.contamination, self.c_ceiling), self.c_floor)
            isolation_forest = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=0, n_jobs=self.n_jobs).fit(self.X_sparse)

        elif self.cont_mode == 'auto':
            isolation_forest = IsolationForest(n_estimators=self.n_estimators, random_state=0, n_jobs=self.n_jobs).fit(self.X_sparse)

        elif self.cont_mode == 'exact':
            isolation_forest = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=0, n_jobs=self.n_jobs).fit(self.X_sparse)

        print('[+] Predicting')
        y_pred = isolation_forest.predict(self.X_sparse)

        y_pred = np.where(y_pred == -1, 'A', 'N')   # Get predicted labels

        eval_results = EvalResult(input_file=self.input_file, true_y=self.y, pred_y=y_pred)
        eval_results.pretty_print()
        return eval_results


    # Returns the static portion of the model id (filename not included)
    @staticmethod
    def get_static_id(args):
        if args.cont_mode == 'custom':
            return f'iForest_{args.n_estimators}_{args.cont_mode}_{args.c_floor}_{args.c_ceiling}'
        else:
            return f'iForest_{args.n_estimators}_{args.cont_mode}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--n-estimators', dest='n_estimators', default=100, type=int)
        argparser.add_argument('--c-mode', dest='cont_mode', default='exact', choices=['exact', 'auto', 'custom'])
        argparser.add_argument('--cc-floor', dest='c_floor', default='0.01', type=float)
        argparser.add_argument('--cc-ceil', dest='c_ceiling', default='0.05', type=float)