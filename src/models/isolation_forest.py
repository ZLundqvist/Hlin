import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest

from util.eval_result import EvalResult

# Unsupervised version of KNN
class IsolationForestModel:
    def __init__(self, args, input_file: str, data_set):
        self.n_estimators = args.n_estimators
        self.contamination = data_set['contamination']
        self.input_file = input_file

        df = data_set['df']
        self.X = df.drop(['label'], axis=1).values
        self.y = df['label'].values

    def train_validate(self): 
        if self.contamination < 0.01:
            self.contamination = 0.01
        elif self.contamination > 0.05:
            self.contamination = 0.05
        isolation_forest = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=0, n_jobs=2).fit(self.X)
        # isolation_forest = IsolationForest(n_estimators=self.n_estimators, random_state=0, n_jobs=-1).fit(X)

        y_pred = isolation_forest.predict(self.X)

        y_pred = np.where(y_pred == -1, 'A', 'N')   # Get predicted labels

        eval_results = EvalResult(input_file=self.input_file, true_y=self.y, pred_y=y_pred)
        eval_results.pretty_print()
        return eval_results

    # Returns the static portion of the model id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'isolation_forest_{args.n_estimators}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--n-estimators', dest='n_estimators', default=100, type=int)
