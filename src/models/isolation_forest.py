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
        self.validation_df = df[['label', 'timestamp']]
        self.df = df.drop(['label', 'timestamp'], axis=1)

    def train_validate(self): 
        df = self.df
        X = df.values

        isolation_forest = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=0).fit(X)

        outlier_pred = isolation_forest.predict(X)

        outlier_true = self.validation_df['label'].values       # Get true (actual) labels
        outlier_pred = np.where(outlier_pred == -1, 'A', 'N')   # Get predicted labels

        eval_results = EvalResult(input_file=self.input_file, true_y=outlier_true, pred_y=outlier_pred)
        eval_results.pretty_print()
        return eval_results

    # Returns the static portion of the model id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'knn_{args.n_estimators}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--n-estimators', dest='n_estimators', default=100, type=int)
