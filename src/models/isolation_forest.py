import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest

# Unsupervised version of KNN
class IsolationForestModel:
    def __init__(self, args, data_set):
        self.n_estimators = args.n_estimators
        self.contamination = data_set['contamination']

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

       # Calcuate TPR/FPR using a confusion matrix
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(outlier_true, outlier_pred, labels=['N', 'A']).ravel()
        true_positive_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / (false_positives + true_negatives)

        # Calculate accuracy
        accuracy = accuracy_score(outlier_true, outlier_pred, normalize=True)

        print(f'ACC: {round(accuracy*100, 2)}%')
        print(f'TPR: {round(true_positive_rate*100, 2)}%')
        print(f'FPR: {round(false_positive_rate*100, 2)}%')

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--n-estimators', dest='n_estimators', default=100, type=int)
