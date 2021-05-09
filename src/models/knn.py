import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Unsupervised version of KNN
class KNNModel:
    def __init__(self, args, data_set):
        self.k = args.k_neighbours
        self.threshold = args.anomaly_threshold
        self.folds = args.folds

        self.validation_df = data_set[['label', 'timestamp']]
        self.df = data_set.drop(['label', 'timestamp'], axis=1)

    def train_validate(self): 
        df = self.df
        X = df.values

        nbrs = NearestNeighbors(n_neighbors=self.k)
        nbrs.fit(X)

        # distances is a matrix where the row at index i contains the distances to the k nearest neighbours (for k=5, each row has 5 distances) 
        distances, indexes = nbrs.kneighbors(X)

        # Calculate the mean distance for each value
        distances_mean = distances.mean(axis = 1)

        # Calculate the value for percentile p
        p = self.threshold
        percentile = np.percentile(distances_mean, p)
        
        outlier_true = self.validation_df['label'].values               # Get true (actual) values
        outlier_pred = np.where(distances_mean > percentile, 'A', 'N')  # Get predicted values

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
        argparser.add_argument('--k-neighbours', dest='k_neighbours', type=int, required=True)
        argparser.add_argument('--anomaly-threshold', dest='anomaly_threshold', type=float, required=True)


