import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Unsupervised version of KNN
class KNNModel:
    def __init__(self, args, data_set):
        self.k = args.k_neighbours
        self.threshold = args.anomaly_threshold
        self.folds = args.folds

        self.validation_df = data_set[['label', 'timestamp']]
        self.data_df = data_set.drop(['label', 'timestamp'], axis=1)

    def train_validate(self): 
        df = self.data_df
        X = df.values

        nbrs = NearestNeighbors(n_neighbors=self.k)
        nbrs.fit(X)

        # distances is a matrix where the row at index i contains the distances to the k nearest neighbours (for k=5, each row has 5 distances) 
        distances, indexes = nbrs.kneighbors(X)

        # Calculate the mean distance for each value
        distances_mean = distances.mean(axis = 1)

        p = self.threshold
        percentile = np.percentile(distances_mean, p)

        # Get all indicies that are considered anomalous
        outlier_indices = np.where(distances_mean > percentile)

        # Get the validation data frame 
        outlier_df = self.validation_df.iloc[outlier_indices[0],:]
        not_outlier_df = self.validation_df.drop(outlier_df.index, axis=0)

        # Confusion matrix
        positives = np.where(self.validation_df['label'] == 'A')[0]
        negatives = np.where(self.validation_df['label'] == 'N')[0]

        true_positives = outlier_df.loc[outlier_df['label'] == 'A'].index
        false_positives = outlier_df.loc[outlier_df['label'] == 'N'].index

        true_negatives = not_outlier_df.loc[not_outlier_df['label'] == 'N'].index
        false_negatives = not_outlier_df.loc[not_outlier_df['label'] == 'A'].index

        true_positive_rate = len(true_positives) / (len(true_positives) + len(false_negatives))
        false_positive_rate = len(false_positives) / (len(false_positives) + len(true_negatives))

        print(f'TPR: {round(true_positive_rate*100, 2)}%')
        print(f'FPR: {round(false_positive_rate*100, 2)}%')

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--k-neighbours', dest='k_neighbours', type=int, required=True)
        argparser.add_argument('--anomaly-threshold', dest='anomaly_threshold', type=float, required=True)


