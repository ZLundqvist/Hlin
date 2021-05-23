import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

from util.eval_result import EvalResult

# Unsupervised version of KNN
class KNNModel:
    def __init__(self, args, input_file: str, data_set):
        self.k = args.k_neighbours
        self.threshold = args.anomaly_threshold
        self.input_file = input_file

        self.validation_df = data_set[['label']]
        self.df = data_set.drop(['label'], axis=1)

    def train_validate(self): 
        df = self.df
        X = df.values

        nbrs = NearestNeighbors(n_neighbors=self.k)
        nbrs.fit(X)

        # distances is a matrix where the row at index i contains the distances to the k nearest neighbours (for k=5, each row has 5 distances) 
        distances, indexes = nbrs.kneighbors(X)

        # Calculate the mean distance for each value
        distances_mean = distances.mean(axis = 1)

        # Calculate the value for top percentile p
        p = self.threshold
        threshold_value = np.percentile(distances_mean, 100 - p)
        
        outlier_true = self.validation_df['label'].values               # Get true (actual) labels
        outlier_pred = np.where(distances_mean > threshold_value, 'A', 'N')  # Get predicted labels

        eval_results = EvalResult(input_file=self.input_file, true_y=outlier_true, pred_y=outlier_pred)
        eval_results.pretty_print()
        return eval_results

    # Returns the static portion of the model id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'knn_{args.k_neighbours}_{args.anomaly_threshold}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--k-neighbours', dest='k_neighbours', type=int, default=5)
        argparser.add_argument('--anomaly-threshold', dest='anomaly_threshold', type=int, default=10)


