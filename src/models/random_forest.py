import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from util.eval_result import EvalResult

class RandomForestModel:
    def __init__(self, args, input_file: str, data_set: pd.DataFrame):
        self.n_estimators = args.n_estimators
        self.input_file = input_file

        self.prepare_data(data_set=data_set)

    def prepare_data(self, data_set: pd.DataFrame):
        train_df, test_df = train_test_split(data_set, train_size=0.80, random_state=0, stratify=data_set['label'].values) # Stratify data
        # train_df, test_df = train_test_split(data_set, train_size=0.80, random_state=0)

        self.X_train = train_df.drop(['label'], axis=1).values
        self.X_test = test_df.drop(['label'], axis=1).values
        self.y_train = train_df['label'].values
        self.y_test = test_df['label'].values

        print(f'[+] DataFrame split (train_len={len(train_df)}, test_len={len(test_df)})')

    def train_validate(self): 
        clf = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1)

        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)

        eval_results = EvalResult(input_file=self.input_file, true_y=self.y_test, pred_y=y_pred)
        eval_results.pretty_print()
        return eval_results

    # Returns the static portion of the model id (filename not included)
    @staticmethod
    def get_static_id(args):
        return f'rForest_{args.n_estimators}'

    @staticmethod
    def append_args(argparser: argparse.ArgumentParser):
        argparser.add_argument('--n-estimators', dest='n_estimators', default=100, type=int)


