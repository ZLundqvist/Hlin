import csv
from sklearn.metrics import confusion_matrix, accuracy_score
import os

from util.filesystem import ensure_output_dir

class EvalResult:
    def __init__(self, input_file: str, true_y: list, pred_y: list):
        self.input_file = input_file
        self.true_y = true_y
        self.pred_y = pred_y
        self.calc()

    def calc(self):
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(self.true_y, self.pred_y, labels=['N', 'A']).ravel()

        true_positive_rate = true_positives / max((true_positives + false_negatives), 1)
        false_positive_rate = false_positives / max((false_positives + true_negatives), 1)

        acc = accuracy_score(self.true_y, self.pred_y, normalize=True)

        self.calculated_results = {
            'run_id': None, # TB inserted
            'input_file': os.path.basename(self.input_file).split('.')[0],
            'tn': true_negatives,
            'fp': false_positives,
            'fn': false_negatives,
            'tp': true_positives,
            'tpr': true_positive_rate,
            'fpr': false_positive_rate,
            'acc': acc
        }

    def get_results(self, run_id: str) -> dict:
        results_with_run_id = self.calculated_results.copy()
        results_with_run_id['run_id'] = run_id
        results_with_run_id['iteration'] = self.iteration

        return results_with_run_id

    def set_iteration(self, iteration: int):
        self.iteration = iteration
    
    def pretty_print(self):
        print(f'[+] TPR: {self.calculated_results["tpr"]}')
        print(f'[+] FPR: {self.calculated_results["fpr"]}')
        print(f'[+] ACC: {self.calculated_results["acc"]}')

