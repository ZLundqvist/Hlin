import csv
from sklearn.metrics import confusion_matrix, accuracy_score

from util.filesystem import ensure_output_dir

class EvalResult:
    def __init__(self, input_file: str, true_y: list, pred_y: list):
        self.input_file = input_file
        self.true_y = true_y
        self.pred_y = pred_y
        self.calc()

    def calc(self):
        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(self.true_y, self.pred_y, labels=['N', 'A']).ravel()

        true_positive_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / (false_positives + true_negatives)

        acc = accuracy_score(self.true_y, self.pred_y, normalize=True)

        self.calculated_results = {
            'input_file': self.input_file,
            'tn': true_negatives,
            'fp': false_positives,
            'fn': false_negatives,
            'tp': true_positives,
            'tpr': true_positive_rate,
            'fpr': false_positive_rate,
            'acc': acc
        }

    def get_results(self) -> dict:
        return self.calculated_results
    
    def pretty_print(self):
        print(f'TPR: {round(self.calculated_results["tpr"] * 100, 3)}%')
        print(f'FPR: {round(self.calculated_results["fpr"] * 100, 3)}%')
        print(f'ACC: {round(self.calculated_results["acc"] * 100, 3)}%')

