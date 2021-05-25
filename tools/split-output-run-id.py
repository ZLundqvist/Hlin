import argparse
import csv
from typing import OrderedDict

def split_column_iForest(row):
    run_id: str = row.get('run_id')
        
    new_row = { 'run_id': run_id } 
    run_id = run_id.split('_')

    idx = 0

    new_row['model'] = run_id[idx]
    idx += 1
    assert new_row['model'] == 'iForest'

    new_row['n_estimators'] = run_id[idx]
    idx += 1

    new_row['cc_mode'] = run_id[idx]
    idx += 1

    if new_row['cc_mode'] == 'custom':
        new_row['cc_floor'] = run_id[idx]
        idx += 1
        new_row['cc_ceil'] = run_id[idx]
        idx += 1

    new_row['preprocessor'] = run_id[idx]
    idx += 1

    if new_row['preprocessor'] == 'fv':
        new_row['delta_t'] = run_id[idx]
        idx += 1
    elif new_row['preprocessor'] == 'sw':
        new_row['window_size'] = run_id[idx]
        idx += 1
        new_row['window_step_size'] = run_id[idx]
        idx += 1
    elif new_row['preprocessor'] == 'n':
        # n-gram identifier is "n_gram", which means the split returns [..., "n", "gram", ...]
        # Recombined this correctly
        new_row['preprocessor'] = new_row['preprocessor'] + '_' + run_id[idx] 
        idx += 1
        new_row['n_gram_size'] = run_id[idx]
        idx += 1

    new_row['label'] = run_id[idx]

    # Combine new_row with row (so that all eval results are included)
    new_row = { **new_row, **row }

    # Add Detected value
    new_row['detected'] = True if float(new_row['tpr']) > 0 else False 

    return new_row

def split_column_knn(row):
    run_id: str = row.get('run_id')
        
    new_row = { 'run_id': run_id } 
    run_id = run_id.split('_')

    idx = 0

    new_row['model'] = run_id[idx]
    idx += 1
    assert new_row['model'] == 'knn'

    new_row['k-neighbours'] = run_id[idx]
    idx += 1

    new_row['threshold'] = run_id[idx]
    idx += 1

    new_row['preprocessor'] = run_id[idx]
    idx += 1
    assert new_row['preprocessor'] == 'fv'

    new_row['delta_t'] = run_id[idx]
    idx += 1

    new_row['label'] = run_id[idx]

    # Combine new_row with row (so that all eval results are included)
    new_row = { **new_row, **row }

    # Add Detected value
    new_row['detected'] = True if float(new_row['tpr']) > 0 else False 

    return new_row

def split_columns(file):
    rows = []
    with open(file, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            assert row.get('run_id') is not None
            rows.append(row)
        
    new_rows = []
    for row in rows:
        model = row.get('run_id').split('_')[0]
        if model == 'iForest':
            new_rows.append(split_column_iForest(row))
        elif model == 'knn':
            new_rows.append(split_column_knn(row))

    with open('transformed_' + file, 'w', encoding='utf-8-sig') as csvfile:
        all_keys = [
            'run_id', 
            'model',
            'n_estimators',
            'k-neighbours',
            'threshold',
            'preprocessor',
            'delta_t',
            'window_size',
            'window_step_size',
            'n_gram_size',
            'cc_mode',
            'cc_floor',
            'cc_ceil',
            'label',
            'input_file',
            'tn',
            'fp',
            'fn',
            'tp',
            'tpr',
            'fpr',
            'acc',
            'detected',
            'iteration'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(new_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', dest='file', required=True)
    args = parser.parse_args()

    split_columns(args.file)