import pandas as pd
import numpy as np

# TODO -> Move all in here to module in src/data_transformers/knn.py

def n_grams_to_dataframe(n_grams: list):
    # Prepare data for convertion to a Dataframe
    copied_n_grams = []
    for n_gram in n_grams:
        copied_n_gram = n_gram['ngrams'].copy()
        copied_n_gram['label'] = n_gram['label']
        copied_n_gram['timestamp'] = n_gram['timestamp']

        copied_n_grams.append(copied_n_gram)

    df = pd.DataFrame(copied_n_grams)

    # open  close   open    close
    #  0      1       1        0

    # Get all columns except 2
    columns = list(df.columns)
    columns.remove('label')
    columns.remove('timestamp')

    # Drop dupliactes based on above columns
    df_no_dups = df.drop_duplicates(subset=columns, ignore_index=True)
    duplicates_dropped = len(df) - len(df_no_dups)

    # # Check that data contains anomalous values as well (sanity check more or less)
    # # if not 'A' in df_no_dups['']
    if not 'A' in df_no_dups['label'].values:
        raise Exception('Data contains no anomalous values, this is not supported')

    return df_no_dups, duplicates_dropped

def n_gram_knn(n_grams: list):
    print(f'[+] Transforming {len(n_grams)} n-grams for KNN')

    df, duplicates_dropped = n_grams_to_dataframe(n_grams)

    print(f'[+] DataFrame created (rows={len(df.index)}, columns={len(df.columns)}, duplicates_dropped={duplicates_dropped})')

    return df

def n_gram_isolation_forest(n_grams: list):
    print(f'[+] Transforming {len(n_grams)} n-grams for Isolation Forest')

    df, duplicates_dropped = n_grams_to_dataframe(n_grams)

    # Isolation Forest expects a contamination values
    # "The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples."
    # We actually know that as we have a labelled dataset, so calculate
    contamination = df['label'].value_counts(normalize=True)['A']

    print(f'[+] Contamination: {round(contamination* 100, 2)}%')
    print(f'[+] DataFrame created (rows={len(df.index)}, columns={len(df.columns)}, duplicates_dropped={duplicates_dropped})')
    return {
        'df': df,
        'contamination': contamination
    }