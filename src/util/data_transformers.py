
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def frequency_vector_to_df(bags: list):
    # Prepare data for convertion to a Dataframe
    copied_bags = []
    for bag in bags:
        copied_bag = bag['bag'].copy()
        copied_bag['label'] = bag['label']
        copied_bag['timestamp'] = bag['timestamp']

        copied_bags.append(copied_bag)

    df = pd.DataFrame(copied_bags)

    # Get all columns except 2
    columns = list(df.columns)
    columns.remove('label')
    columns.remove('timestamp')

    # Drop dupliactes based on above columns
    df_no_dups = df.drop_duplicates(subset=columns, ignore_index=True)
    duplicates_dropped = len(df) - len(df_no_dups)

    return df_no_dups, duplicates_dropped
    

def frequency_vector_knn(bags: list):
    print(f'[+] Transforming {len(bags)} frequency vector bags for KNN')

    df, duplicates_dropped = frequency_vector_to_df(bags)

    print(f'[+] DataFrame created (rows={len(df.index)}, columns={len(df.columns)}, duplicates_dropped={duplicates_dropped})')

    return df

def frequency_vector_isolation_forest(bags: list):
    print(f'[+] Transforming {len(bags)} frequency vector bags for Isolation Forest')

    df, duplicates_dropped = frequency_vector_to_df(bags)

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
