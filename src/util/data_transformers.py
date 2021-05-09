
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def frequency_vector_knn(bags: list):
    print(f'[+] Transforming {len(bags)} bags for KNN')

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

    print(f'[+] DataFrame created (rows={len(df_no_dups.index)}, columns={len(df_no_dups.columns)}, duplicates_dropped={duplicates_dropped})')

    return df_no_dups
