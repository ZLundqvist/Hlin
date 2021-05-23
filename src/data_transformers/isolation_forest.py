import pandas as pd

def from_frequency_vector(df: pd.DataFrame):
    print(f'[+] Transforming {len(df)} frequency vector bags for Isolation Forest')

    # Isolation Forest expects a contamination values
    # "The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples."
    # We actually know that as we have a labelled dataset, so calculate
    contamination = df['label'].value_counts(normalize=True)['A']

    df = df.drop(['timestamp'], axis=1)

    print(f'[+] Contamination: {round(contamination* 100, 2)}%')
    return {
        'df': df,
        'contamination': contamination
    }

def from_sliding_window(df: pd.DataFrame):
    print(f'[+] Transforming {len(df)} sliding window bags for Isolation Forest')

    # Isolation Forest expects a contamination values
    # "The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples."
    # We actually know that as we have a labelled dataset, so calculate
    contamination = df['label'].value_counts(normalize=True)['A']

    print(f'[+] Contamination: {round(contamination* 100, 2)}%')
    return {
        'df': df,
        'contamination': contamination
    }

def from_n_gram(df: pd.DataFrame):
    print(f'[+] Transforming {len(df)} n-grams for Isolation Forest')

    # Isolation Forest expects a contamination values
    # "The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples."
    # We actually know that as we have a labelled dataset, so calculate
    contamination = df['label'].value_counts(normalize=True)['A']

    print(f'[+] Contamination: {round(contamination* 100, 2)}%')
    return {
        'df': df,
        'contamination': contamination
    }