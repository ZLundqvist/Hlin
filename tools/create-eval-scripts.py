
N_GRAM_SIZES = [3, 5, 10]
DELTA_TS = [50, 100, 200]
N_ESTIMATORS = [50, 100, 200]
SLIDING_WINDOWS = [(11, 11), (11, 6), (23, 12), (7, 4), (10, 5)] # (size, step_size)

def sliding_window():
    params = [f'--window-size {pair[0]} --window-step-size {pair[1]}' for pair in SLIDING_WINDOWS]
    
    return {
        'name': 'sliding_window',
        'params': params
    }

def n_gram():
    params = [f'--ngram-size {size}' for size in N_GRAM_SIZES]
    return {
        'name': 'n_gram',
        'params': params
    }

def frequency_vector():
    params = [f'--delta-t {T}' for T in DELTA_TS]
    return {
        'name': 'frequency_vector',
        'params': params
    }

def isolation_forest():

    return {
        'name': 'isolation_forest',
        'params': 0
    }

preprocessors = [
    sliding_window(),
    n_gram(),
    frequency_vector()
]

models = [
    isolation_forest()
]

print(preprocessors)