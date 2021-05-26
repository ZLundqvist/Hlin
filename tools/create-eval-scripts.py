from multiprocessing import get_all_start_methods
import inquirer
import numpy as np

dd_mode = 'none'
input_dir = './input/'

iforest_param_values = {
    'n-estimators': [50, 100, 200, 500],
    'c-mode': ['exact', 'auto', 'custom'],
    'cc-floor': [ .01, .02, .03, .04, .05, .06, .07 ],
    'cc-ceil': [ .01, .02, .03, .04, .05, .06, .07 ]
}

sliding_window_param_values = {
    'window_params': [(11, 11), (10, 5), (23, 12), (11, 6), (7, 4), (11, 5), (50, 26)] # (size, step_size)
}

frequency_vector_param_values = {
    'delta_t': [50, 100, 200]
}

n_gram_param_values = {
    'n': [1, 3, 5, 7, 10, 20]
}

def query_n_gram():
    questions = [ 
        inquirer.Checkbox('n', message='n-gram param: n', choices=n_gram_param_values.get('n')),
    ]
    answers = inquirer.prompt(questions)

    pp_combos = []
    for n in answers.get('n'):
        pp_combos.append(f'-p n_gram --ngram-size {n}')

    return pp_combos

def query_frequency_vector():
    questions = [ 
        inquirer.Checkbox('delta_t', message='FV param: delta-t', choices=frequency_vector_param_values.get('delta_t')),
    ]
    answers = inquirer.prompt(questions)

    pp_combos = []
    for delta_t in answers.get('delta_t'):
        pp_combos.append(f'-p frequency_vector --delta-t {delta_t}')

    return pp_combos

def query_sliding_window():
    questions = [ 
        inquirer.Checkbox('window_params', message='SW param: <size, step_size>', choices=[f'{window_params[0]},{window_params[1]}' for window_params in sliding_window_param_values.get('window_params')]),
    ]
    answers = inquirer.prompt(questions)

    pp_combos = []
    for window_params in answers.get('window_params'):
        size, step_size = window_params.split(',')
        pp_combos.append(f'-p sliding_window --window-size {size} --window-step-size {step_size}')

    return pp_combos

def query_isolation_forest():
    questions = [ 
        inquirer.Checkbox('n-estimators', message='iForest param: n-estimators', choices=iforest_param_values.get('n-estimators')),
        inquirer.Checkbox('c-mode', message='iForest param: c-mode', choices=iforest_param_values.get('c-mode'))
    ]
    answers = inquirer.prompt(questions)

    if 'custom' in answers.get('c-mode'):
        cc_questions = [
            inquirer.Checkbox('cc-floor', message='iForest param: cc-floor', choices=iforest_param_values.get('cc-floor')),
            inquirer.Checkbox('cc-ceil', message='iForest param: cc-ceil', choices=iforest_param_values.get('cc-ceil'))
        ]
        cc_answers = inquirer.prompt(cc_questions)

    model_combos = []
    for n_estimators in answers.get('n-estimators'):
        for c_mode in answers.get('c-mode'):
            if c_mode == 'custom':
                for cc_floor in cc_answers.get('cc-floor'):
                    for cc_ceil in cc_answers.get('cc-ceil'):
                        model_combos.append(f'-m isolation_forest --n-estimators {n_estimators} --c-mode {c_mode} --cc-floor {cc_floor} --cc-ceil {cc_ceil}')
            else:
                model_combos.append(f'-m isolation_forest --n-estimators {n_estimators} --c-mode {c_mode}')

    return model_combos

def query():
    model_combos = []
    preprocessor_combos = []
    questions = [
        inquirer.Text('iteration', message="Iteration?"),
        inquirer.Checkbox('models', message="Models to include", choices=['isolation_forest']),
        inquirer.Checkbox('preprocessors', message="Pre-processors to include", choices=['sliding_window', 'frequency_vector', 'n_gram']),
    ]
    answers = inquirer.prompt(questions)

    if 'isolation_forest' in answers.get('models'):
        model_combos.append(query_isolation_forest())

    if 'sliding_window' in answers.get('preprocessors'):
        preprocessor_combos.append(query_sliding_window())
    if 'frequency_vector' in answers.get('preprocessors'):
        preprocessor_combos.append(query_frequency_vector())
    if 'n_gram' in answers.get('preprocessors'):
        preprocessor_combos.append(query_n_gram())

    # Flatten
    model_combos = [item for sublist in model_combos for item in sublist]
    preprocessor_combos = [item for sublist in preprocessor_combos for item in sublist]

    cmds = []
    for model_combo in model_combos:
        for preprocessor_combo in preprocessor_combos:
            cmds.append(f'python src/core.py {model_combo} {preprocessor_combo} --dd {dd_mode} --input-dir {input_dir} --iteration {answers.get("iteration")}\n')

    return cmds



if __name__ == '__main__':
    cmds = query()
    print(f'{len(cmds)} commands created')
    output = input('Enter output file name > ')

    with open(output, 'w') as file:
        file.writelines(cmds)
