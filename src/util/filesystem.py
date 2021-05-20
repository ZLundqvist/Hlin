import os
import json
import pandas
import csv

cache_directory = os.path.join(os.path.dirname(__file__), '../../cache/')
cache_directory = os.path.realpath(cache_directory)

output_directory = os.path.join(os.path.dirname(__file__), '../../output/')
output_directory = os.path.realpath(output_directory)

def ensure_cache_dir():
    if not os.path.exists(cache_directory):
        print(f'[+] Creating cache directory: {cache_directory}')
        os.makedirs(cache_directory)

def ensure_output_dir():
    if not os.path.exists(output_directory):
        print(f'[+] Creating output directory: {output_directory}')
        os.makedirs(output_directory)

def ensure_file(file_path):
    path = resolve_abs_path(file_path)
    return os.path.exists(path)

def resolve_abs_path(file_path):
    if os.path.isabs(file_path):
        return file_path
    else:
        cwd = os.getcwd()
        return os.path.abspath(os.path.join(cwd, file_path))

def does_eval_results_file_exist(run_id: str):
    ensure_output_dir()
    output_filename = os.path.join(output_directory, run_id + '.csv')
    return ensure_file(output_filename)

# Gets the absolute path to every file in the passed directory
def get_dir_files_abs(dir: str):
    abs_path = resolve_abs_path(dir)
    files = [f for f in os.listdir(abs_path) if os.path.isfile(os.path.join(abs_path, f))]
    abs_files = [os.path.join(abs_path, f) for f in files]
    abs_files.sort()
    return abs_files

def write_eval_results_to_csv(run_id: str, eval_results: list):
    ensure_output_dir()
    output_filename = os.path.join(output_directory, run_id + '.csv')

    result_dicts = [eval_result.get_results(run_id) for eval_result in eval_results]

    keys = result_dicts[0].keys()

    with open(output_filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result_dicts)

    print(f'\n\n[+] Output written to {output_filename}')
    

def write_cache_json(id: str, data):
    file = os.path.join(cache_directory, id)
    f = open(file, 'w')
    f.write(json.dumps(data, indent=4))
    f.close()

    print(f'[+] Results cached: {file}')
    
def read_cache_json(id: str) -> list:
    file = os.path.join(cache_directory, id)

    if os.path.exists(file):
        print(f'[+] Found cached results: {file}')
        f = open(file)
        data = json.load(f)
        assert type(data) == list
        return data
    
    return None

def write_cache_pickle(id: str, df: pandas.DataFrame):
    print(f'[+] Writing DataFrame to cache')
    file = os.path.join(cache_directory, id + '.gz')
    df.to_pickle(path=file, compression='gzip')
    print(f'[+] DataFrame cached: {file}')

def read_cache_pickle(id: str) -> pandas.DataFrame:
    file = os.path.join(cache_directory, id + '.gz')

    if os.path.exists(file):
        print(f'[+] Found cached DataFrame: {file}')
        return pandas.read_pickle(file, compression='gzip')
    
    return None