import os
import json

cache_directory = os.path.join(os.path.dirname(__file__), '../../cache/')
cache_directory = os.path.realpath(cache_directory)

def ensure_cache_dir():
    if not os.path.exists(cache_directory):
        print(f'[+] Creating cache directory: {cache_directory}')
        os.makedirs(cache_directory)

def ensure_file(file_path):
    path = resolve_abs_path(file_path)
    return os.path.exists(file_path)

def resolve_abs_path(file_path):
    if os.path.isabs(file_path):
        return file_path
    else:
        cwd = os.getcwd()
        return os.path.abspath(os.path.join(cwd, file_path))

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
