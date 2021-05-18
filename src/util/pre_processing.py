import time
import pandas as pd
from decimal import Decimal

from .filesystem import resolve_abs_path

# Reads the input file and returns a list of system call dict's
def read_log_file(file_path: str) -> list:
    file_path = resolve_abs_path(file_path)
    print(f'[+] Reading logfile: {file_path}')
    t0 = time.time()

    system_calls = []
    with open(file_path) as file:
        for line in file:
            system_call = line_to_system_call(line.rstrip())
            if system_call:
                system_calls.append(system_call)


    print(f'[+] Read complete: {round(time.time() - t0, 2)}s')
    return system_calls

def drop_duplicates(df: pd.DataFrame, mode: str):
    original_len = len(df)
    columns = list(df.columns)

    if mode == 'none':
        return df, 0
    elif mode == 'label':
        if 'timestamp' in columns:
            columns.remove('timestamp')
        df.drop_duplicates(subset=columns, ignore_index=True, inplace=True)
        duplicates_dropped = original_len - len(df)
        return df, duplicates_dropped
    elif mode == 'first':
        columns.remove('label')
        if 'timestamp' in columns:
            columns.remove('timestamp')
        df.drop_duplicates(subset=columns, ignore_index=True, inplace=True)
        duplicates_dropped = original_len - len(df)
        return df, duplicates_dropped
    else:
        raise Exception(f'Invalido drop_duplicates mode: {mode}')

def get_system_calls_metadata(file_path: str):
    print(f'[+] Reading metadata: {file_path}')
    t0 = time.time()
    unique_calls = []
    num_calls = 0

    for system_call in get_system_calls(file_path=file_path):
        num_calls = num_calls + 1

        if system_call['name'] not in unique_calls:
            unique_calls.append(system_call['name'])

    print(f'[+] Read complete: {round(time.time() - t0, 2)}s')
    return num_calls, unique_calls

def get_system_calls(file_path: str):
    file_path = resolve_abs_path(file_path)

    with open(file_path) as file:
        for line in file:
            system_call = line_to_system_call(line.rstrip())
            if system_call:
                yield system_call
                
def line_to_system_call(line: str) -> dict:
    parts = line.split(' ')
    assert len(parts) == 3

    # Rarely Sysdig logs the system call name as <unknown>
    if parts[2] == '<unknown>':
        return None

    return {
        'label': parts[0],
        'timestamp': Decimal(parts[1]),
        'name': parts[2]
    }
