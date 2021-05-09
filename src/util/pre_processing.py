import re
import time

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

def line_to_system_call(line: str) -> dict:
    parts = line.split(' ')
    assert len(parts) == 3

    # Rarely Sysdig logs the system call name as <unknown>
    if parts[2] == '<unknown>':
        return None

    return {
        'label': parts[0],
        'timestamp': float(parts[1]),
        'name': parts[2]
    }

def get_unique_calls(system_calls: list) -> set: 
    system_call_name_set = set()

    for system_call in system_calls:
        system_call_name_set.add(system_call['name'])

    return system_call_name_set

# Maps a set of unique system calls to indexes
def get_system_call_index_map(unique_calls: set) -> dict:
    unique_calls_list = list(unique_calls)
    sorted_unique_calls_list = sorted(unique_calls_list)

    mapping = dict()
    for i in range(0, len(sorted_unique_calls_list)):
        mapping[sorted_unique_calls_list[i]] = i

    return mapping
