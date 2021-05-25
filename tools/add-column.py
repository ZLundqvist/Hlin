import csv
import argparse
import glob

def run(folder: str, column: str, value):
    files = [i for i in glob.glob(f'{folder}/*.csv')]

    if not input(f'Adding column {column} with value {value} to {len(files)} files in folder {folder}. Continue? [y/n]') == 'y':
        return

    for file in files:
        run_file(file, column, value)

def run_file(file: str, column: str, value):
    rows = []
    with open(file, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            assert not row.get('iteration'), f'Row in file {file} already has column {column}'
            rows.append(row)

        assert len(rows) > 0

    for row in rows:
        row[column] = value

    with open(file, encoding='utf-8-sig', mode='w') as csvfile:
        keys = rows[0].keys()

        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', dest='folder', required=True)
    parser.add_argument('--column', '-c', dest='column', required=True)
    parser.add_argument('--value', '-v', dest='value', required=True)
    args = parser.parse_args()

    run(args.folder, args.column, args.value)