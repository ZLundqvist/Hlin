import pandas as pd
import glob
import os
import argparse


def run(args):
    output_file = f'{args.folder}-combined.csv'
    assert not os.path.isfile(output_file), 'Output file already exists'

    all_filenames = [i for i in glob.glob(f'{args.folder}/*.csv')]

    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

    combined_csv.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f'Combined {len(all_filenames)} output files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', dest='folder', required=True)

    args = parser.parse_args()
    run(args)
