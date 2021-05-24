import pandas as pd
import glob
import os

OUTPUT_FILE = 'output_combined.csv'

assert not os.path.isfile(OUTPUT_FILE), 'Output file already exists'

extension = 'csv'
all_filenames = [i for i in glob.glob('./output_final/*.{}'.format(extension))]

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f'Combined {len(all_filenames)} output files')