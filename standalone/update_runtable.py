import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Update the __done column for a specific __run_id in the CSV")
parser.add_argument('csv_file', type=str, help="Path to the CSV file")
parser.add_argument('run_id', type=str, help="The __run_id for which to set __done to DONE")
args = parser.parse_args()

df = pd.read_csv(args.csv_file)

df.loc[df['__run_id'] == args.run_id, '__done'] = 'DONE'

df.to_csv(args.csv_file, index=False)

#print(df)
