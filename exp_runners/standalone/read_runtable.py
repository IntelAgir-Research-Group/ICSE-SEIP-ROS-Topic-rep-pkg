import csv
import argparse

def read_csv_and_generate_vector(csv_filename):
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if row['__done'] == 'TODO': 
                vector = [
                    row['__run_id'],
                    row['__done'],
                    row['msg_type'],
                    row['msg_interval'],
                    row['msg_size'],
                    row['exec_time'],
                    row['cli']
                ]
                print(' '.join(vector))

def main():
    parser = argparse.ArgumentParser(description="Read a CSV file and print rows where __done is TODO")
    parser.add_argument('csv_file', type=str, help="Path to the CSV file")
    
    args = parser.parse_args()
    
    read_csv_and_generate_vector(args.csv_file)

if __name__ == "__main__":
    main()