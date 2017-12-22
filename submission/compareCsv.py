import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f0", "--csvFile0", required=True, help="csv file 0")
parser.add_argument("-f1", "--csvFile1", required=True, help="csv file 1")
args = parser.parse_args()

with open(args.csvFile0, mode='r') as f:
    reader0 = csv.reader(f)
    labelDict0= {row[0]:row[1] for row in reader0}

with open(args.csvFile1, mode='r') as f:
    reader1 = csv.reader(f)
    diffList = [row for row in reader1 if labelDict0[row[0]] != row[1]]

for row in diffList:
    print('{}:{},{}'.format(row[0], row[1], labelDict0[row[0]]))




