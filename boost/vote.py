import csv
import argparse
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--item', action='store', dest='alist', type=str, nargs='*', default=['item1', 'item2', 'item3'], help="Examples: -i item1 item2, -i item3")
parser.add_argument('-o', '--output', help='output file')
args = parser.parse_args()

labelDict = defaultdict(list)
for fn in args.alist:
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].endswith('.wav'):
                labelDict[row[0]].append(row[1])

with open(args.output, 'w') as f:
    print('fname,label', file=f)
    for fn, labelList in labelDict.items():
        print('{},{}'.format(fn, Counter(labelList).most_common(1)[0][0]), file=f)



