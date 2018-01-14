import csv
import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f0", "--conf0", required=True, help="confidence file 0")
parser.add_argument("-f1", "--conf1", required=True, help="confidence file 1")
parser.add_argument("-o", "--out", type=float, help="the output ratio confidence")
args = parser.parse_args()

Table = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

confData0 = pd.read_csv(args.conf0, index_col=0, header=None)
confData1 = pd.read_csv(args.conf1, index_col=0, header=None)

fileList = set(confData0.index.values.tolist())
fileList1 = set(confData0.index.values.tolist())

assert len(fileList) == 158538
assert len(fileList1) == 158538

ratioList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
subFileList = [open("sub_{}".format(ratio), 'w') for ratio in ratioList]
for f in subFileList:
    print('fname,label', file=f)
if args.out is not None:
    mixConfFile = open('conf_{}.csv'.format(args.out), 'a')

for fn in fileList:
    conf0 = confData0.loc[fn].values
    conf1 = confData1.loc[fn].values
    for ratio, f in zip(ratioList, subFileList):
        mixConf = ratio * conf0 + (1-ratio) * conf1
        try:
            label = Table[np.argmax(mixConf)]
        except:
            import pdb; pdb.set_trace()

        print('{},{}'.format(fn, label), file=f)
        if args.out is not None and ratio == args.out:
            print(fn + ',' + ','.join(map(str,mixConf.tolist())), file=mixConfFile)

for f in subFileList:
    f.close()

try:
    mixConfFile.close()
except:
    pass











