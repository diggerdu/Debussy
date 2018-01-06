import csv
import argparse
import time
import select
import sys
import numpy as np
from util.getPatch import getLabelDict
import os

TABLE = 'yes, no, up, down, left, right, on, off, stop, go, silence, unknown'.split(', ')
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--submissionFile", required=True, help="submission file")
parser.add_argument("-p", "--patchPath", help="patch file Path")
parser.add_argument("-o", "--outputFile", required=True, help="output file")
args = parser.parse_args()

patchDict = getLabelDict(args.patchPath)
print(len(patchDict.keys()))


with open(args.submissionFile, mode='r') as f:
    reader = csv.reader(f)
    labelDict = {rows[0]:rows[1] for rows in reader if rows[0].endswith('.wav')}
    assert len(set(list(labelDict.keys()))) == 158538

mergeDict = dict()

count = 0
for fn, label in labelDict.items():
    try:
        mergeDict.update({fn:patchDict[fn]})
        if patchDict[fn] != label:
            count += 1
    except:
        mergeDict.update({fn: label})

print("relabel {} entries".format(count))

# check twice

assert len(set(list(mergeDict.keys()))) == 158538
for fn, label in mergeDict.items():
    assert label in TABLE

outputFile = open(args.outputFile, "w+")
print('fname,label', file=outputFile)
for fn, label in mergeDict.items():
    print('{},{}'.format(fn, label), file=outputFile)
outputFile.close()

