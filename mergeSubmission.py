import csv
import soundfile as sf
import sounddevice as sd
import argparse
import time
import select
import sys
import numpy as np
from util.getPatch import getLabelDict
import os

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--submissionFile", required=True, help="submission file")
parser.add_argument("-p", "--patchPath", help="patch file Path")
parser.add_argument("-o", "--outputFile", required=True, help="output file")
args = parser.parse_args()

patchDict = getLabelDict(args.patchPath)


with open(args.submissionFile, mode='r') as f:
    reader = csv.reader(f)
    labelDict = {rows[0]:rows[1] for rows in reader if rows[0].endswith('.wav')}
    
    assert len(set(labelDict.keys())) == 158538

mergeDict = dict()
count = 0
for fn, label in labelDict.items():
    try:
        mergeDict.update({fn:patchDict[fn]})
        count += 1
    except:
        mergeDict.update({fn: label})

print("relabel {} entries".format(count))

# check twice

assert len(set(mergeDict.keys())) == 158538

outputFile = open(args.outputFile, "w+")
print('fname,label', file=outputFile)
for fn, label in mergeDict.items():
    print('{},{}'.format(fn, label), file)
outputFile.close()

