import csv
import soundfile as sf
import sounddevice as sd
import argparse
import time
import select
import sys
import numpy as np
from util.getPatch import getPatch

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--csvFile", required=True, help="csv file")
parser.add_argument("-f1", "--csvFile1", help="csv file 1")
parser.add_argument("-l", "--label", help="label")
parser.add_argument("-r", "--root", help="root path")
parser.add_argument("-o", "--outputFile", required=True, help="output file")
parser.add_argument("-s", "--speed", type=int, required=True, help="speed")
args = parser.parse_args()

configIntension = {'timeout': 2}
configMiddle = {'timeout': 6}
configSlow = {'timeout': 10}

configTable = [configIntension, configMiddle, configSlow]
config = configTable[args.speed]


patchList = getPatch('./patch')


def timeout_input(timeout=6):
    print('waiting for input:')
    i, _, _ = select.select( [sys.stdin], [], [], timeout)
    if i:
        sys.stdin.readline()
        msg = sys.stdin.readline().strip()
        if len(msg) == 0:
            return 'timeout'
        elif msg == 'r':
            sd.play(samples * (1. / np.max(samples)), sr)
            msg = sys.stdin.readline().strip()
        return msg
    else:
        return 'timeout'



with open(args.csvFile, mode='r') as f:
    reader = csv.reader(f)
    labelDict = {rows[0]:rows[1] for rows in reader if rows[0].endswith('.wav')}
    playList = [key for key in list(labelDict.keys()) if labelDict[key] == args.label or args.label is None]

if args.csvFile1 is not None:
    with open(args.csvFile1, mode='r') as f:
        reader = csv.reader(f)
        labelDict1 = {rows[0]:rows[1] for rows in reader if rows[0].endswith('.wav')}
        playList1 = [key for key in list(labelDict1.keys()) if labelDict1[key] == args.label or args.label is None]
    playList = list(set(playList).symmetric_difference(set(playList1)))


playList = list(set(playList).difference(set(patchList)))
#import pdb; pdb.set_trace()
Length = len(playList)


outputFile = open(args.outputFile, "w+")
for i in range(len(playList)):
    fn = playList[i]
    print('processing {}, {}/{}'.format(args.root+'/'+fn, i+1, Length))
    samples, sr = sf.read(args.root+'/'+fn)
    sd.play(samples * (1. / np.max(samples)), sr)
    inputText = timeout_input(timeout=config['timeout'])
    humanLabel = args.label if inputText == 'timeout' else inputText
    print('{},{}'.format(fn, humanLabel), file=outputFile)
    outputFile.flush()
outputFile.close()

