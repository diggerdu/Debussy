import csv
import os
from collections import defaultdict
from shutil import copyfile


fileDict = defaultdict(list)
labelDict = dict()

PATH = './'
testPath = '/home/diggerdu/dataset/tfsrc/test/audio'
trainPath = '/home/diggerdu/dataset/tfsrc/extendTrain'
for fn in os.listdir(PATH):
    if not fn.endswith('csv'):
        continue
    with open(fn, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            fn = row[0]
            label = row[1]
            if label == '\'':
                label = 'unknown'
            if label == '\\':
                label = 'silence'
            if label != '#':
                labelDict.update({fn:label})
                fileDict[label].append(fn)

def getPatchDict():
    return labelDict

def getFileDict():
    return fileDict

if __name__ == '__main__':
    fileDict = getFileDict()
    for k, v in fileDict.items():
        try:
            assert len(k) > 0
        except:
            print(v)
            continue
        path = trainPath + '/' + k
        if not os.path.isdir(path):
            os.mkdir(path)
        for fn in v:
            oriPath = testPath + '/' + fn
            dstPath = path + '/' + fn
            copyfile(oriPath, dstPath)


