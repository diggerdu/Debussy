import pickle
import csv

#PATH = 'LMfree'
PATH = 'withLM'

TABLE = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
with open(PATH + "/testlabel.pkl", "rb") as f:
    labelDict = pickle.load(f)

with open(PATH + "/meshDict.pkl", "rb") as f:
    meshDict = pickle.load(f)


for k, v in labelDict.items():
    if len(v.strip()) == 0:
        labelDict[k] = 'silence'
    else:
        labelDict[k] = 'unknown' if not (v in meshDict.keys() and meshDict[v] in TABLE) else meshDict[v]
    assert labelDict[k] in TABLE

with open(PATH + '/meshTestLabel.csv', 'w') as csvFile:
    print('fname,label', file=csvFile)
    for k, v in labelDict.items():
        print('{},{}'.format(k,v), file=csvFile)
    csvFile.close()

