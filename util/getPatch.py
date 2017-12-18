import csv
import os


def getPatch(Path):
    patchList = list()
    for fn in os.listdir(Path):
        try:
            assert fn.endswith("csv")
        except:
            continue
        with open(Path+'/'+fn) as f:
            reader = csv.reader(f)
            patchList.extend([row[0] for row in reader])
    return patchList

def getLabelDict(Path):
    labelDict = dict()
    for fn in os.listdir(Path):
        try:
            assert fn.endswith("csv")
        except:
            continue
        with open(Path+'/'+fn) as f:
            reader = csv.reader(f)
            labelDict.update({row[0]:row[1] for row in reader if row[1] != '#'})
    return labelDict







