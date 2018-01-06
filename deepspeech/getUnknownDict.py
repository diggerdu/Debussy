import pickle
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
import table
import getPatch


patchDict = getPatch.getLabelDict('../labeling')

with open("withLM/testlabel.pkl", 'rb') as f:
    withLMDict = pickle.load(f)
    f.close()

with open("LMfree/testlabel.pkl", 'rb') as f:
    LMfreeDict = pickle.load(f)
    f.close()

unknownDict = dict()
for fn, label in withLMDict.items():
    if label == LMfreeDict[fn] and label in table.unknownTable:
        if fn not in patchDict.keys():
            unknownDict.update({fn:label})


with open("unverifyUnknown.csv", "w") as f:
    for fn, label in unknownDict.items():
        print('{},{}'.format(fn, label), file=f)

    f.close()

