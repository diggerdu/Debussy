import pickle
from collections import Counter, defaultdict


PATH = 'withLM'
d = defaultdict(list)
meshDict = dict()
labelDict = pickle.load(open(PATH+"/trainlabel.pkl", 'rb'))

for k, v in labelDict.items():
    d[v].append(k.split('/')[0])

for k, v in d.items():
    meshDict.update({k : Counter(v).most_common(1)[0][0]})

print(meshDict)

with open(PATH+"/meshDict.pkl", "wb") as f:
    pickle.dump(meshDict, f)
    f.close()





