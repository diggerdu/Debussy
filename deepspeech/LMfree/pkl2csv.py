import pickle
import csv

table = ['yes,' 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
with open("testlabel.pkl", "rb") as f:
    labelDict = pickle.load(f)
    f.close()

with open("testlabel.csv", mode="w") as f:
    writer = csv.writer(f)
    values = list(labelDict.values())
    for i in range(len(values)):
        if values[i] == '':
            values[i] = 'silence'
        elif values[i] not in table:
                values[i] = 'unknown'
    #import pdb; pdb.set_trace()
    writer.writerows(zip(['fname'] + list(labelDict.keys()), ['label'] + values))
    f.flush()
    f.close()
