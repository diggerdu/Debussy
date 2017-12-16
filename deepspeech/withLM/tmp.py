import csv
import pickle
with open("testlabel.csv", mode='r') as f:
    reader = csv.reader(f)
    labelDict = {rows[0]:rows[1] for rows in reader}

with open("testlabel.pkl", mode='wb') as f:
    pickle.dump(labelDict, f)
    f.flush()
    f.close()

