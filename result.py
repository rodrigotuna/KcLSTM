import json
import numpy as np
import matplotlib.pyplot as plt

with open("lstm_pred.json", 'r') as f:
  pred = json.load(f)


mase = []
for i in range(0,200):
    mase.append(pred[i]["mase"])




plt.hist(mase, bins=40)
plt.savefig("mase-pred.pdf")
print(np.mean(mase))
print(np.median(mase))
