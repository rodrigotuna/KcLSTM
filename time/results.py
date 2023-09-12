import json
import numpy as np

lstm = []
clstm = []
gclstm = []


with open("time_lstm.json", "r") as f:
    lstm = json.load(f)

with open("time_clstm.json", "r") as f:
    clstm = json.load(f)

with open("time_gclstm.json", "r") as f:
    gclstm = json.load(f)


print("LSTM: ", np.mean(lstm))
print("CLSTM: ", np.mean(clstm))
print("GCLSTM: ", np.mean(gclstm))