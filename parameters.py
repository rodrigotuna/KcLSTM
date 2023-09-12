import json
import numpy as np
import matplotlib.pyplot as plt

with open("lstm_validation.json", 'r') as f:
  tuning = json.load(f)


mase = []
best = []

for i in range(1,201):
    series_dict = {}
    for batch_size in [1,2,4,8]:
        for learning_rate in [0.0001,0.001,0.01,0.1]:
            config_str = f"{batch_size}-{learning_rate}"
            series_dict[config_str] = tuning[str(i)][config_str]["validation"]
    
    best_config = min(series_dict, key=series_dict.get)
    params = best_config.split('-')
    best.append({"batch_size" : int(params[0]), "learning_rate" : float(params[1])} )
    mase.append(tuning[str(i)][best_config]["test"]["mase"])

with open(f"lstm_params.json", 'w') as outfile:
        json.dump(best, outfile, indent=2)


plt.hist(mase)
plt.savefig("hyper-params.pdf")
print(np.mean(mase))
print(np.median(mase))

###Train the model with the correct parameters
###After selection
