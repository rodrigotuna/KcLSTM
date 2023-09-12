import json
import numpy as np
import matplotlib.pyplot as plt

with open("gclstm_validation.json", 'r') as f:
  tuning = json.load(f)

best = []

for i in range(1,201):
    series_dict = tuning[str(i)]
    
    best_config = min(series_dict, key=series_dict.get)
    params = best_config.split('-')
    best.append({"batch_size" : int(params[0]), "learning_rate" : float(params[1])} )

with open(f"gclstm_params.json", 'w') as outfile:
        json.dump(best, outfile, indent=2)
