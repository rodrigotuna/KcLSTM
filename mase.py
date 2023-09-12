import json
import numpy as np
import matplotlib.pyplot as plt
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
import pandas as pd

from dm_test import dm_test

with open("clstm_pred.json", 'r') as f:
  pred_gclstm = json.load(f)

with open("./lstm_val/lstm_pred.json") as f:
   pred_lstm = json.load(f)

dfc = pd.read_csv("score_M_0.6.csv")


mase_gclstm = []
mase_clstm = []
mase_lstm = []

for i in range(0,199):
   id = str(i+1)
   pred = np.array(dfc.iloc[i]["clstm_pred"][1:-1].split())
   pred = pred.astype(float)
   if i == 0:
      plt.plot(pred, label="CLSTM")
      plt.plot(pred_gclstm[id]["y_pred"], label="KCLSTM")
      plt.plot(pred_gclstm[id]["y_true"], label="TRUE")
      plt.legend()
      plt.savefig("DEBUG.pdf")
   mase_clstm.append(mean_absolute_scaled_error(y_train=np.array(pred_gclstm[id]["y_train"]), y_pred=pred, y_true=np.array(pred_gclstm[id]["y_true"])))

for i in range(1,200):
    id = str(i)
    mase_gclstm.append(mean_absolute_scaled_error(y_train=np.array(pred_gclstm[id]["y_train"]), y_pred=np.array(pred_gclstm[id]["y_pred"]), y_true=np.array(pred_gclstm[id]["y_true"])))


for i in range(0,199):
    mase_lstm.append(pred_lstm[i]["mase"])


print("LSTM - Median: ", np.median(mase_lstm), "Mean : ", np.mean(mase_lstm), "Standard : ", np.std(mase_lstm))
print("CLSTM - Median: ", np.median(mase_clstm), "Mean : ", np.mean(mase_clstm), "Standard : ", np.std(mase_clstm))
print("KCLSTM - Median: ", np.median(mase_gclstm), "Mean : ", np.mean(mase_gclstm), "Standard : ", np.std(mase_gclstm))
print("\n")

lstm = 0
gclstm = 0
ties = 0
for i in range(0,199):
   id = str(i+1)
   (DM,_) = dm_test(np.array(pred_gclstm[id]["y_true"]), np.array(pred_gclstm[id]["y_pred"]), np.array(pred_lstm[i]["y_pred"]))
   if DM < -1.96 or DM > 1.96:
      if DM < -1.96:
         gclstm += 1
      else:
         lstm += 1
   else:
      ties += 1
print("LSTM vs KCLTSM")
print("WINS LSTM:", lstm)
print("WINS KCLSTM", gclstm)
print("TIES", ties)
print("\n")


clstm = 0
gclstm = 0
ties = 0
for i in range(0,199):
   id = str(i+1)
   pred = np.array(dfc.iloc[i]["clstm_pred"][1:-1].split())
   pred = pred.astype(float)
   (DM,_) = dm_test(np.array(pred_gclstm[id]["y_true"]), np.array(pred_gclstm[id]["y_pred"]), pred)
   if DM < -1.96 or DM > 1.96:
      if DM < -1.96:
         gclstm += 1
      else:
         clstm += 1
   else:
      ties += 1
print("CLSTM vs KCLSTM")
print("WINS CLSTM:", clstm)
print("WINS KCLSTM", gclstm)
print("TIES", ties)

# matrix = []

# detect_params = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
# correct_params = [0.2, 0.3, 0.4, 0.5, 0.6]

# for detection_threshold in detect_params:
#     line = []
#     for correction_threshold in correct_params:
#         result = {}
#         with open(f"{detection_threshold}-{correction_threshold}/metric.json", 'r') as infile:
#             result = json.load(infile)
#         mase_list = []
#         for unique_id in range(1,51):
#             series = result[str(unique_id)]
#             mase = mean_absolute_scaled_error(y_true=np.array(series["y_true"]), y_pred=np.array(series["y_pred_g"]), y_train=np.array(series["y_train"]))
#             mase_list.append(mase)

#         diff_list = []
#         loss = 0
#         for i, _ in enumerate(lstm):
#             diff_list.append(lstm[i] - mase_list[i])
#             if(lstm[i] - mase_list[i] > 2):
#                 print(i)
#             if lstm[i] - mase_list[i] < 0:
#                 loss += 1
        
#         print(f"{detection_threshold}-{correction_threshold} loss = {loss}/50")

#         plt.clf()
#         plt.hist(diff_list, bins=200)
#         plt.savefig(f"{detection_threshold}-{correction_threshold}-hist-diff.pdf")

#         diff_list.sort()
#         plt.clf()
#         plt.plot(diff_list)
#         plt.plot(50*[0])
#         plt.savefig(f"{detection_threshold}-{correction_threshold}-plot-diff.pdf")
#         plt.clf()
#         plt.hist(mase_list, bins=200)
#         plt.savefig(f"{detection_threshold}-{correction_threshold}-hist.pdf")
#         line.append(np.median(mase_list))
#     matrix.append(line)

# ax = sns.heatmap(matrix, xticklabels=correct_params, yticklabels=detect_params, cmap=sns.cm.rocket_r)
# ax.set_ylabel("Detection Threshold")
# ax.set_xlabel("Correction Threshold")
# ax.set_title('Average MASE')
# plt.savefig(f"grid_median.pdf")
# plt.clf()
