import torch
#@title Load Python libraries

# pip install numpy
import numpy as np

# pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# pip install alpha_vantage
from alpha_vantage.timeseries import TimeSeries

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from ESRNN.m4_data import *
import six
import sys
sys.modules['sklearn.externals.six'] = six
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import pmdarima as pm
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import mean_squared_error
import similaritymeasures
from IPython.display import clear_output
import time
import json

print("All libraries loaded")


# Configuration
np.random.seed(0)
torch.manual_seed(0)
config = {
    "data": {
        "window_size": 1,
        "train_split_size": 0.88,
    }, 
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature
        "num_lstm_layers": 1,
        "lstm_size": 12,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 1,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

#validation set no lstm
# Define the Gaussian kernel
def gaussian_kernel(x, x_, sigma):
    return np.exp(-(x - x_)**2/(2 * sigma**2))

def kernel_smooth(data, kernel, window):
    n = len(data)
    smoothed_data = np.array([np.zeros(12) for _ in range(n)])
    for i in range(n):
        numerator = np.zeros(12)
        denominator = 0
        for j in range(max(0,i-window),min(n,i+window)):
            if i == j: continue
            numerator += data[j]*kernel(data[i],data[j], 1)
            denominator += kernel(data[i],data[j], 1)
        smoothed_data[i] = numerator/denominator
    
    smoothed_data_cpy = np.array(smoothed_data)
    data_cpy = np.array(data)
    print(data_cpy.shape)
    print(smoothed_data_cpy.shape)
    fig = figure(figsize=(15, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_cpy.flatten()[1200:1260], label="Original States")
    plt.plot(smoothed_data_cpy.flatten()[1200:1260], label="Smoothed States")
    plt.legend()
    plt.savefig("hiddenstates_zoom.pdf")
    return smoothed_data

# Detection Component
def detect(LNG_FOR, model_pmd=None, all_input_=[], detection_threshold=-1, corrected=False, auto_arima=False):
      
    if model_pmd is not None:
        #Epoch 49
        print("Testing Correction at Cell states index: ", (len(all_input_)-1))
        
        SARIMAX_forecast = model_pmd[len(all_input_)-1:]
        # Creating an index from 2018-10-29 to Forecast Length, frequency indicates business day then putting it all together into a SARIMAX_forecast dataframe
        idx = pd.DatetimeIndex(pd.date_range('2018-10-29', periods=1, freq="B").tolist())

        SARIMAX_forecast = pd.DataFrame(list(zip(list(idx),list(SARIMAX_forecast))),
        columns=['Date','Forecast']).set_index('Date')

        #print("\tRoot Mean Squared Error - SARIMA:", rootmsq)
        # Generate random experimental data
        all_input_ =list([[0 for x in range((LNG_FOR))] for _ in range(3)]) + np.array(all_input_).tolist()
        
        exp_data = np.zeros((LNG_FOR, 2))
        exp_data[:, 0] = list(range(len(all_input_), LNG_FOR + len(all_input_))) 
        exp_data[:, 1] = SARIMAX_forecast['Forecast'][len(SARIMAX_forecast['Forecast'])-1:][0]

        # Generate random numerical data

        num_data = np.zeros((LNG_FOR, 2))
        num_data[:, 0] = list(range(len(all_input_), LNG_FOR + len(all_input_))) 
        num_data[:, 1] = all_input_[len(all_input_)-1:][0]

        #plt.figure(figsize=(7, 3))
        #plt.plot(exp_data[:, 0], exp_data[:, 1], label="SARIMA", color="red")
        #plt.plot(num_data[:, 0], num_data[:, 1], label="LSTM cell state", color="black")
        #plt.legend()
        #plt.show()

        # quantify the difference between the two curves using
        # Dynamic Time Warping distance
        dtw_, d = similaritymeasures.dtw(exp_data, num_data)
        ret_empty =[]
    
        return [dtw_]
    else: 
        pd_ar= kernel_smooth(all_input_, gaussian_kernel, 10)
        
        detect_error = []

        for i in range(len(pd_ar)):
            exp_data = np.zeros((LNG_FOR, 2))
            exp_data[:, 0] = list(range(0, LNG_FOR)) 
            exp_data[:, 1] = pd_ar[i]

            # Generate random numerical data
            num_data = np.zeros((LNG_FOR, 2))
            num_data[:, 0] = list(range(0, LNG_FOR))
            num_data[:, 1] = all_input_[i]
            dtw_, d = similaritymeasures.dtw(exp_data, num_data)

            if dtw_ > detection_threshold:
                detect_error = np.concatenate((detect_error, [i]))
            else:
                detect_error = np.concatenate((detect_error, [0])) 
        
        print(f"cLSTM will change {sum(detect_error != 0)} data points")
        return pd_ar, detect_error
    
# Data Preparation
def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides=(x.strides[0],x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    
    # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size
    # use the next day as label
    
    output = x[window_size:]
    return output

def prepare_data(normalized_data_close_price, config, plot=False):
    data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
    data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

    # split dataset
    print(data_y.shape[0])
    split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]


    return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
        

# LSTM
class LSTMModel(nn.Module):

        def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.linear_1 = nn.Linear(input_size, hidden_layer_size)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
            self.init_weights()
            self.historycorrectorlstm = []
            self.historycorrection = []
      
        def init_weights(self):
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight_ih' in name:
                    nn.init.kaiming_normal_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
        def forward(self, x):
            batchsize = x.shape[0]
            # layer 1
            x = self.linear_1(x)
            x = self.relu(x)

            # LSTM layer
            lstm_out, (h_n, c_n) = self.lstm(x)     

            # reshape output from hidden cell into [batch, features] for `linear_2`
            x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 

            # layer 2
            x = self.dropout(x)
            predictions = self.linear_2(x)
            return predictions[:,-1]

# Corrector LSTM
class cLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)

        self.init_weights()
        self.historycorrectorlstm = []
        self.historycorrection = []
        self.historycorrectorlstm_archive = []
        
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    
    def forward(self, x, model_sarima=0, epoch_ready=False, pre_epoch=False, corrected=False, detection_threshold=0.8, auto_arima=False, run_detection=False ):
        batchsize = x.shape[0]
        # layer 1
        value_best = x[0][0][0]
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        #Try cell state (c_n)
        return_model = False
        all_input_ = []
        #Epoch 48
        if pre_epoch:
            self.historycorrectorlstm.append(lstm_out.detach().numpy()[0][0])
            all_input_ = self.historycorrectorlstm

        if run_detection == True:     
            
            pd_ar, detect_error = detect(12, 
                                            model_pmd =  model_sarima, 
                                            all_input_= all_input_, 
                                            detection_threshold= detection_threshold, 
                                            corrected=True, 
                                            auto_arima=auto_arima)
            
            self.historycorrectorlstm_archive = self.historycorrectorlstm
            self.historycorrectorlstm = []           
        #Epoch 49
        if epoch_ready:
            print("Len of all_input:" , len(all_input_))
            if corrected:
                self.historycorrectorlstm = self.historycorrectorlstm[:-1] 
                self.historycorrectorlstm.append(lstm_out.detach().numpy()[0][0])
                all_input_ = self.historycorrectorlstm
            else:
                self.historycorrectorlstm.append(lstm_out.detach().numpy()[0][0])
                all_input_ = self.historycorrectorlstm

            if corrected == False:
                self.historycorrection = []

            print("Length of cell states: ", len( all_input_ ))
            if corrected:
                print("Corrected")
                results_correction = detect(12, 
                                            model_pmd =  model_sarima, 
                                            all_input_= all_input_, 
                                            detection_threshold= detection_threshold, 
                                            corrected=True, 
                                            auto_arima=auto_arima)
            else:
                print("No correction")                    
                results_correction = detect(12, 
                                            model_pmd =  model_sarima, 
                                            all_input_= all_input_, 
                                            detection_threshold= detection_threshold, 
                                            corrected=False, 
                                            auto_arima=auto_arima)

            if (results_correction is not None):                    
                self.historycorrection.append(results_correction)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)

        if run_detection:
            return predictions[:,-1], all_input_, self.historycorrectorlstm_archive, self.historycorrection, pd_ar, detect_error
        else:
            return predictions[:,-1], all_input_, self.historycorrection
            
# detection_threshold = 1.0
# correction_threshold = 0.2
use_auto_arima = False # find orders of ARIMA using auto.arima


def run_epoch(dataloader, LSTM = False,  model_sarima=None, is_training=False, epoch_ready=False , pre_epoch=False, pre_pre_epoch=False, detect_error=  np.array([]), auto_arima = use_auto_arima):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()
        
    original_signal = np.array([])
    corrected_signal = np.array([])
    y_corrected_signal = np.array([])    
    return_model = False
    historycorrectorlstm_archive = []
    
    for idx, (x, y) in enumerate(dataloader):
        
        if is_training:
            optimizer.zero_grad()
            batchsize = x.shape[0]
            #Epoch 48
            if pre_epoch and idx < (len(dataloader)-1):
                #print("Pre epoch - ID: ", idx)    
                torch.save(model.state_dict(), PATH) 
                x = x.to(config["training"]["device"])
                y = y.to(config["training"]["device"])

                #historycorrectorlstm - hiddenstates concatenated, historycorrection always empty
                out, historycorrectorlstm, historycorrection = model(x, model_sarima, pre_epoch=True , detection_threshold = detection_threshold)
                
            elif pre_epoch and idx == (len(dataloader)-1):                
                
                #print("Pre epoch - ID: ", idx)  
                torch.save(model.state_dict(), PATH)                
                x = x.to(config["training"]["device"])
                y = y.to(config["training"]["device"])
                
                out, historycorrectorlstm, historycorrectorlstm_archive, historycorrection, model_sarima, detect_error = model(x, model_sarima, pre_epoch=True, run_detection=True, auto_arima = auto_arima, detection_threshold = detection_threshold)
            #Epoch 49
            elif epoch_ready:
                
                #print("Epoch ready - ID: ", idx)
                original_signal = np.concatenate((original_signal, x[0][0]))
                #print("Raw time series value: ", x[0][0][0])      
                
                torch.save(model.state_dict(), PATH) 
                
                x = x.to(config["training"]["device"])
                y = y.to(config["training"]["device"])
                
                out, historycorrectorlstm, historycorrection = model(x, model_sarima, epoch_ready, detection_threshold = detection_threshold)
                initial_distance = historycorrection[-1][0]
                initial_value = x[0][0][0]
                #print("Initial value:", initial_value)
                
                past_distances = []
                past_signs = []
                past_distances.append(initial_distance)               
                
                idsarima = 0                          
                past_signs.append(-1)
                verified = 0       
                
                breaker = False
                update_value = 0.1
                
                if idx < 3:
                    corrected_signal = np.concatenate((corrected_signal, x[0][0]))
                    y_corrected_signal = np.concatenate((y_corrected_signal, y))
                
                if idx in detect_error and idx >= 3:
                    while historycorrection[-1][0] > detection_threshold:
                        if breaker:
                            break
                        while historycorrection[-1][0] > correction_threshold:
                            
                            corrected__signal = np.concatenate((corrected_signal, x[0][0]))
                            #plt.plot(corrected__signal, label="Corrected")
                            #plt.plot(original_signal, label="Original")
                            #plt.legend()
                            #plt.show()
                            
                            if idsarima == 0:

                                model.load_state_dict(torch.load(PATH))
                                x = x.to(config["training"]["device"])
                                y = y.to(config["training"]["device"])
                                # Recompute Sarima Orders if it is first time detection
                                out, historycorrectorlstm, historycorrection = model(x, model_sarima, epoch_ready, corrected=True, detection_threshold = detection_threshold)
                                idsarima += 1
                                breaker = False
                                
                            elif idsarima == 1:
                                
                                #print("sign: ", sign)
                                #print(update_value)
                                x[0][0][0] = x[0][0][0] + sign * update_value                      
                                
                                model.load_state_dict(torch.load(PATH))
                                x = x.to(config["training"]["device"])
                                y = y.to(config["training"]["device"])
                
                                out, historycorrectorlstm, historycorrection = model(x, model_sarima, epoch_ready, corrected=True, detection_threshold = detection_threshold)
                                breaker = False
                                
                            else:
                                
                                x[0][0][0] = initial_value
                                #print("Initial value:", x[0][0][0])
                                model.load_state_dict(torch.load(PATH))
                                x = x.to(config["training"]["device"])
                                y = y.to(config["training"]["device"])
                                out, historycorrectorlstm, historycorrection = model(x, model_sarima, epoch_ready, corrected=True, detection_threshold = detection_threshold)
                                breaker = True
                                break
                                
                            #print("Previous distance:", past_distances[-1])
                            # No exceed limits & verify distance                            
                            if historycorrection[-1] is not None:
                                if historycorrection[-1][0] > past_distances[-1]:
                                    sign = past_signs[-1] * -1
                                else:
                                    sign = past_signs[-1] * 1

                                if (past_distances.count(historycorrection[-1][0]) > 1):
                                    update_value = 0.01
                                    #print("Reduce value of change")
                                if (past_distances.count(historycorrection[-1][0]) > 2):
                                    idsarima = 2
                                    #print("Early Stop Excuted")
                                    
                                past_signs.append(sign)
                                past_distances.append(historycorrection[-1][0])
                                
                                #print("Data id:",idx," distance: ", historycorrection[-1][0])   
                                #print("New value: ", x[0][0][0])
                            
                            else:
                                corrected = True
                                break
                if idx >= 3:
                    corrected_signal = np.concatenate((corrected_signal, x[0][0]))
                    y_corrected_signal = np.concatenate((y_corrected_signal, y)) 
            else:  
                x = x.to(config["training"]["device"])
                y = y.to(config["training"]["device"])
                if LSTM:
                    out = model(x) 
                else:
                    out, historycorrectorlstm, historycorrection = model(x)
        
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()
        
        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]
    
    if pre_pre_epoch:
        return epoch_loss, lr, model_sarima
    elif pre_epoch:
        return epoch_loss, lr, model_sarima, detect_error, historycorrectorlstm_archive
    elif epoch_ready and is_training:
        return epoch_loss, lr, historycorrectorlstm, historycorrection, original_signal, corrected_signal, y_corrected_signal, detect_error
    else:
        return epoch_loss, lr 
    
X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(
dataset_name="Monthly", 
directory= "../error_gen/data", 
num_obs=200)
time_list = []

metric = {}

with open("gclstm_params.json", "r") as infile:
    params = json.load(infile)

detection_threshold = 0.6
correction_threshold = 0.5

for unique_id in range(1,201):
    metric[unique_id] = {}
    clear_output(wait=True)
    # read the csv file
    # print the location and filename
    start_time = time.time()
    print("Working on series :", unique_id)
    data_close_price = y_train_df.loc[y_train_df.unique_id=="M"+str(unique_id), "y"][-500:].values
    
    # normalize
    split_index = int(len(data_close_price)*config["data"]["train_split_size"])
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price) 
    normalized_data_close_price_extra = normalized_data_close_price[split_index:]
    #normalized_data_close_price = normalized_data_close_price[:-round(len(normalized_data_close_price)*0.12)]

    split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(normalized_data_close_price, config, plot=config["plots"]["show_plots"])
    split_index_cpy = split_index
    data_y_val_cpy = data_y_val
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    #cLSTM
    model = cLSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
    model = model.to(config["training"]["device"])

    PATH = "statedictmode" 
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=params[unique_id-1]["batch_size"], shuffle=False)
    train_dataloader_cpy = train_dataloader
    val_dataloader = DataLoader(dataset_val, batch_size=params[unique_id-1]["batch_size"], shuffle=False)

    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params[unique_id-1]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)
    detect_error = np.array([])
    # begin training
    for epoch in range(config["training"]["num_epoch"]):
        if epoch == 48:
            loss_train, lr_train, model_sarima, detect_error, historycorrectorlstm_archive = run_epoch(train_dataloader,  model_sarima= None, is_training=True, pre_epoch=True)
            scheduler.step()
        elif epoch == 49: #round(config["training"]["num_epoch"]/3):
            loss_train, lr_train, historycorrectorlstm, historycorrection, original_signal, corrected_signal, y_corrected_signal, detect_error = run_epoch(train_dataloader, model_sarima= model_sarima, is_training=True, epoch_ready=True, detect_error=detect_error)
            scheduler.step()
        elif epoch < 48 :
            loss_train, lr_train = run_epoch(train_dataloader,  model_sarima= None, is_training=True, epoch_ready=False)
            scheduler.step() 
        else:
            corrected_timeseries = np.append(corrected_signal, y_corrected_signal[-1])
            corrected_timeseries = np.append(corrected_timeseries, normalized_data_close_price_extra)            

            split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = prepare_data(corrected_timeseries, config, plot=config["plots"]["show_plots"])
            # print("GETREADY TO NEW SPLIT INDEX")
            # print(split_index)
            dataset_train = TimeSeriesDataset(data_x_train, data_y_train)            
            train_dataloader = DataLoader(dataset_train, batch_size=params[unique_id-1]["batch_size"], shuffle=False)

            loss_train, lr_train = run_epoch(train_dataloader, None, is_training=True, epoch_ready=False)
            scheduler.step()

        print('Epoch[{}/{}] | loss train:{:.6f} | lr:{:.6f}'
                    .format(epoch+1, config["training"]["num_epoch"], loss_train, lr_train))

    time_list.append([unique_id, (time.time() - start_time)])

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    torch.manual_seed(0)    

    model.eval()
    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])
    for idx, (x, y) in enumerate(train_dataloader_cpy):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out[0].cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    print(predicted_train)
    print(len(predicted_train))
    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out[0].cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    print(len(predicted_val))
    

    data_date = np.array(y_train_df.loc[y_train_df.unique_id=="M"+str(unique_id), 'ds'][-500:].values)

    num_data_points = len(data_date)

    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[config["data"]["window_size"]:split_index_cpy+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index_cpy+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    # plots

    fig = figure(figsize=(15, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted (train)", color=config["plots"]["color_pred_train"])
    plt.plot(data_date[1:], data_close_price[:-1], label="Original Data", color=config["plots"]["color_actual"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted (test)", color="green")
    plt.xticks([])
    plt.legend()
    plt.savefig(f"plots/normal-{unique_id}.pdf")

    # prepare data for plotting, zoom in validation
    to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val_cpy)
    to_plot_predicted_val = scaler.inverse_transform(predicted_val)
    print(len(to_plot_data_y_val_subset))
    print(len(to_plot_predicted_val))
    to_plot_data_date = data_date[split_index_cpy+config["data"]["window_size"]:]
    # plots
    fig = figure(figsize=(15, 3), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted (validation)", color=config["plots"]["color_pred_val"])
    plt.plot(to_plot_data_date[1:], to_plot_data_y_val_subset[:-1], label="Actual Timeseries", color=config["plots"]["color_actual"])
    plt.title("Zoom in to examine predicted timeseries on validation data portion")
    plt.grid(b=None, which='major', axis='y', linestyle='--')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.savefig(f"plots/scale-{unique_id}.pdf")

    rmse = np.sqrt(mean_squared_error(to_plot_data_y_val_subset[:len(to_plot_data_y_val_subset)-1], to_plot_predicted_val[1:]))
    metric[unique_id]["y_train"] = data_close_price[:split_index].tolist()
    metric[unique_id]["y_true"] = to_plot_data_y_val_subset[:len(to_plot_data_y_val_subset)-1].tolist()
    metric[unique_id]["y_pred"] = to_plot_predicted_val[1:].tolist()
    print("cLSTM RMSE -- ", rmse)


with open(f"clstm_pred.json", 'w') as outfile:
     json.dump(metric, outfile, indent=2)

