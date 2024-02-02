import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic
from numpy.lib.stride_tricks import sliding_window_view

class SensorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window, training=True):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

        # create dummy dataset   (100,2)
        x1 = np.arange(0, 40)
        x2 = np.arange(100,140)
        x3 = np.arange(200, 240)
        x4 = np.arange(300,340)
        x5 = np.arange(400,440)
           
        data = np.vstack((x1,x2,x3)) 
        data = np.transpose(data, axes=[1, 0])

      #  print(data.shape)
        # scaling
        data = self.transform.fit_transform(data[:,0:3])


        # training data
        data_training = sliding_window_view(data, training_length, 0)
        data_training = np.transpose(data_training, axes=[0, 2, 1])

        self.Y = data_training[1:,:,0:3] #0:2 because only predict x and y # this is a copy, for each sequence i on X corresponds sequence i+1 from X on Y
        self.X = data_training[:-1,:,:]

     #   print("training:",self.X.shape)

        # test data
        if(training == False):
            data_inference = sliding_window_view(data, 20, 0)[::20,:,:]
            data_inference = np.transpose(data_inference, axes=[0, 2, 1])
            self.Y = data_inference[:,8:,0:3]
            self.X = data_inference[:,0:8,:]

        #    print("test:",self.X.shape)




        #print(self.X.shape)

        dump(self.transform, 'scalar_item.joblib')


    def __len__(self):
        # return number of sensors
        return len(self.X)

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        
        # Sensors are indexed from 1
        #idx = idx+1

        # np.random.seed(0)
        
        _input = torch.tensor(self.X[idx])
        target = torch.tensor(self.Y[idx])


        #start = np.random.randint(0, len(self.df[self.df["reindexed_id"]==idx]) - self.T - self.S) 
        #sensor_number = str(self.df[self.df["reindexed_id"]==idx][["sensor_id"]][start:start+1].values.item())
        #index_in = torch.tensor([i for i in range(start, start+self.T)])
        #index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        #_input = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour"]][start : start + self.T].values)
        #target = torch.tensor(self.df[self.df["reindexed_id"]==idx][["humidity", "sin_hour"]][start + self.T : start + self.T + self.S].values)

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        #scaler = self.transform

       # scaler.fit(_input[:,0:1].unsqueeze(-1))
        #scaler.fit(_input[:,0:2])

        #_input[:,0:2] = torch.tensor(scaler.transform(_input[:,0:2]))
        #target[:,0:2] = torch.tensor(scaler.transform(target[:,0:2]))

        # save the scalar to be used later when inverse translating the data for plotting.
        #dump(scaler, 'scalar_item.joblib')

        #return index_in, index_tar, _input, target, sensor_number
        return _input, target