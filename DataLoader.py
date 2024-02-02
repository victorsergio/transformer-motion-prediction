import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from joblib import load
from joblib import dump
from numpy.lib.stride_tricks import sliding_window_view

class inDDataset(Dataset):

    def __init__(self, dataset_file, training_length, input_length, forecast_window, training=True):
        
        # load pandas dataframe
        self.df = load(dataset_file)

        #csv_file = os.path.join(root_dir, csv_name)
        #.df = pd.read_csv(csv_file)
        #self.root_dir = root_dir
        #self.transform = StandardScaler()
       
        self.training_length = training_length
        self.forecast_window = forecast_window
        self.input_length = input_length

        print(list(self.df.columns))

         # Group all frames by track
        self.data = self.df.groupby(['globalSequenceId'])

        #self.df = self.df.drop(columns=['case_id','object_id','frame_ix','vx','vy'])
        #data = self.df.to_numpy()
        # create dummy dataset   (100,2)
        #x1 = np.arange(0, 40)
        #x2 = np.arange(100,140)
        #x3 = np.arange(200, 240)
        #x4 = np.arange(300,340)
        #x5 = np.arange(400,440)  
        #data = np.vstack((x1,x2,x3)) 
        #data = np.transpose(data, axes=[1, 0])

      
        # scaling
        #data = self.transform.fit_transform(data[:,:])


        # training data
       
        #data_training = sliding_window_view(data, self.training_length, 0)[::self.training_length,:,:]
#        data_training = sliding_window_view(data, training_length, 0)

        #data_training = np.transpose(data_training, axes=[0, 2, 1])

        #self.Y = data_training[:,1:,:] #0:3 because we need to re-inject prediction at inference # this is a copy, for each sequence i on X corresponds sequence i+1 from X on Y
        #self.X = data_training[:,:-1,:]

        # test data
        #if(training == False):
        #    data_inference = sliding_window_view(data, 20, 0)[::20,:,:]
        #    data_inference = np.transpose(data_inference, axes=[0, 2, 1])
        #    self.Y = data_inference[:,8:,0:3]
        #    self.X = data_inference[:,0:8,:]


        #print("data X shape:", self.X.shape)
        #print("data Y shape:", self.Y.shape)
 #       print(self.X[0])
 #       print(self.Y[0])
 #       print(self.X[1])
 #       print(self.Y[1])

       # dump(self.transform, 'scalar_item.joblib')


    def __len__(self):
        # return number of tracks in the dataset
        return len(self.data.ngroups)

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        
        # Sensors are indexed from 1
        #idx = idx+1

        # np.random.seed(0)
        
        
        # Select a specific frames for a track
        agent = self.data.get_group(idx)
        
        print(agent)

        #data_training = np.transpose(data_training, axes=[0, 2, 1])
        
        
        _input = None
        target = None
           #_input = torch.tensor(self.X[idx])
            #target = torch.tensor(self.Y[idx])


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
    
train_dataset = inDDataset(dataset_file= "./data/inD-dataset-v1.0/data/test.joblib", 
                           training_length = 20, input_length=8, forecast_window = 12, training=True)

train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=False)


next(train_dataloader)