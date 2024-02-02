import argparse
from train_teacher_forcing import *
#from train_with_sampling import *
from DataLoader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from helpers import *
from inference import *





def main(
    epoch,
    k,
    batch_size,
    frequency,
    training_length,
    forecast_window,
    train_csv,
    test_csv,
    path_to_save_model,
    path_to_save_loss, 
    path_to_save_predictions, 
    device
):

    clean_directory()
    
    root_dir = "/content/drive/Othercomputers/My Laptop/github-repositories/transformer-multi/Data/INTERACTION"
    #root_dir = "Data/INTERACTION/"
    
    train_dataset = SensorDataset(csv_name = train_csv, root_dir = root_dir, training_length = training_length, forecast_window = forecast_window, training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = SensorDataset(csv_name = test_csv, root_dir = root_dir, training_length = training_length, forecast_window = forecast_window, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #best_model = transformer(train_dataloader, epoch, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
    best_model = transformer(train_dataloader, epoch, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)

    inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--epoch", type=int, default=1)
    #parser.add_argument("--k", type=int, default=60)
    #parser.add_argument("--batch_size", type=int, default=1)
    #parser.add_argument("--frequency", type=int, default=100)
    #parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    #parser.add_argument("--path_to_save_loss",type=str,default="save_loss/")
    #parser.add_argument("--path_to_save_predictions",type=str,default="save_predictions/")
    #parser.add_argument("--device", type=str, default="cpu")
    #args = parser.parse_args()

    main(
        epoch=100,
        k = 60,
        batch_size=64,
        frequency=100,
        training_length = 19, # Use the complete sequence data available for training
        forecast_window = 12,
        train_csv="_train_MA.csv",
        test_csv="_train_MA.csv",
        path_to_save_model="save_model/",
        path_to_save_loss="save_loss/",
        path_to_save_predictions="save_predictions/",
        device="cuda",  
    )

