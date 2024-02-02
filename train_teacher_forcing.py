from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time # debugging
from plot import *
from helpers import *
from joblib import load
from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
import transformers as hftransformers

import wandb
# start a new experiment
wandb.login(key='d3abe6de5c06c1c57934b06af9db1fc364be0487')
# Project name
wandb_project_name = "transformer-prediction"
wandb.init(project=wandb_project_name, name="trans")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def transformer(dataloader, EPOCH, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):

    device = torch.device(device)

    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    #scheduler = torch.optim.lr_scheduler.CosineScheduler(20, warmup_steps=10, base_lr=0.3, final_lr=0.01)
    scheduler = hftransformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=(len(dataloader) * 10), num_training_steps=len(dataloader) * EPOCH)

    
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):

        train_loss = 0
        val_loss = 0

        ## TRAIN -- TEACHER FORCING
        model.train()
        for _input, target in dataloader: # for each data set 
        
            optimizer.zero_grad()

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]

            #src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # torch.Size([24, 1, 7])
            src = _input.permute(1,0,2).double().to(device)
      #      print("shape src: ", src.size())
            #target = _input.permute(1,0,2).double().to(device)[1:,:,:] # src shifted by 1.
            target = target.permute(1,0,2).double().to(device)
            #print("shape target: ", target.size(),target)

            prediction = model(src, device) # torch.Size([24, 1, 7])
            #print("shape prediction: ", prediction.size(), prediction)

            loss = criterion(prediction, target[:,:,0:3])
            loss.backward()
            optimizer.step()


            # scheduler.step(loss.detach().item())
            train_loss += loss.detach().item()

        if train_loss < min_train_loss:
        #if epoch == 99:
            torch.save(model.state_dict(), path_to_save_model + f"best_train_.pth")
            torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_.pth")
            min_train_loss = train_loss
            best_model = f"best_train_.pth"


        if epoch % 1 == 0: # Plot 1-Step Predictions
            train_loss /= len(dataloader)

            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            wandb.log({"epoch": epoch, "loss": train_loss})
            #scaler = load('scalar_item.joblib')
           # src_humidity = scaler.inverse_transform(src[:,:,0:2].cpu()) #torch.Size([35, 1, 7])
           # target_humidity = scaler.inverse_transform(target[:,:,0:2].cpu()) #torch.Size([35, 1, 7])
           # prediction_humidity = scaler.inverse_transform(prediction[:,:,0:2].detach().cpu().numpy()) #torch.Size([35, 1, 7])
            #plot_training(epoch, path_to_save_predictions, src_humidity, prediction_humidity, sensor_number, index_in, index_tar)

        #train_loss /= len(dataloader)
        #log_loss(train_loss, path_to_save_loss, train=True)

        #wandb.log({"epoch": epoch, "loss": train_loss})

        scheduler.step()
        
    plot_loss(path_to_save_loss, train=True)
    return best_model