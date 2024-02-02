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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):

    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(path_to_save_model+best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad():

        model.eval()
        for plot in range(1):

            for _input, target, in dataloader:
                
                # starting from 1 so that src matches with target, but has same length as when training
                src = _input.permute(1,0,2).double().to(device) # 47, 1, 7: t1 -- t47
                target = target.permute(1,0,2).double().to(device) # t48 - t59

                next_input_model = src
                all_predictions = []

                for i in range(forecast_window):
                    
                    prediction = model(next_input_model, device) # 47,1,1: t2' - t48'
                   # print("prediction shape:", prediction.size())

                    if all_predictions == []:
                        all_predictions = prediction[-1,:,:].unsqueeze(0) # 47,1,1: t2' - t48'
                    else:
                       # print("all",all_predictions.size())
                       # print("squeeze",prediction[-1,:,:].unsqueeze(0).size())
                        all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'
                    #    print("prediction shape cat:", prediction[-1,:,:].unsqueeze(0).size())

                    #pos_encoding_old_vals = src[i+1:, :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                    #pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                    #pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
                    
                    #next_input_model = src
                    next_input_model = torch.cat((src[:, :, :], all_predictions[:,:,:]))
                    #next_input_model = next_input_model[i+1:,:,:]
                    #next_input_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) #t2 -- t47, t48'
                    #next_input_model = torch.cat((next_input_model, pos_encodings), dim = 2) # 47, 1, 7 input for next round

                #true = torch.cat((src[1:,:,0],target[:-1,:,0]))
                true = target
        #        print("all shape", all_predictions.size())
         #       print("target true", true.size())
               # print("true: ", true)
                #print("target: ", target)

                # prin samples
 #               scaler = load('scalar_item.joblib')
                
                #prediction_val = scaler.inverse_transform(all_predictions[:,0,0:3].detach().cpu().numpy())
                #true_val = scaler.inverse_transform(true[:,0,0:3].detach().cpu().numpy())
                #print("true>",true_val)
                #print("pred>",prediction_val)

                #prediction_val = scaler.inverse_transform(all_predictions[:,1,0:3].detach().cpu().numpy())
                #true_val = scaler.inverse_transform(true[:,1,0:3].detach().cpu().numpy())
                #print("true>",true_val)
                #print("pred>",prediction_val)
                



                loss = criterion(true[:,:,0:2], all_predictions[:,:,0:2]) # only x,y
                val_loss += loss
            
            print("dataloader len:", len(dataloader))
            val_loss = val_loss/len(dataloader)
            #scaler = load('scalar_item.joblib')
            #src_humidity = scaler.inverse_transform(src[:,:,0].cpu())
            #target_humidity = scaler.inverse_transform(target[:,:,0].cpu())
            #prediction_humidity = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())
            #plot_prediction(plot, path_to_save_predictions, src_humidity, target_humidity, prediction_humidity, sensor_number, index_in, index_tar)

        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")