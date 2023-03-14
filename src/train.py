"""
This file contains the training loop of the residual models
Author: Danush Kumar Venkatesh
"""
import os
from time import sleep
from glob import glob
import random
from tqdm import tqdm
import copy
import ntpath

import numpy as np
from imageio import imread
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.models import resnet18
import click

from models import *
from data_load import *

@click.command()
@click.option('-nr', '--n_rotation', type=int, 
              help="no. of rotations")
@click.option('-nf', '--n_filters', type=int, 
              help="no. of filters")
@click.option('-rf', '--flip', type=bool, 
              help="the flag to turn on dihedral group")
@click.option('-nc', '--num_class', type=int, default=11, 
              help="the number of classes in the dataset")
@click.option('-sp', '--save_path', type=str, 
              help="the path to save the file")
@click.option('-mt', '--model_type', type=str, default="res",
              help="the type of model to run")
@click.option('-ep', '--epoch', type=int, default=10, 
              help="the option for model")
@click.option('-ap', '--ace_path', type=str, 
              help="the path to the Acevedo dataset")
@click.option('-mp', '--mat_path', type=str, 
              help="the path to the Matek dataset")
@click.option('-tf', '--test_frac', type=float, default=0.2,
              help="the percentage of data for the test set")
@click.option('-vf', '--val_frac', type=float, default=0.125,
              help="the percentage of data for the validation set")
@click.option('-bc', '--batch_size', type=int, default=32,
              help="the batch size of the dataloader")
@click.option('-lr', '--learn_rate', type=float, default=1e-5,
              help="the learning rate of the training function")
@click.option('-ss', '--split_size', type=float, default=1.0,
              help="the fraction of training data to use")
def main(n_rotation, n_filters, flip, num_class, save_path, model_type, epoch, 
         ace_path, mat_path, test_frac, val_frac, batch_size, learn_rate,
        split_size):

    data_path = {
            "Ace_20": ace_path, # Acevedo_20 Dataset
            "Mat_19": mat_path, # Matek_19 Dataset
        }
    
    dataframe, train_index, test_index, val_index = get_indexes(test_frac, val_frac)

    #dataset-creation

    train_dataset = DatasetGenerator(dataframe.loc[train_index,:], 
                                     reshape_size=resize, 
                                     dataset = source_domains,
                                     label_map=label_map_all, 
                                     transform = train_transform,
                                     )
    val_dataset = DatasetGenerator(dataframe.loc[val_index,:], 
                                     reshape_size=resize, 
                                     dataset = source_domains,
                                     label_map=label_map_all, 
                                     transform = val_transform,
                                     )

    test_dataset = DatasetGenerator(dataframe.loc[test_index,:], 
                                     reshape_size=resize, 
                                     dataset = source_domains,
                                     label_map=label_map_all, 
                                     transform = test_transform,
                                     )

    train_data_len = len(train_dataset)
    train_split_size = int(split_size*train_data_len)
    remain_split_size = abs(train_data_len - train_split_size)
    train_dataset,_ = torch.utils.data.random_split(train_dataset, 
                                                   lengths=(train_split_size, remain_split_size))
    print(f"length of tr:{train_data_len}, but using {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    epochs=epoch # max number of epochs
    lr=learn_rate # learning rate
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = num_class
    if model_type=='res':
        model = EqRes(n_rot=n_rotation, n_filter=n_filters, n_class=num_classes,
                     flip=flip)
    else:
        model = EqSimple(n_rot=n_rotation, n_filter=n_filters, n_class=num_classes,
                        flip=flip)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, 
                                                steps_per_epoch=len(train_loader), 
                                               epochs=epochs+1, cycle_momentum=False)

    model_save_path=save_path #path where model with best f1_macro should be stored

    #running variables
    epoch=0
    update_frequency=5 # number of batches before viewed acc and loss get updated
    counter=0 #counts batches
    f1_macro_best=0 #minimum f1_macro_score of the validation set for the first model to be saved
    loss_running=0
    acc_running=0
    val_batches=0

    y_pred=torch.tensor([], dtype=int)
    y_true=torch.tensor([], dtype=int)
    y_pred=y_pred.to(device)
    y_true=y_true.to(device)


    #Training

    for epoch in range(0, epochs):
        #training
        model.train()

        with tqdm(train_loader) as tepoch:   
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}")
                counter+=1

                x, y = data
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                
                logits = torch.softmax(out.detach(), dim=1)
                predictions = logits.argmax(dim=1)
                acc = accuracy_score(y.cpu(), predictions.cpu())

                if counter >= update_frequency:
                    tepoch.set_postfix(loss=loss.item(), accuracy=acc.item())
                    counter=0

        #validation       
        model.eval()
        with tqdm(valid_loader) as vepoch: 
            for i, data in enumerate(vepoch):
                vepoch.set_description(f"Validation {epoch+1}")

                x, y = data
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                logits = torch.softmax(out.detach(), dim=1)
                predictions = logits.argmax(dim=1)
                y_pred=torch.cat((y_pred, predictions), 0)
                y_true=torch.cat((y_true, y), 0)

                acc = accuracy_score(y_true.cpu(), y_pred.cpu())

                loss_running+=(loss.item()*len(y))
                acc_running+=(acc.item()*len(y))
                val_batches+=len(y)
                loss_mean=loss_running/val_batches
                acc_mean=acc_running/val_batches

                vepoch.set_postfix(loss=loss_mean, accuracy=acc_mean)

            f1_micro=f1_score(y_true.cpu(), y_pred.cpu(), average='micro')
            f1_macro=f1_score(y_true.cpu(), y_pred.cpu(), average='macro')
            print(f'f1_micro: {f1_micro}, f1_macro: {f1_macro}')  
            if f1_macro > f1_macro_best:
                f1_macro_best=f1_macro
                torch.save(model.state_dict(), model_save_path)
                print('model saved')

            #reseting running variables
            loss_running=0
            acc_running=0
            val_batches=0

            y_pred=torch.tensor([], dtype=int)
            y_true=torch.tensor([], dtype=int)
            y_pred=y_pred.to(device)
            y_true=y_true.to(device)



    print('Finished Training')
    
if __name__ == "__main__":
    main()
    
