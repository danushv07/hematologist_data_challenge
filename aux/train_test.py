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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, matthews_corrcoef, classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,  precision_score, recall_score

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

from res_model import *
from data_test import *

@click.command()
@click.option('-nr', 'rots', type=int, help="no. of rotations")
@click.option('-nf', 'filters', type=int, help="no. of filters")
@click.option('-rf', 'flip', type=bool, help="no. of filters")
@click.option('-ml', 'mdl_path', type=str, help="the model save path")
@click.option('-mo', 'mdl', type=str, help="the option for model")
def main(rots, filters, flip, mdl_path, mdl):
    data_path = {
            "Ace_20": "/bigdata/haicu/venkat31/hemat/Acevedo/", # Acevedo_20 Dataset
            "Mat_19": "/bigdata/haicu/venkat31/hemat/Matek/", # Matek_19 Dataset
            "WBC1": "/bigdata/haicu/venkat31/hemat/WBC1/" # WBC1 dataset
        }

    #dataset-creation

    train_dataset = DatasetGenerator(example_metadata.loc[train_index,:], 
                                     reshape_size=resize, 
                                     dataset = source_domains,
                                     label_map=label_map_all, 
                                     transform = train_transform,
                                     )
    val_dataset = DatasetGenerator(example_metadata.loc[val_index,:], 
                                     reshape_size=resize, 
                                     dataset = source_domains,
                                     label_map=label_map_all, 
                                     transform = val_transform,
                                     )

    test_dataset = DatasetGenerator(example_metadata.loc[test_index,:], 
                                     reshape_size=resize, 
                                     dataset = source_domains,
                                     label_map=label_map_all, 
                                     transform = test_transform,
                                     )

    number_workers=0
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=number_workers)
    valid_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=True, num_workers=number_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=number_workers)

    epochs=10 # max number of epochs
    lr=1e-5 # learning rate
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 11
    if mdl=='res':
        model = EqRes(n_rot=rots, n_filter=filters, n_class=num_classes, flip=flip)
    else:
        model = EqSimple(n_rot=rots, n_filter=filters, n_class=num_classes)
    #model = torch.nn.DataParallel(model) 
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, 
                                                steps_per_epoch=len(train_loader), 
                                               epochs=epochs+1, cycle_momentum=False)

    model_save_path=mdl_path #path where model with best f1_macro should be stored

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
    
    #model.load_state_dict(torch.load('theta4_filter32'))
    
if __name__ == "__main__":
    main()
    