
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
from data_load import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EqRes(n_rot=8, n_filter=32, n_class=11, flip=True)
model.to(device)
model.load_state_dict(torch.load('theta8_filter32_flip'))

def prediction(metadata, 
               source_domains=['Ace_20', 'Mat_19'], label_map=label_map_all):



    pred_dataset = DatasetGenerator(metadata, 
                                 reshape_size=resize, 
                                 dataset = source_domains,
                                 label_map=label_map, 
                                 transform = test_transform,
                                 )
    
    pred_loader = DataLoader(pred_dataset, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=0
                            )
    n=len(pred_loader)
    model.eval()
    preds=torch.tensor([], dtype=int)
    preds=preds.to(device)
    prediction=torch.tensor([])
    prediction=prediction.to(device)
    for i, data in enumerate(pred_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        x, y = data
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = torch.softmax(out.detach(), dim=1)
        prediction = torch.cat((prediction, logits), 0)
        predic = logits.argmax(dim=1)
        preds=torch.cat((preds, predic), 0)

    preds=preds.cpu()
    preds=preds.detach().numpy()
    np.save('preds', preds)
    y_pred = [label_map_reverse[p] for p in  preds]
    y_true=metadata['label']
    return y_true, y_pred, preds

metadata = pd.read_csv('./metadata.csv')
wbc_metadata=metadata.loc[metadata['dataset']=='WBC1'].reset_index(drop = True)

label_map_pred = {
        'DATA-VAL': 0
    }
y_true, y_pred, preds=prediction(metadata=wbc_metadata, source_domains=['WBC1'], label_map=label_map_pred)
outputdata=wbc_metadata.drop(columns=['file', 'label', 'dataset', 'set', 'mean1', 'mean2', 'mean3'])
outputdata['Label']=y_pred
outputdata['LabelID']=preds
'''
for i in range(len(y_pred)):
    outputdata['LabelID'].loc[i]=y_pred[i]
    outputdata['Label'].loc[i]=label_map_reverse[y_pred[i]]
'''
outputdata.to_csv('submission_8_flip.csv')
print(outputdata)
