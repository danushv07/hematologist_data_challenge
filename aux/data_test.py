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

data_path = {
        "Ace_20": "/bigdata/haicu/venkat31/hemat/Acevedo/", # Acevedo_20 Dataset
        "Mat_19": "/bigdata/haicu/venkat31/hemat/Matek/", # Matek_19 Dataset
        "WBC1": "/bigdata/haicu/venkat31/hemat/WBC1/" # WBC1 dataset
    }

label_map_all = {
        'basophil': 0,
        'eosinophil': 1,
        'erythroblast': 2,
        'myeloblast' : 3,
        'promyelocyte': 4,
        'myelocyte': 5,
        'metamyelocyte': 6,
        'neutrophil_banded': 7,
        'neutrophil_segmented': 8,
        'monocyte': 9,
        'lymphocyte_typical': 10
    }

label_map_reverse = {
        0: 'basophil',
        1: 'eosinophil',
        2: 'erythroblast',
        3: 'myeloblast',
        4: 'promyelocyte',
        5: 'myelocyte',
        6: 'metamyelocyte',
        7: 'neutrophil_banded',
        8: 'neutrophil_segmented',
        9: 'monocyte',
        10: 'lymphocyte_typical'
    }

# The unlabeled WBC dataset gets the classname 'Data-Val' for every image

label_map_pred = {
        'DATA-VAL': 0
    }

metadata = pd.read_csv('./metadata.csv')
example_metadata=metadata
source_domains=['Ace_20', 'Mat_19']
source_index = example_metadata.dataset.isin(source_domains)
example_metadata = example_metadata.loc[source_index,:].copy().reset_index(drop = True)

test_fraction=0.2 #of the whole dataset
val_fraction=0.125 #of 0.8 of the dataset (corresponds to 0.1 of the whole set)

train_index, test_index, train_label, test_label = train_test_split(
    example_metadata.index,
    example_metadata.label + "_" + example_metadata.dataset,
    test_size=test_fraction,
    random_state=0, 
    shuffle=True,
    stratify=example_metadata.label
    )
example_metadata.loc[test_index, 'set']='test'
train_val_metadata=example_metadata.loc[train_index]

train_index, val_index, train_label, val_label = train_test_split(
    train_val_metadata.index,
    train_val_metadata.label + "_" + train_val_metadata.dataset,
    test_size=val_fraction,
    random_state=0, 
    shuffle=True, 
    stratify=train_val_metadata.label
    )
example_metadata.loc[val_index, 'set']='val'

crop_Ace20=250
crop_Mat19=345
crop_WBC1=288

dataset_image_size = {
    "Ace_20":crop_Ace20,   #250,
    "Mat_19":crop_Mat19,   #345, 
    "WBC1":crop_WBC1,   #288,  
}

class DatasetGenerator(Dataset):

    def __init__(self, 
                metadata, 
                reshape_size=64, 
                label_map=[],
                dataset = [],
                transform=None,
                selected_channels = [0,1,2],
                dataset_image_size=None):

        self.metadata = metadata.copy().reset_index(drop = True)
        self.label_map = label_map
        self.transform = transform
        self.selected_channels = selected_channels
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        dataset =  self.metadata.loc[idx,"dataset"]
        crop_size = dataset_image_size[dataset]
        
        h5_file_path = self.metadata.loc[idx,"file"]
        image= imread(h5_file_path)[:,:,self.selected_channels]
        
        h1 = (image.shape[0] - crop_size) /2
        h1 = int(h1)
        h2 = (image.shape[0] + crop_size) /2
        h2 = int(h2)
        
        w1 = (image.shape[1] - crop_size) /2
        w1 = int(w1)
        w2 = (image.shape[1] + crop_size) /2
        w2 = int(w2)
        image = image[h1:h2,w1:w2, :]
        image = np.transpose(image, (2, 0, 1))
        label = self.metadata.loc[idx,"label"]
 
    
        mean=[0.485, 0.456, 0.406] #values from imagenet
        std=[0.229, 0.224, 0.225] #values from imagenet
        normalization = torchvision.transforms.Normalize(mean,std)
        
        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image)) 
        image = image.to(dtype=torch.uint8)
        
        if self.transform:
            image = self.transform(image) 
        
        image = image / 255.
        #image = image.float()
        image = normalization(image)
        
        
        label = self.label_map[label]
        label = torch.tensor(label).long()
        return image.float(),  label
    
resize=224 #image pixel size
number_workers=0

random_crop_scale=(0.8, 1.0)
random_crop_ratio=(0.8, 1.2)

mean=[0.485, 0.456, 0.406] #values from imagenet
std=[0.229, 0.224, 0.225] #values from imagenet

bs=32 #batchsize

normalization = torchvision.transforms.Normalize(mean,std)

train_transform = transforms.Compose([ 
        #normalization,
        transforms.Resize(resize),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1)),
            #transforms.RandomPerspective()
        ]), p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.5),
        transforms.RandomEqualize(p=0.6)
])

val_transform = transforms.Compose([ 
        #normalization,
        transforms.Resize(resize)])

test_transform = transforms.Compose([ 
        #normalization,
        transforms.Resize(resize),
        
])

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
