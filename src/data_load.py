"""
This file contains the data preprocessing phases for the data challenge
"""

# import the dependencies
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

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


# The procedure has been partly adapted from the starter notebook provided by the 
# organizers of the challenge
# For details of each dataset- refer to train_book.ipynb

# the labels of the dataset
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
# The unlabeled WBC dataset gets the classname 'Data-Val' for every image

label_map_pred = {
        'DATA-VAL': 0
    }

# read the files and pre-processed mean from the file "metadata.csv"
metadata = pd.read_csv('./metadata.csv')
example_metadata=metadata
source_domains=['Ace_20', 'Mat_19']
source_index = example_metadata.dataset.isin(source_domains)
example_metadata = example_metadata.loc[source_index,:].copy().reset_index(drop = True)

def get_indexes(test_fraction, val_fraction, 
                example_metadata=example_metadata):
    """
    Function to split the dataset into train, test and validation sets
    
    Parameter
    -------------------------
    example_metadata : pd.DataFrame, with all the info. regarding the datasets
    test_fraction : float, the percentage of data for test set
    val_fraction : float, the percentage of data for validation set

    Returns
    ----------------------
    example_metadata : pd.DataFrame, updated version of the input
        the indexes of val and test data points are set
    train_index : the list of training sample indexes
    test_index : the list of testing sample indexes
    val_index : the list of validation sample indexes
    """
    # split the dataset into train & test index
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

    # split the dataset into train and validation index
    train_index, val_index, train_label, val_label = train_test_split(
        train_val_metadata.index,
        train_val_metadata.label + "_" + train_val_metadata.dataset,
        test_size=val_fraction,
        random_state=0, 
        shuffle=True, 
        stratify=train_val_metadata.label
        )
    example_metadata.loc[val_index, 'set']='val'
    
    return example_metadata, train_index, test_index, val_index

# the crop of images for each dataset
crop_Ace20=250
crop_Mat19=345
crop_WBC1=288

dataset_image_size = {
    "Ace_20":crop_Ace20,   #250,
    "Mat_19":crop_Mat19,   #345, 
    "WBC1":crop_WBC1,   #288,  
}


resize=224 #image pixel size

random_crop_scale=(0.8, 1.0)
random_crop_ratio=(0.8, 1.2)

mean=[0.485, 0.456, 0.406] #values from imagenet
std=[0.229, 0.224, 0.225] #values from imagenet

bs=32 #batchsize

# the transformations for the datasets
normalization = torchvision.transforms.Normalize(mean,std)

train_transform = transforms.Compose([ 
        transforms.Resize(resize),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1)),
        ]), p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=0.8, p=0.5),
        transforms.RandomEqualize(p=0.6)
])

val_transform = transforms.Compose([ 
        transforms.Resize(resize)])

test_transform = transforms.Compose([ 
        transforms.Resize(resize)])


# the dataset generator class for data curation
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
        iimage = image.to(dtype=torch.uint8)
        
        
        if self.transform:
            image = self.transform(image) 
            
        image = image / 255.
        image = normalization(image)
        
        
        label = self.label_map[label]
        label = torch.tensor(label).long()
        return image.float(),  label