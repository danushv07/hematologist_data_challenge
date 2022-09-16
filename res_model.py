from e2cnn import nn, gspaces
import torch
import numpy as np
from layers import EqConvBlock



class ResBlock(torch.nn.Module):
    """
    The ResNet block with convs+skip connections
    
    Parameters
    ----------------
    in_field : int
        The no. of input feature fields
    inner_field : int
        The no. of output fields from the conv. block
    grp_space : e2cnn.gspace
            The group space in which the a given signal lives
    b_norm : bool [Optional], by default True
            The flag to perform batch normalization after conv. layers
    Returns
    ---------------
    The forward method performs 
    conv -> bn -> ReLU -> conv -> bn -> ReLU -> shortcut+x
    """
    #todo: add the elu activation function
    def __init__(self, in_field, inner_field, grp_space, b_norm):
        super().__init__()

        #inner_filters = CreateField(grp_space, n_filters=in_field)
        self.conv_ = EqConvBlock(in_field, inner_field, 3, 1, 1, grp_space,
            b_norm=b_norm, res=False)
        #self.act_ = nn.ReLU(inner_filters)
        
    def forward(self, x):
        x_conv = self.conv_(x)
        #x_act = self.act_(x_conv)
        x_shortcut = nn.tensor_directsum([x_conv, x])  
        return x_shortcut



class EqRes(torch.nn.module):
    
    def __init__(self, n_rot, flip=False):
        
        super().__init__()
        self.n_theta = n_rot
        self.flip = flip
    
    
    if self.flip:
        if n_rot==1:
            self.gspace = gspaces.Flip2dOnR2()
        else:
            self.gspace = gspaces.FlipRot2dOnR2(self.n_theta)
    else:
        if n_rot==1:
            self.gspace = gspaces.TrivialOnR2()
        else:
            self.gspace = gspaces.Rot2dOnR2(self.n_theta)