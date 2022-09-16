"""This files contains all layers for the equivariant networks developed in this project
   Author: Danush Kumar Venkatesh
   Date:
"""

# import the dependencies
from e2cnn import nn, gspaces
import torch
import numpy as np


def create_field(grp_space, n_filters=1, rep_type=True):
    """
    Function to create the e2cnn.nn.FieldType of trivial/regular representation 
    of n_filters
    
    Parameters
    ----------------
    grp_space : e2cnn.gspace
            The group space in which the a given signal lives
    n_filters : int, [Optional] by default 1
            The no. of feature fields to be created
    rep_type : bool, [Optional], by default True
            The required represention of the gspace
            if True - regular representation 
            if False - trivial representation
    
    Return
    ---------------
    FieldType of gspace with the n_filters of feature fields
    """
    if rep_type:
        return nn.FieldType(grp_space, n_filters*[grp_space.regular_repr])
    else:
        return nn.FieldType(grp_space, n_filters*[grp_space.trivial_repr])

    
def EqConv(in_field, out_field, k_size, pad, stride, grp_space, f_cutoff=.8*np.pi, 
            b_norm=False, init=True, in_rep=True, out_rep=True, act=False,
          act_func='r'):
    """
    The equi. conv. operation
    - can be used for designing blocks for other model arch.
    
    Parameters
    ----------------
    in_field : int
            The no. of input feature fields
    out_field : int
            The no. of output feature fields
    k_size : int
            The kernel size for the conv. operation
    pad : int
            The padding size for the conv. operation
    stride : int
            The stride for the conv. operation
    grp_space : e2cnn.gspace
            The group space in which the a given signal lives
    f_cutoff : float [Optional, by default 0.8*np.pi
            The frequency cutoff for the conv. operation
            multiple of np.pi
    b_norm : bool [Optional], by default False
            The flag to perform batch normalization after conv. layers
    init : bool [Optional], by default True
            The defualt initialization in e2cnn is he' method, as suggested in 
            e2cnn documentation vary this parameter depending on task
    act : bool [Optional], by default False
            The flag to add relu activation after batch norm layers
    act_func : str, [Optional], by default - 'r'
            The flag to chose betwwen relu('r) & elu('e')
            
    Return
    ----------------
    Sequential module of e2cnn with (1x) Conv. with batch normalization(if specified)
    
    """
    in_type = create_field(grp_space, in_field, in_rep)
    out_type = create_field(grp_space, out_field, out_rep)
    layer = nn.SequentialModule(nn.R2Conv(in_type, out_type, kernel_size=k_size, padding=pad, 
                                stride=stride, frequencies_cutoff=f_cutoff, initialize=init))
    if b_norm:
        layer.add_module("bn", nn.InnerBatchNorm(out_type))
    if act:
        if act_func == 'r':
            layer.add_module("act", nn.ReLU(out_type))
        elif act_func == 'e':
            layer.add_module("act", nn.ELU(out_type))
    
    return layer


def EqConvBlock(in_field, out_field, k_size, pad, stride, grp_space, f_cutoff=.8*np.pi, 
            b_norm=False, res=False, act_func='r'):
    """
    The equi. conv. block for the U-Net.
    
    - the direct conversion of feature space to g-space is implemented within the function
    
    Parameters
    ----------------
    in_field : int
        The no. of input feature fields
    out_field : int
        The no. of output feature fields
    k_size : int
        The kernel size for the conv. operation
    pad : int
        The padding size for the conv. operation
    stride : int
        The stride for the conv. operation
    grp_space : e2cnn.gspace
        The group space in which the a given signal lives
    f_cutoff : float [Optional, by default 0.8*np.pi
        The frequency cutoff for the conv. operation
        multiple of np.pi
    b_norm : bool [Optional], by default False
        The flag to perform batch normalization after conv. layers
    res : bool [Optional], by default False
        The block for resnet models
    act_func : str, [Optional], by default - 'r'
            The flag to chose betwwen relu('r) & elu('e')
            
    Return
    ----------------
    Sequential module of e2cnn with (2x) Conv. each follwed by ReLU activation
    
    """
    in_type = create_field(grp_space, in_field)
    out_type = create_field(grp_space, out_field)
    layers = []
    
    # TODO: include option for other activation
    for i in range(2):
        layers.append(nn.R2Conv(in_type, out_type, kernel_size=k_size, padding=pad, 
                                stride=stride, frequencies_cutoff=f_cutoff))
        if(b_norm):
            layers.append(nn.InnerBatchNorm(out_type))
        #if(res and i==0):
        #    layers.append(nn.ReLU(out_type))
        #if not res:
        if act_func == 'r':
            act_layer = nn.ReLU(out_type)
        elif act_func == 'e':
            act_layer = nn.ELU(out_type)
        
        #ayers.append(nn.ReLU(out_type))
        layers.append(act_layer)
        in_type = out_type
    return nn.SequentialModule(*layers)


def Pool(in_field, grp_space, k_size=2, pad=0, stride=2, pool_type="max", alias=True, sig=0.6):
    """
    The equi. max pooling block
    
    Parameters
    -------------
    in_field : int
            The no. of input feature fields
    grp_space : e2cnn.gspace
            The group space in which the a given signal lives
    k_size : int, [Optional], by default 2
            The kernel size for the max operation
    pad : int, [Optional], by default 0
            The padding size for the max operation
    stride : int, [Optional], by default 2
            The stride for the max operation
    pool_type : str, [Optional] by default "max"
            "max" - MaxPool opration is applied
            "avg" - Avg. pool option is applied
    alias : bool, [Optiona], by default True
            if True - PointWise(Max/Avg)PoolAAntialisaed option based on shift-invariant 
                        conv. operation
            if False - PointWise(Max/Avg)Pool option is selected
    sig : float, [Optional], by default 0.6
            The std. deviation of gaussian blur
    
    Return
    --------------
    Channel-wise max pooled feature maps of specified options
    """
    in_type = create_field(grp_space, in_field)
    
    if(pool_type == "max"):
        if alias:
            return nn.PointwiseMaxPoolAntialiased(in_type, kernel_size=k_size, stride=stride,
                                                  padding=pad, sigma=sig)
        else:
            return nn.PointwiseMaxPool(in_type, kernel_size=k_size, stride=stride,
                                                  padding=pad)
    elif(pool_type == "avg"):
        if alias:
            return nn.PointwiseAvgPoolAntialiased(in_type, stride=stride, sigma=sig,
                                                  padding=pad)                                                  
        else:
            return nn.PointwiseAvgPool(in_type, kernel_size=k_size, stride=stride,
                                              padding=pad)
    else:
        raise NotImplemented

def UpSample(in_field, grp_space, scale=2, align=True):
    """
    Function for upsampling on the given feature field
    
    NOTE: only "bilinear" mode preserves equivariance, so the default is used as in e2cnn
    Parameter
    ------------
    in_field : int
            The no. of input feature fields
    grp_space : e2cnn.gspace
            The group space in which the a given signal lives
    scale : int, [Optional], by default 2
        The scaling factor for the Upsampling
    align : bool, [Optional], by default True
        For the alignment of corner pixels
    
    Return
    ----------------
    Sequential module of e2cnn with (1x) UPsampled feature fields
    """
    in_type = create_field(grp_space, in_field)
    return nn.SequentialModule(nn.R2Upsampling(in_type, scale_factor=scale, align_corners=align))

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
    

    
class VGGBlock(torch.nn.Module):
    """
    The VGGNet block 
    
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
    conv_stackv : bool [Optional], by default True
        The 1x1 conv block in 3,4,5 block for VGG 16 arch
        if False: follows VGG 13 arch
    Returns
    ---------------
    The forward method performs 
    conv -> bn -> ReLU -> conv -> bn -> ReLU -> shortcut+x
    """
    def __init__(self, in_field, inner_field, grp_space, b_norm, conv_stack=True):
        super().__init__()

        #inner_filters = CreateField(grp_space, n_filters=in_field)
        self.conv = EqConvBlock(in_field, inner_field, 3, 1, 1, grp_space,
            b_norm=b_norm, res=False)
        self.conv1x1 = EqConv(inner_field, inner_field, 1, 0, 1, grp_space,
                             b_norm=b_norm)
        self.conv_add = conv_stack
        #self.act_ = nn.ReLU(inner_filters)
        
    def forward(self, x):
        x_conv = self.conv(x)
        #x_act = self.act_(x_conv)
        if self.conv_add:
            x_conv = self.conv1x1(x_conv)
        return x_conv

def UEncode(n_filter, depth, grp_space, k_size, backbone="unet", pad=1, stride=1, f_cutoff=0.8*np.pi, 
            b_norm=False, pk_size=2, ppad=0, pstride=2, pool_type="max", alias=True, act_func='r'):
    """
    The down sampling part of the Unet
    - at each value of depth the n_filter is multiplied
    Parameter
    ------------------
    n_filer : int
        The inital no. of feature fields to start the encoding block
    depth : int
        The depth of the UNet
    grp_space : e2cnn.gspace
        The group space in which the a given signal lives
    k_size : int
        The kernel size for the conv. operation
    backbone : str, [Optional], by default "unet"
        "resnet" - the resnet arch. for the encoder
    pad : int, [Optional], by default 1
        The padding size for the conv. operation
    stride : int, [Optional], by default 1
        The stride for the conv. operation
    f_cutoff : float [Optional, by default None (maybe 0.8*np.pi)
        The frequency cutoff for the conv. operation
        multiple of np.pi
    b_norm : bool [Optional], by default False
        The flag to perform batch normalization after conv. layers
    pk_size : int, [Optional], by default 2
        The kernel size for the max operation
    ppad : int, [Optional], by default 0
        The padding size for the max operation
    pstride : int, [Optional], by default 2
        The stride for the max operation
    pool_type : str, [Optional] by default "max"
        "max" - MaxPool opration is applied
        "avg" - Avg. pool option is applied
    alias : bool, [Optiona], by default True
        if True - PointWise(Max/Avg)PoolAAntialisaed option based on shift-invariant 
                    conv. operation
        if False - PointWise(Max/Avg)Pool option is selected
    act_func : str, [Optional], by default - 'r'
            The flag to chose betwwen relu('r) & elu('e')
            
    Return
    ------------
    convs_down : list
        Each value is a SequentialModule of (2X) Eq. conv. block with ReLU activation
    max_down : list
        Each value is the Pooling operation applied channel-wise
    in_filter : int
        The no. of filters used in the last layer
    """
    convs_down = torch.nn.ModuleList() # conv. operation
    max_down = torch.nn.ModuleList()  # pooling operation
    in_filter = n_filter
    
    #TODO: log the feature fields for each depth
    for i in range(depth+1):
        out_filter = n_filter*2**i
        
        if(backbone=="unet"):
            convs_down.append(EqConvBlock(in_field=in_filter, out_field=out_filter, k_size=k_size, 
                                         pad=pad, stride=stride, grp_space=grp_space, f_cutoff=f_cutoff,
                                         b_norm=b_norm, act_func=act_func))
            if(i<= depth): 
                max_down.append(Pool(in_field=out_filter, grp_space=grp_space, k_size=pk_size,
                                    pad=ppad, stride=pstride, pool_type=pool_type, alias=alias))
        
            in_filter = out_filter
        
        elif(backbone=="resnet"):
            convs_down.append(ResBlock(in_filter, out_filter, grp_space, b_norm=True))
            if(i<= depth): 
                max_down.append(Pool(in_field=out_filter+in_filter, grp_space=grp_space, k_size=pk_size,
                                    pad=ppad, stride=pstride, pool_type=pool_type, alias=alias))
            
            in_filter += out_filter
        
        elif(backbone=="vgg"):
            convs_down.append(VGGBlock(in_filter, out_filter, grp_space, b_norm=True))
            if(i<= depth): 
                max_down.append(Pool(in_field=out_filter, grp_space=grp_space, k_size=pk_size,
                                    pad=ppad, stride=pstride, pool_type=pool_type, alias=alias))
            
            in_filter = out_filter
              
    return convs_down, max_down, in_filter


def UDecode(n_filter, depth, grp_space, k_size, pad=1, stride=1, f_cutoff=.8*np.pi, b_norm=False,
           scale=2, align=True, act_func='r'):
    """
    The up sampling part of the Unet
    - at each value of depth the n_filter is halved
    
    Parameters
    -------------
    n_filer : int
        The final no. of feature fields to from the encoding block
        from convs_down[-1].size, represents the final filter size
    depth : int
        The depth of the UNet
    grp_space : e2cnn.gspace
        The group space in which the a given signal lives
    k_size : int
        The kernel size for the conv. operation
    pad : int, [Optional], by default 1
        The padding size for the conv. operation
    stride : int, [Optional], by default 1
        The stride for the conv. operation
    f_cutoff : float [Optional, by default 0.8*np.pi
        The frequency cutoff for the conv. operation
        multiple of np.pi
    b_norm : bool [Optional], by default False
        The flag to perform batch normalization after conv. layers
    scale : int, [Optional], by default 2
        The scaling factor for the Upsampling
    align : bool, [Optional], by default True
        For the alignment of corner pixels
    act_func : str, [Optional], by default - 'r'
            The flag to chose betwwen relu('r) & elu('e')
        
        Return
    ------------
    convs_up : list
        Each value is a SequentialModule of (2X) Eq. conv. block with ReLU activation
    up_sample : list
        Each value is the Upsampled feature field 
    in_filter : int
        The no. of filters used in the first layer
    """
    convs_up = torch.nn.ModuleList()
    up_sample = torch.nn.ModuleList()
    in_filter = n_filter
    
    for i in range(depth-1, -1, -1):
        out_filter = in_filter//2
        up_sample.append(UpSample(in_filter, grp_space=grp_space))
        convs_up.append(EqConvBlock(in_field=in_filter+out_filter, out_field=out_filter, k_size=k_size, 
                                     pad=pad, stride=stride, grp_space=grp_space, f_cutoff=f_cutoff,
                                     b_norm=b_norm, act_func=act_func))
        in_filter = out_filter
        
    return up_sample, convs_up, out_filter