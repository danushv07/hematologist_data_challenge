from e2cnn import nn, gspaces
import torch
import numpy as np
import torch.nn.functional as F

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
          act_func='r', bias=False):
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
                                stride=stride, frequencies_cutoff=f_cutoff, initialize=init, bias=bias))
    if b_norm:
        layer.add_module("bn", nn.InnerBatchNorm(out_type))
    if act:
        if act_func == 'r':
            layer.add_module("act", nn.ReLU(out_type))
        elif act_func == 'e':
            layer.add_module("act", nn.ELU(out_type))
    
    return layer


def EqConvBlock(in_field, out_field, k_size, pad, stride, grp_space, f_cutoff=.8*np.pi, 
            b_norm=False, res=False, act_func='r', change_stride=False):
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
    stride : list
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
                                stride=stride[i], frequencies_cutoff=f_cutoff))
        if(b_norm):
            layers.append(nn.InnerBatchNorm(out_type))
        #if(res and i==0):
        #    layers.append(nn.ReLU(out_type))
        #if not res:
        if not i==1:
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
        
        
class ResBlock(torch.nn.Module):
    """
    The equivariant resblock
    
    Parameters
    ----------------
    in_field : int
        The no. of input feature fields
    inner_field : int
        The no. of output fields from the conv. block
    grp_space : e2cnn.gspace
            The group space in which the a given signal lives
    kernel : int, 
            The kernel size for the max operation
    pad : int, 
            The padding size for the max operation
    stride : int, 
            The stride for the max operation
    b_norm : bool [Optional], by default True
            The flag to perform batch normalization after conv. layers
    
    Returns
    ---------------
    The forward method performs 
    conv -> bn -> ReLU -> conv -> bn -> ReLU -> shortcut+x
    """
    #todo: add the elu activation function
    def __init__(self, in_field, inner_field, grp_space, kernel,
                 pad, stride, b_norm=True):
        super().__init__()

        #inner_filters = CreateField(grp_space, n_filters=in_field)
        self.conv_ = EqConvBlock(in_field, inner_field, kernel, pad, stride, grp_space,
            b_norm=b_norm, res=False)
        #self.act_ = nn.ReLU(inner_filters)
        self.short=None
        if stride[0] > 1 or stride[1] > 1:
        #if stride !=1:
            self.short = EqConv(in_field, inner_field, 1,0,2, grp_space, bias=False)
        
    def forward(self, x):
        x_conv = self.conv_(x)
        
        #x_act = self.act_(x_conv)
        if self.short is not None:
            x_conv += self.short(x)
        else:
            x_conv += x
        #x_shortcut = nn.tensor_directsum([x_conv, x])  
        return x_conv
    
class EqRes(torch.nn.Module):
    """
    The equivariant resnet-18 module
    
    Parameters
    ----------------
    n_rot : int, the parameter "N" for the CN or DN group
    n_filter : int, the no of filters of the lifting layer,
        the filters are progressively doubled in the layers
    n_class : int, the number of classes in the dataset
    flip : bool, the flag to add the dihedral group
    """
    
    # the eq. resnet has been adapted and modified for the 
    # challenge from: https://github.com/QUVA-Lab/e2cnn/pull/46
    
    def __init__(self, n_rot, n_filter, n_class, flip=False):
        
        super().__init__()
        self.n_theta = n_rot
        self.flip = flip
        self.n_filter = n_filter
        self.num_class = n_class
    
    
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

        self.in_type = create_field(self.gspace, n_filters=3, rep_type=False)       
        out_type = create_field(self.gspace, n_filters=self.n_filter) #64
    
        # lifting layer
        self.lifting = nn.SequentialModule(
                nn.R2Conv(self.in_type, out_type, kernel_size=7, padding=3, stride=2, bias=False),
                nn.InnerBatchNorm(out_type),
                nn.ReLU(out_type, inplace=True),
                Pool(self.n_filter, self.gspace, k_size=2, pad=0, stride=2)
            )
        
        # block-layer 1
        self.block1 = ResBlock(self.n_filter, self.n_filter, self.gspace, 3,1,[1,1])
        self.block2 = ResBlock(self.n_filter, self.n_filter, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block2.conv_.out_type)
        
        # block-layer 2
        self.block3 = ResBlock(self.n_filter, self.n_filter*2, self.gspace, 3,1,[2,1])
        self.block4 = ResBlock(self.n_filter*2, self.n_filter*2, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block4.conv_.out_type)
        
        # block-layer 3
        self.block5 = ResBlock(self.n_filter, self.n_filter*2, self.gspace, 3,1,[2,1])
        self.block6 = ResBlock(self.n_filter*2, self.n_filter*2, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block6.conv_.out_type)
        
        # block-layer 4
        self.block7 = ResBlock(self.n_filter, self.n_filter*2, self.gspace, 3,1,[2,1])
        self.block8 = ResBlock(self.n_filter*2, self.n_filter*2, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block8.conv_.out_type)
        
        # projection layer
        self.project = nn.GroupPooling(create_field(self.gspace, self.n_filter))
        self.linear = torch.nn.Linear(len(self.project.out_type), self.num_class)
    
    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        
        x = self.lifting(x)
        
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.block7(x)
        x = self.block8(x)
        
        x = self.project(x)
        
        x = x.tensor
        b,c,w,h = x.shape
        
        x = F.avg_pool2d(x, (w,h))
        
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        
        return x

    
class EqSimple(torch.nn.Module):
    """
    The equivariant convolution module
    
    - ELU activation function is exclusively used
    - Max pooling is used for downsampling
    Parameters
    ----------------
    n_rot : int, the parameter "N" for the CN or DN group
    n_filter : int, the no of filters of the lifting layer,
        the filters are progressively doubled in the layers
    n_class : int, the number of classes in the dataset
    flip : bool, the flag to add the dihedral group
    """
    def __init__(self, n_rot, n_filter, n_class, flip=False):
        
        super().__init__()
        self.n_theta = n_rot
        self.flip = flip
        self.n_filter = n_filter
        self.num_class = n_class
    
    
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

        self.in_type = create_field(self.gspace, n_filters=3, rep_type=False)       
        out_type = create_field(self.gspace, n_filters=self.n_filter) #64
    
        # lifting layer
        self.lifting = nn.SequentialModule(
                nn.R2Conv(self.in_type, out_type, kernel_size=5, padding=2, stride=2, bias=False),
                nn.InnerBatchNorm(out_type),
                nn.ELU(out_type, inplace=True)
            )
        
        # block -1
        self.block1 = EqConvBlock(self.n_filter, self.n_filter, 3,1,[1,1], self.gspace, b_norm=True,
                                 act_func='e')
        self.pblock1 = Pool(self.n_filter, self.gspace, k_size=2)
        self.n_filter = len(self.pblock1.out_type)
        
        # block -2
        self.block2 = EqConvBlock(self.n_filter, self.n_filter*2, 3,1,[1,1], self.gspace, b_norm=True,
                                 act_func='e')
        self.pblock2 = Pool(self.n_filter*2, self.gspace, k_size=2)
        self.n_filter = len(self.pblock2.out_type)
        
        # block -3
        self.block3 = EqConvBlock(self.n_filter, self.n_filter*2, 3,1,[1,1], self.gspace, b_norm=True,
                                 act_func='e')
        self.pblock3 = Pool(self.n_filter*2, self.gspace, k_size=2)
        self.n_filter = len(self.pblock3.out_type)
        
        # block -4
        self.block4 = EqConvBlock(self.n_filter, self.n_filter*2, 3,1,[1,1], self.gspace, b_norm=True,
                                 act_func='e')
        self.pblock4 = Pool(self.n_filter*2, self.gspace, k_size=2)
        self.n_filter = len(self.pblock4.out_type)
        
        # projection layer
        self.project = nn.GroupPooling(create_field(self.gspace, self.n_filter))
        self.linear = torch.nn.Linear(len(self.project.out_type), self.num_class)
    
    
    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        
        x = self.lifting(x)
        
        
        x = self.block1(x)
        x = self.pblock1(x)
        
        x = self.block2(x)
        x = self.pblock2(x)
        
        x = self.block3(x)
        x = self.pblock3(x)
        
        x = self.block4(x)
        x = self.pblock4(x)
        
        x = self.project(x)
        
        x = x.tensor
        b,c,w,h = x.shape
        
        x = F.avg_pool2d(x, (w,h))
        
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        
        return x