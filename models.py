"""This files contains all layers for the equivariant networks developed in this project
   Author: Danush Kumar Venkatesh
   Date:
"""

# import the dependencies
from e2cnn import nn, gspaces
import torch
import numpy as np
from layers import (create_field, EqConv, UEncode,
                   UDecode)


class EQNet(torch.nn.Module):
    """
    The Equivariant U-net class
    
    Parameters
    ---------------------
    n_rot : int
        The number of rotations for the cyclic group
    flip : bool, [Optional], dafault - False
        The flag to include dihedral group
        False - only cyclic group is considered
    n_filter : int, [Optional], default - 32
        The no. of filters to be used
        The specified filter are divided by sqrt(n_rot) to avoid explosion
        of filter
    backbone : str, [Optional], default - "unet"
        The encoding part of the unet
        available encoders, "unet", "resnet", "vgg"
    depth : int, [Optional], default - 2
        The depth of the unet 
    lkernel_size : int, [Optional], default - 3
        The kernel size of the lifting layers
    lpad : int, [Optional], default - 1
        The padding size of the lifting layers
    conv_kernel : int, [Optional], default - 3
        The kernel size of the conv. layers for both downsample & upsample
    conv_pad : int, [Optional], default - 1
       The pad size of the conv. layers for both downsample & upsample
    conv_stride : int, [Optional], default - 1
        The stride size of the conv. layers for both downsample & upsample
    f_cutoff : float, [Optional], default - 0.8*np.pi
        The cut of frequncy for the filter bassi expansion
    conv_bnorm : bool, [Optional], default - True
        to include batch normalization between conv. layers
    pool_kernel : int, [Optional], default - 2
        The kernel size of pooling layers
    pool_pad : int, [Optional], default - 0
        The pad size of pooling layers
    pool_stride : int, [Optional], default - 2
        The stride size of pooling layers
    pool_type : str, [Optional], default - "max"
        The pool type for pooling layers
        available: "max", "avg"
        The kernel size of pooling layers
    pool_alias : bool, [Optional], default - True
        to include alias for pool type
    scale_factor : int, [Optional], default -2
        The upscaling scale factor
    rescale_params : bool,[Optional] default - True
        The option to perform rescaling filters
    final_layer : str, [Optional], default -True
        The type of final layer to use
        options - "gpool" - GroupPooloing+Conv2d
    n_class : int, [Optional], default -1
        The no. of classes to segment
    
    """
    def __init__(self, n_rot, flip=False, n_filter=32, backbone="unet", depth=2, lkernel_size=3, lpad=1, conv_kernel=3, 
                 conv_pad=1, conv_stride=1, f_cutoff=0.8*np.pi, conv_bnorm=False, pool_kernel=2, pool_padd=0, pool_stride=2,
                 pool_type="max", pool_alias=True, scale_factor=2, rescale_params=True, final_layer='gpool', n_class=1,
                act_func='r'):
                 
        super().__init__()
        self.n_theta = n_rot
        self.flip = flip
        self.n_filter = n_filter
        self.encoder = backbone
        self.n_depth = depth
        self.lift_kernel = lkernel_size
        self.lift_pad = lpad
        self.k_size = conv_kernel
        self.k_pad = conv_pad
        self.k_stride = conv_stride
        self.frequency = f_cutoff
        self.b_norm = conv_bnorm
        self.pk_size = pool_kernel
        self.p_pad = pool_padd
        self.p_stride = pool_stride
        self.p_type = pool_type
        self.p_alias = pool_alias
        self.scale = scale_factor
        self.fl = final_layer
        self.activation = act_func
        
        
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
        

        if rescale_params:
            n_filter = int(self.n_filter / np.sqrt(self.gspace.fibergroup.order()))
            print(f" rescaling filter number to {n_filter}")
            s = n_filter
            
        self.in_type = create_field(self.gspace, rep_type=False)       
        out_type = create_field(self.gspace, n_filters=n_filter)
        #if self.activation:
        #    act_layer
        # lifting layer 
        self.lifting = nn.SequentialModule(
            nn.R2Conv(self.in_type, out_type, kernel_size=self.lift_kernel, padding=self.lift_pad),
            nn.ReLU(out_type, inplace=True)
        )
        
        self.convs_down, self.max_down, n_filter = UEncode(n_filter=n_filter, depth=self.n_depth, grp_space=self.gspace, 
                                                            k_size=self.k_size, backbone=self.encoder, pad=self.k_pad,
                                                            stride=self.k_stride, f_cutoff=self.frequency, b_norm=self.b_norm,   
                                                            pk_size=self.pk_size, ppad=self.p_pad, pstride=self.p_stride, 
                                                            pool_type=self.p_type, alias=self.p_alias, act_func=self.activation)
        
        self.up_, self.convs_up, n_filter = UDecode(n_filter=n_filter, depth=self.n_depth, grp_space=self.gspace, k_size=self.k_size, 
                                                   pad=self.k_pad, stride=self.k_stride, f_cutoff=self.frequency,
                                                   b_norm=self.b_norm, scale=self.scale, align=True, act_func=self.activation)
        
        #self.final = torch.nn.Conv2d(s*self.n_theta, 1, kernel_size=3, padding=1)
        # convert from group space to Z^2 space
        if self.fl == 'gpool':
            #self.fconv = EqConv(in_field=len(self.convs_up[-1].out_type), out_field=self.n_theta, k_size=3, pad=1, stride=1,
            #                    grp_space=self.gspace, act=True)
            #self.pool = nn.GroupPooling(CreateField(self.gspace, len(self.fconv.out_type)))
            self.pool = nn.GroupPooling(create_field(self.gspace, n_filter))
            self.tfinal = torch.nn.Conv2d(len(self.pool.out_type), n_class, kernel_size=3, padding=1)
        else:
            self.pool = EqConv(in_field=s, out_field=1, k_size=3, pad=1, stride=1, 
                                grp_space=self.gspace, f_cutoff=None, out_rep=False)
            self.tfinal = torch.nn.Conv2d(1, n_class, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        x = self.lifting(x)

        convs = []
        for lay, maxp in zip(self.convs_down[:-1], self.max_down):
            c = lay(x)
            x = maxp(c)
            convs.append(c)

        x = self.convs_down[-1](x)

        for up, lay, c in zip(self.up_, self.convs_up, convs[::-1]):
            x = up(x)
            x = nn.tensor_directsum([x, c])
            x = lay(x)
        
        #xf = self.fconv(x)
        #xg = self.pool(xf)
        x = self.pool(x)
        x = self.tfinal(x.tensor)
        return x
    
    
class Unet(torch.nn.Module):
    """
    A simple U-net model for comparison with EQNet
    """
    def __init__(self, depth=2, kernel_size=3, conv_pad=1, pool=2, n_filter=32, in_channel=1,
                out_channel=1, multi_star=False):
        super().__init__()
        self.n_depth = depth
        self.n_filter = n_filter
        self.k_size = kernel_size
        self.c_pad = conv_pad
        self.pk_size = pool
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.r_out = multi_star
        
        self.firstconv = torch.nn.Sequential(torch.nn.Conv2d(self.in_channel, self.n_filter, kernel_size=self.k_size,
                                                            padding=self.c_pad),torch.nn.ReLU(inplace=True))
        
        self.convs_down, self.max_down, n_filter = self.encode(self.n_filter, self.n_depth, self.k_size)
        self.up_, self.covs_up, n_filter = self.decode(n_filter, self.n_depth, self.k_size)
        
        self.final = torch.nn.Conv2d(n_filter, self.out_channel, kernel_size=self.k_size, padding=self.c_pad)
        
        if self.r_out:
            self.rfinal = torch.nn.Conv2d(n_filter, 16, kernel_size=self.k_size, padding=self.c_pad)
        
    def double_conv(self, in_channels, out_channels, ksize, pad, b_norm):
        layers = []
        for i in range(2):
            layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad))
            if(b_norm):
                layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU(inplace=True))
            in_channels = out_channels
        return torch.nn.Sequential(*layers)

    def pool(self, k_size=2, pad=0, stride=2, pool_type="max"):
        if pool_type == "max":
            return torch.nn.MaxPool2d(kernel_size=k_size, padding=pad, stride=stride)
        elif pool_type == "avg":
            return torch.nn.AvgPool2d(kernel_size=k_size, padd=pad, stride=stride)

    def upsample(self, scale, align):
        return torch.nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=align)
    
    def resblock(self):
        conv_ = double_conv(self, in_channel, out_channel, ksize, pad, bnorm=True)

    def encode(self, n_filter, depth, k_size, pad=1, stride=1, b_norm=False, pk_size=2, ppad=0, pstride=2, 
               pool_type="max"):

        convs_down = torch.nn.ModuleList() # conv. operation
        max_down = torch.nn.ModuleList()  # pooling operation
        in_channel = n_filter

        for i in range(depth+1):
            out_channel = n_filter*2**i

            convs_down.append(self.double_conv(in_channel, out_channel, k_size, pad, b_norm=b_norm))
            if(i<= depth): 
                max_down.append(self.pool(pk_size, ppad, pstride, pool_type=pool_type))

            in_channel = out_channel
        return convs_down, max_down, in_channel

    def decode(self, n_filter, depth, k_size, pad=1, stride=1, b_norm=False, scale=2, align=True):
        convs_up = torch.nn.ModuleList()
        up_sample = torch.nn.ModuleList()
        in_channel = n_filter

        for i in range(depth-1, -1, -1):
            out_channel = in_channel//2

            up_sample.append(self.upsample(scale=scale, align=align))
            convs_up.append(self.double_conv(in_channel+out_channel, out_channel, k_size, pad, 
                                             b_norm=b_norm)) 
            in_channel = out_channel

        return up_sample, convs_up, out_channel
    
    def forward(self, x):
        x = self.firstconv(x)
        convs = []
        for lay, maxp in zip(self.convs_down[:-1], self.max_down):
            c = lay(x)
            x = maxp(c)
            convs.append(c)
        
        x = self.convs_down[-1](x)
        
        for up, lay, c in zip(self.up_, self.covs_up, convs[::-1]):
            x = up(x)
            x = torch.cat([x, c], dim=1)
            x = lay(x)
        
        if self.r_out:
            y = self.rfinal(x)
        x = self.final(x)
        
        return x
    

        