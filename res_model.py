from e2cnn import nn, gspaces
import torch
import numpy as np
from layers import EqConvBlock, create_field, Pool, EqConv
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
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

        self.lifting = nn.SequentialModule(
                nn.R2Conv(self.in_type, out_type, kernel_size=7, padding=3, stride=2, bias=False),
                nn.InnerBatchNorm(out_type),
                nn.ReLU(out_type, inplace=True),
                Pool(self.n_filter, self.gspace, k_size=2, pad=0, stride=2)
            )
        
        self.block1 = ResBlock(self.n_filter, self.n_filter, self.gspace, 3,1,[1,1])
        self.block2 = ResBlock(self.n_filter, self.n_filter, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block2.conv_.out_type)
        
        self.block3 = ResBlock(self.n_filter, self.n_filter*2, self.gspace, 3,1,[2,1])
        self.block4 = ResBlock(self.n_filter*2, self.n_filter*2, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block4.conv_.out_type)
        
        self.block5 = ResBlock(self.n_filter, self.n_filter*2, self.gspace, 3,1,[2,1])
        self.block6 = ResBlock(self.n_filter*2, self.n_filter*2, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block6.conv_.out_type)
        
        self.block7 = ResBlock(self.n_filter, self.n_filter*2, self.gspace, 3,1,[2,1])
        self.block8 = ResBlock(self.n_filter*2, self.n_filter*2, self.gspace, 3,1,[1,1])
        self.n_filter = len(self.block8.conv_.out_type)
        
        self.project = nn.GroupPooling(create_field(self.gspace, self.n_filter))
        self.linear = torch.nn.Linear(len(self.project.out_type), self.num_class)
    
    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        #print(f"before lift: {x.shape}")
        x = self.lifting(x)
        #print(f"after lift: {x.shape}")
        
        x = self.block1(x)
        x = self.block2(x)
        #print(f"after layer1: {x.shape}")
        
        x = self.block3(x)
        x = self.block4(x)
        #print(f"after layer2: {x.shape}")
        
        x = self.block5(x)
        x = self.block6(x)
        #print(f"after layer3: {x.shape}")
        
        x = self.block7(x)
        x = self.block8(x)
        #print(f"after layer4: {x.shape}")
        
        x = self.project(x)
        #print(f"after gpool: {x.shape}")
        
        x = x.tensor
        b,c,w,h = x.shape
        
        x = F.avg_pool2d(x, (w,h))
        #print(f"after avgpool: {x.shape}")
        
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        
        return x