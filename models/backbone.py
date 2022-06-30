# coding: utf-8
import torch.nn as nn
import torch
from .mynet3 import F_mynet3




def define_F(in_c, f_c, type='unet'):
    if type == 'mynet3':
        print("using mynet3 backbone")
        return F_mynet3(backbone='resnet18', in_c=in_c,f_c=f_c, output_stride=32)
    else:
        NotImplementedError('no such F type!')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




