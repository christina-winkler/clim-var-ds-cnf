"""
This script implements constraints for mass conservation in downscaling operations using PyTorch.
Taken from: https://github.com/RolnickLab/constrained-downscaling/blob/main/models.py
Classes:
    - MultDownscaleConstraints: Enforces multiplicative constraints.
    - AddDownscaleConstraints: Enforces additive constraints.
    - ScAddDownscaleConstraints: Enforces signed additive constraints.
    - SoftmaxConstraints: Enforces softmax constraints.

Each class defines a downscaling method that aims to ensure the total mass is conserved during the transformation process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class MultDownscaleConstraints(nn.Module):
    """mul"""
    def __init__(self, upsampling_factor):
        super(MultDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        out = self.pool(y)
        out = y*torch.kron(lr*1/out, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out

class AddDownscaleConstraints(nn.Module):
    """add"""
    def __init__(self, upsampling_factor):
        super(AddDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        out =y+ torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out

class ScAddDownscaleConstraints(nn.Module):
    """
    scadd
    """
    def __init__(self, upsampling_factor):
        super(ScAddDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        diff_P_x = torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        sigma = torch.sign(-diff_P_x)
        out =y+ diff_P_x*(sigma+y)/(sigma+torch.kron(sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda')))
        return out

class SoftmaxConstraints(nn.Module):
    """softmax"""
    def __init__(self, upsampling_factor, exp_factor=1):
        super(SoftmaxConstraints, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
    def forward(self, y, lr):
        y = torch.exp(y)
        sum_y = self.pool(y)
        out = y*torch.kron(lr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
