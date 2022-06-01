## BinaPs -- Binary Pattern Networks
## Copyright 2021 Jonas Fischer <fischer@mpi-inf.mpg.de>
## Permission of copy is granted under the GNU General Public License v3.0

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

from scipy.stats import binom

import my_layers as myla




## weigh the different cases (positive vs negative) differently
## based on the data sparsity
class weightedXor(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu):
        super(weightedXor, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w):

        relu = nn.ReLU()
        diff = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()
        diff += self.weight_decay*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2).sum())

        return diff



class xor(nn.Module):

    def __init__(self, weight_decay, device_gpu):
        super(xor, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, output, target, w):
        diff = (output - target).pow(2).sum(1).mean()

        # set minimum of weight to 0, to avoid penalizing too harshly for large matrices
        diff += (w - 1/target.size()[1]).pow(2).sum()*self.weight_decay

        return diff
