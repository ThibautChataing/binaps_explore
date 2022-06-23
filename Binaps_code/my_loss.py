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
# The weightedXor class is a custom loss function that takes in the output of the neural network, the
# target, and the weight vector. It then calculates the difference between the output and the target,
# and multiplies the difference by the weight vector. It then adds the weight decay term to the loss
class weightedXor(nn.Module):

    def __init__(self, weight, weight_decay, device_gpu):
        """
        The function takes in the sparsity of the data and the decay rate as inputs and initializes the
        class
        
        :param weight: the sparsity of the data
        :param weight_decay: the decay rate of the weights
        :param device_gpu: the GPU device to use
        """
        super(weightedXor, self).__init__()
        ## sparsity of data
        self.weight = weight
        ## decay rate
        self.weight_decay = weight_decay
        print("Data Sparsity:")
        print(self.weight)

    def forward(self, output, target, w):
        """
        The function takes in the output of the network, the target, and the weight vector. It then
        calculates the difference between the output and the target, and then adds a weight decay term to
        the loss
        
        :param output: the output of the network
        :param target: the target output of the network
        :param w: the weight of the loss function
        :return: The difference between the output and target, weighted by the weight and the weight decay.
        """

        relu = nn.ReLU()

        # ND: relu((output - target)) == relu(output - target)
        #  relu((output - target)) get the diff between output and target cropping when target > output ( = -1)
        # --> 1 if output wrong (o=1 and t=0)
        # --> 0 (0 or -1) if output true or output wrong but = 0 (t=1)
        # -----> loss emphasize when FP over FN
        #  tensor.sum(1) do the sum of the tensor over features (columns)
        #  tensor.mul(weight)  multiply the tensor by weight = sparcity of the data
        #  tensor.mean() get the mean of the 1D vector (instead of sum to be independant of data size)
        diff = relu((output - target)).sum(1).mul(self.weight).mean() + relu((target - output)).sum(1).mul(1-self.weight).mean()

        diff += self.weight_decay*(((w - 1/target.size()[1])).sum(1).clamp(min=1).pow(2).sum())
        #   In practice self.weight_decay == 0 (be default and not used in paper) I assume the formule
        #   is wrong because of the clamp(min=1) should be clamp(max=1)
        return diff



# It takes the output of the network, the target, and the weight matrix, and returns the mean squared
# error between the output and the target, plus the weight decay term
class xor(nn.Module):

    def __init__(self, weight_decay, device_gpu):
        """       
        :param weight_decay: The weight decay parameter
        :param device_gpu: the device to use for training
        """
        super(xor, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, output, target, w):
        """
        It takes the difference between the output and the target, squares it, sums it, and averages it.
        Then it adds a penalty term to the loss function
        
        :param output: the output of the network
        :param target: the target matrix
        :param w: the weight matrix
        :return: The mean squared error between the output and the target, plus the sum of the squared
        weights minus the sum of the squared inverse of the target size.
        """
        diff = (output - target).pow(2).sum(1).mean()

        # set minimum of weight to 0, to avoid penalizing too harshly for large matrices
        diff += (w - 1/target.size()[1]).pow(2).sum()*self.weight_decay

        return diff
