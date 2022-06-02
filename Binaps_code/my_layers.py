## BinaPs -- Binary Pattern Networks
## Copyright 2021 Jonas Fischer <fischer@mpi-inf.mpg.de>
## Permission of copy is granted under the GNU General Public License v3.0

import torch
import torch.nn as nn
import math
import torch.distributions as torchdist
from torch.autograd import Variable
from torch.autograd import Function

from torch.nn import init



def BinarizeTensorThresh(tens, thresh):
    """
    > This function takes a tensor and a threshold value and returns a tensor with the same shape as the
    input tensor, but with all values above the threshold set to 1 and all values below the threshold
    set to 0
    
    :param tens: the tensor to binarize
    :param thresh: the threshold value for binarization
    :return: A tensor with the same shape as the input tensor, but with values of 1 where the input
    tensor is greater than the threshold, and 0 otherwise.
    """
    t = (tens > thresh).float()
    return t

def BinarizeTensorStoch(tens, device_gpu):
    """
    It takes a tensor and returns a tensor of the same shape, where each element is either 0 or 1, with
    a probability of 0.5
    
    :param tens: the tensor to be binarized
    :param device_gpu: the device you're using (e.g. torch.device("cuda:0"))
    :return: A tensor with the same shape as the input tensor, but with values of either 0 or 1.
    """
    t = tens.bernoulli()
    return t

def SteppyBias(tensor, is_dec):
    """
    If the tensor is positive, set it to zero. If it's negative, set it to -1
    
    :param tensor: the tensor to be stepped
    :param is_dec: whether the tensor is from the decoder or not
    :return: a tensor that is either 0 or -1.
    """
    if is_dec:
        t = tensor.clamp(min=0,max=0)
    else:
        t = tensor.clamp(max=-1)
    t = t.int().float()
    return t

# The forward pass of the class takes in the input, bias, and biasNI and returns the result of the
# binarization function. The backward pass takes in the gradient of the output and returns the
# gradient of the input and the gradient of the bias
class BinarizeFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, biasNI):
        """
        `forward` is the function that is called when you call the module
        
        :param ctx: a context object that can be used to stash information for backward computation
        :param input: the input tensor
        :param bias: the bias vector
        :param biasNI: the bias that is added to the input before clamping and rounding
        :return: The output of the forward pass.
        """
        ctx.save_for_backward(input, bias, biasNI)
        res = (input+biasNI).clamp(0,1).round()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        If the bias is negative, then the gradient of the input is the gradient of the output multiplied by
        the gradient of the activation function. 
        
        If the bias is positive, then the gradient of the input is the gradient of the output. 
        
        The gradient of the bias is the gradient of the output multiplied by the gradient of the activation
        function, but only if the node was active. 
        
        The gradient of the bias is the gradient of the output, but only if the node was not active.
        
        :param ctx: context object that can be used to stash information for backward computation
        :param grad_output: The gradient of the loss with respect to the output of this layer
        :return: The gradient of the input and the gradient of the bias
        """
        input, bias, biasNI = ctx.saved_tensors
        if (bias[0] < 0):
            grad_input = (input+biasNI).clamp(0,1).round()*grad_output
        else:
            grad_input = grad_output
        # Throw out negative gradient to bias if node was not active
        # (Do not punish for things it did not do)
        grad_bias = (1-(input+biasNI).clamp(0,1).round())*grad_output.clamp(max=0).sum(0) + (input+biasNI).clamp(0,1).round()*grad_output.sum(0)
        return grad_input, grad_bias, None


# It implements a binary activation function that is differentiable and can be used in a neural
# network
class BinaryActivation(nn.Module):

    def __init__(self, size, device_gpu):
        """
        The BinaryActivation class is a subclass of the nn.Module class. It has a constructor that takes in
        two arguments: size and device_gpu. The constructor initializes the superclass (nn.Module) and
        creates a bias parameter that is initialized to -1.0. The bias parameter is a tensor of size size
        and is on the device device_gpu. The bias parameter is also set to require gradients. The biasNI
        parameter is a clone of the bias parameter that is detached from the graph and is on the device
        device_gpu.
        
        :param size: the size of the input tensor
        :param device_gpu: the device you're using (e.g. torch.device("cuda:0"))
        """
        super(BinaryActivation, self).__init__()
        self.bias = nn.Parameter(-torch.ones(size, device=device_gpu), requires_grad=True)
        self.biasNI = self.bias.clone().detach().to(device_gpu)

    def forward(self, input, is_dec):
        """
        `BinarizeFunction.apply(input, self.bias, self.biasNI)`
        
        This function is called in the forward pass of the `Binarize` module. It takes in the input, the
        bias, and the biasNI. It then returns the output of the forward pass
        
        :param input: the input tensor
        :param is_dec: whether or not we're in the decoder
        :return: The output of the forward function is the output of the BinarizeFunction.apply function.
        """
        with torch.no_grad():
            self.biasNI = SteppyBias(self.bias, is_dec)
        return BinarizeFunction.apply(input, self.bias, self.biasNI)

    def clipBias(self):
        """
        It clips the bias of the model to be between -1 and 1
        """
        with torch.no_grad():
            self.bias.clamp_(max=-1)

    def noBias(self):
        """
        > The function `noBias` sets the bias of the layer to zero
        """
        with torch.no_grad():
            self.bias.clamp_(min=0, max=0)



# `BinaryLinearFunction` is a class that implements the forward and backward passes of a linear layer
# with binary weights
class BinaryLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, weightB):
        """
        `forward` takes in the input, weight, and weightB, and returns the output.
        
        :param ctx: a context object that can be used to stash information for backward computation
        :param input: the input tensor
        :param weight: the weight matrix of the layer
        :param weightB: the weight matrix for the bias
        :return: The output of the forward pass.
        """
        ctx.save_for_backward(input, weight, weightB)
        out = input.matmul(weightB.t())
        return out


    @staticmethod
    def backward(ctx, grad_output):
        """
        `backward` is the function that is called when the gradient of the output of the `Linear` function
        is calculated. 
        
        The `backward` function takes in the gradient of the output of the `Linear` function, and returns
        the gradient of the input of the `Linear` function. 
        
        The `backward` function also takes in the input and weight tensors that were saved in the `forward`
        function. 
        
        The `backward` function calculates the gradient of the input and weight tensors, and returns them. 
        
        The `backward` function is called by PyTorch when the gradient of the output of the `Linear`
        function is calculated. 
        
        The `backward` function is called by PyTorch when the gradient of the output of the `Linear`
        function is calculated. 
        
        The `backward` function is called by PyTorch when the gradient
        
        :param ctx: the context object that was returned by the forward function. It can be used to stash
        information for backward computation. You can cache arbitrary objects for use in the backward pass
        using the ctx.save_for_backward method
        :param grad_output: the gradient of the loss with respect to the output of the layer
        :return: The gradients of the input and weight tensors.
        """
        input, weight, weightB = ctx.saved_tensors

        grad_input = grad_output.matmul(weight)

        grad_weight = grad_output.t().matmul(input)

        return grad_input, grad_weight, None



# It's a linear layer that uses the binary linear function to compute the output
class BinarizedLinearModule(nn.Module):

    def __init__(self, inum, onum, threshold, data_sparsity, is_dec, enc_weights, enc_weightsB, bInit, device_cpu, device_gpu):
        """
        The function takes in the number of input and output nodes, the threshold for binarization, the
        sparsity of the data, whether the layer is a decoder or not, the weights of the encoder, the
        binarized weights of the encoder, whether the weights are initialized or not, and the CPU and GPU
        devices
        
        :param inum: number of input neurons
        :param onum: number of output neurons
        :param threshold: the threshold to use for binarization
        :param data_sparsity: the sparsity of the data. This is used to calculate the threshold for
        binarization
        :param is_dec: whether the layer is a decoder layer or not
        :param enc_weights: the weights of the encoder
        :param enc_weightsB: the binarized weights of the encoder
        :param bInit: whether to initialize the weights to be binarized or not
        :param device_cpu: the CPU device
        :param device_gpu: the GPU device
        """
        super(BinarizedLinearModule, self).__init__()
        self.inum = inum
        self.onum = onum
        self.devGPU = device_gpu
        self.devCPU = device_cpu
        # threshold to use for binarization
        self.threshold = threshold

        self.is_dec = is_dec
        if is_dec:
            self.weight = nn.Parameter(torch.zeros(inum, onum, device = device_gpu))
            self.weightB = torch.zeros(inum, onum, device = device_gpu)
            self.weight.data = enc_weights.transpose(0,1)
            self.weightB.data = enc_weightsB.transpose(0,1)
        else:
            self.weight = nn.Parameter(enc_weights)
            # binarized weight matrix
            self.weightB = torch.zeros(onum, inum, device = device_gpu)

    def forward(self, input):
        """
        `BinaryLinearFunction.apply(input, self.weight, self.weightB)`
        
        This is the function that performs the forward pass of the binary linear layer. It takes in the
        input, the weight, and the binarized weight. It then performs the forward pass of the binary linear
        layer
        
        :param input: the input tensor
        :return: The output of the forward pass of the binary linear layer.
        """
        # these tensors are not tracked for gradient computation
        with torch.no_grad():
            if not(self.is_dec):
                self.weightB.data.copy_(BinarizeTensorStoch(self.weight, self.devGPU))
        out = BinaryLinearFunction.apply(input, self.weight, self.weightB)
        return out

    def extra_repr(self):
        """
        `extra_repr` is a function that returns a string that is used to represent the object
        :return: The output of the forward pass of the network.
        """
        return 'input_features={}, output_features={}, threshold={}'.format(
            self.inum, self.onum, self.threshold
        )

    def clipWeights(self, mini=-1, maxi=1):
        """
        > The function clips the weights of the layer to be between a minimum and maximum value
        
        :param mini: minimum value of the weights
        :param maxi: the maximum value of the weights, defaults to 1 (optional)
        """
        with torch.no_grad():

            # weights are clamped at 1/input_dim corresponding to - on expectation - one incoming weight per node being active. Counteracting dying neurons
            self.weight.clamp_(1/(self.inum), maxi)
