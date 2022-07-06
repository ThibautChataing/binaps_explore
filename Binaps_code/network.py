## BinaPs -- Binary Pattern Networks
## Copyright 2021 Jonas Fischer <fischer@mpi-inf.mpg.de>
## Permission of copy is granted under the GNU General Public License v3.0

import torch
import torch.nn as nn  # basic bulding blocks for graphs
import torch.nn.functional as F  # convolution functions
import torch.optim as optim  # package implementing various optimization algorithms
from torch.nn import init  # package to init
from torch.optim.lr_scheduler import MultiStepLR  # Adapt learning rate : Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
import logging
import numpy as np
import math

import gc # garbage collector

import dataLoader as mydl
import my_layers as myla
import my_loss as mylo


def initWeights(w, data):
    """
    > This function initializes the weights of the network to zero
    
    :param w: The weight tensor to be initialized
    :param data: The data that is passed to the function
    """
    init.constant_(w, 0)


# It's a neural network with one hidden layer, with the hidden layer having the same number of neurons
# as the input layer
class Net(nn.Module):
    """
    nn.Module : Base class for all neural network modules
    """
    def __init__(self, init_weights, init_bias, data_sparsity, device_cpu, device_gpu):
        """
        It initializes the network with the given weights and biases, and sets the data sparsity
        
        :param init_weights: initial weights for the model, tensor 2D {n <= features (size of hidden layer),
        features}
        :param init_bias: initial bias for the model, Tensor of -1 with the dim of init_weights.size[0]
        :param data_sparsity: data_non_zeros/prod_data.shape
        :param device_cpu: the CPU device
        :param device_gpu: the GPU device
        """
        # TODO : Why sparsity ?
        # TODO : vérify dim of init_weights
        super(Net, self).__init__()
        input_dim = init_weights.size()[1]
        hidden_dim = init_weights.size()[0]
        self.fc0_enc = myla.BinarizedLinearModule(input_dim, hidden_dim, .5, data_sparsity, False, init_weights, None, init_bias, device_cpu, device_gpu)
        self.fc3_dec = myla.BinarizedLinearModule(hidden_dim, input_dim, .5, data_sparsity, True, self.fc0_enc.weight.data, self.fc0_enc.weightB.data, None, device_cpu, device_gpu)
        self.act0 = myla.BinaryActivation(hidden_dim, device_gpu)
        self.act3 = myla.BinaryActivation(input_dim, device_gpu)
        self.clipWeights()


    def forward(self, x):
        """
        > The function takes in an input, passes it through a fully connected layer, applies an activation
        function, passes it through another fully connected layer, and applies another activation function
        
        :param x: the input to the network
        :return: The output of the network.
        """
        x = self.fc0_enc(x)
        x = self.act0(x, False)
        x = self.fc3_dec(x)
        output = self.act3(x, True)
        return output

    def clipWeights(self, mini=-1, maxi=1):
        """
        It clips the weights of the first layer of the encoder to be between -1 and 1, and it sets the bias
        of the first activation layer to be 0
        
        :param mini: minimum value for the weights
        :param maxi: the maximum value of the weights, defaults to 1 (optional)
        """
        self.fc0_enc.clipWeights(mini, maxi)
        self.act0.clipBias()
        self.act3.noBias()


def train(model, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval):
    """
    > We iterate over the training data, and for each batch we zero the gradients, compute the output of
    the network, compute the loss, backpropagate the loss, and update the weights
    
    :param model: the model we're training
    :param device_cpu: the CPU device
    :param device_gpu: the device to use for training
    :param train_loader: the training data
    :param optimizer: the optimizer used to train the model
    :param lossFun: the loss function to use
    :param epoch: the number of times we go through the entire dataset
    :param log_interval: how often to print the training loss
    :return: The loss of the model
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device_gpu)
        optimizer.zero_grad()
        output = model(data)
        itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
        loss = lossFun(output, data, next(iter(itEW)))  # data equal target, == loss.forward()
        loss.backward()
        optimizer.step()
        model.clipWeights()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return


def test(model, device_cpu, device_gpu, test_loader, lossFun):
    """
    It runs the model on the test set, and prints the average loss and accuracy
    
    :param model: the model we're training
    :param device_cpu: the CPU device
    :param device_gpu: the GPU device to use
    :param test_loader: the test data loader
    :param lossFun: the loss function to use
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device_gpu)
            output = model(data)
            itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
            test_loss += lossFun(output, data, next(iter(itEW)))
            correct += (output.ne(data.data.view_as(output)).sum(1) == 0).sum()

    _, target = next(iter(test_loader))
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def learn(input, lr, gamma, weight_decay, epochs, hidden_dim, train_set_size, batch_size, test_batch_size, log_interval, device_cpu, device_gpu):
    """
    > This function takes in a bunch of parameters, loads the data, initializes the model, optimizer,
    and loss function, and then trains the model
    
    :param input: the name of the file containing the data
    :param lr: learning rate
    :param gamma: the learning rate decay factor
    :param weight_decay: the weight decay parameter for the Adam optimizer
    :param epochs: number of epochs to train for
    :param hidden_dim: the number of hidden units in the network. If -1, then it's set to the number of
    columns in the input data
    :param train_set_size: the number of rows in the training set
    :param batch_size: the number of samples to use in each batch
    :param test_batch_size: the number of samples to use for testing
    :param log_interval: how often to print out the loss
    :param device_cpu: the device to use for the CPU
    :param device_gpu: the GPU device to use
    :return: The model, the weights, and the dataset.
    """


    kwargs = {}
    logging.debug(f"Start pull data from {input}")
    logging.critical("We pull two times all the data to extract train then test instead of splitting after one pull")
    trainDS = mydl.DatDataset(input, train_set_size, True, device_cpu)
    train_loader = torch.utils.data.DataLoader(trainDS,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mydl.DatDataset(input, train_set_size, False, device_cpu),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    logging.debug("Pull data ok")
    if hidden_dim == -1:
        hidden_dim = trainDS.ncol()

    logging.debug("Init weight")

    # Breaking and toooooo long (maybe there is a reason to use torch.zeros instead of numpy but i don't know it)
    new_weights = torch.zeros(hidden_dim, trainDS.ncol(), device=device_gpu)
    #new_weights = np.zeros((hidden_dim, trainDS.ncol())) + 1/(trainDS.ncol())
    #new_weights = torch.from_numpy(new_weights)

    # commented because unused, because force to init to zero but it's already zero
    initWeights(new_weights, trainDS.data)
    # commented because take too much power, see upper for the init
    new_weights.clamp_(1/(trainDS.ncol()), 1)  # init weights with 1/nbr_of_feature

    logging.debug("Init bias")
    # idem as before
    bInit = torch.zeros(hidden_dim, device=device_gpu)
    init.constant_(bInit, -1)
    #bInit = np.zeros(hidden_dim) -1

    logging.debug("Init model")
    model = Net(new_weights, bInit, trainDS.getSparsity(), device_cpu, device_gpu).to(device_gpu)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lossFun = mylo.weightedXor(trainDS.getSparsity(), weight_decay, device_gpu)

    scheduler = MultiStepLR(optimizer, [5,7], gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval)

        test(model, device_cpu, device_gpu, test_loader, lossFun)
        scheduler.step()

    return model, new_weights, trainDS