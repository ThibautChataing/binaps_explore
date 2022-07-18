## BinaPs -- Binary Pattern Networks
## Copyright 2021 Jonas Fischer <fischer@mpi-inf.mpg.de>
## Permission of copy is granted under the GNU General Public License v3.0

import torch
import torch.nn as nn  # basic bulding blocks for graphs
import torch.nn.functional as F  # convolution functions
import torch.optim as optim  # package implementing various optimization algorithms
from torch.nn import init  # package to init
from torch.optim.lr_scheduler import MultiStepLR  # Adapt learning rate : Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

import numpy as np
import math

import gc # garbage collector

import dataLoader as mydl
import my_layers as myla
import my_loss as mylo
import logging

def initWeights(w, data):
    """
    Fills the input Tensor with the value 0
    :param w: Tensor input
    :param data: Not used
    :return:
    """
    init.constant_(w, 0)


class Net(nn.Module):
    """
    nn.Module : Base class for all neural network modules
    """
    def __init__(self, init_weights, init_bias, data_sparsity, device_cpu, device_gpu):
        """

        :param init_weights: init weights for the model, tensor 2D {n <= features (size of hidden layer), features}
        :param init_bias: init bias for the model, Tensor of -1 with the dim of init_weights.size[0]
        :param data_sparsity: data_non_zeros/prod_data.shape
        :param device_cpu: def for torch of the cpu
        :param device_gpu: def for torch of the gpu and = device_cpu if there is no gpu available
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
        x = self.fc0_enc(x)
        x = self.act0(x, False)
        x = self.fc3_dec(x)
        output = self.act3(x, True)
        return output

    def clipWeights(self, mini=-1, maxi=1):
        self.fc0_enc.clipWeights(mini, maxi)
        self.act0.clipBias()
        self.act3.noBias()


def train(model, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device_gpu)
        optimizer.zero_grad()
        output = model(data)
        itEW = [par for name, par in model.named_parameters() if name.endswith("enc.weight")]
        loss = lossFun(output, data, next(iter(itEW)))
        loss.backward()
        optimizer.step()
        model.clipWeights()

        if batch_idx % log_interval == 0:
            logging.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                         f"({int(100 * batch_idx / len(train_loader))}%)]\tLoss: "
                         f"{round(100 * batch_idx / len(train_loader),3)}")

    return


def test(model, device_cpu, device_gpu, test_loader, lossFun):
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
    logging.info(f"Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} "
                 f"({int(100*correct/len(test_loader.dataset))}%)")


def learn(input, lr, gamma, weight_decay, epochs, hidden_dim, train_set_size, batch_size, test_batch_size, log_interval, device_cpu, device_gpu):

    logging.debug("Init dataset")
    kwargs = {}
    trainDS = mydl.DatDataset(input, train_set_size, True, device_cpu)
    train_loader = torch.utils.data.DataLoader(trainDS,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mydl.DatDataset(input, train_set_size, False, device_cpu),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    if hidden_dim == -1:
        hidden_dim = trainDS.ncol()

    logging.info(f"Init weight line={hidden_dim}, col={trainDS.ncol()}")
    new_weights = torch.zeros(hidden_dim, trainDS.ncol(), device=device_gpu)
    initWeights(new_weights, trainDS.data)
    new_weights.clamp_(1/(trainDS.ncol()), 1)

    logging.info(f"Init bias line={hidden_dim}")
    bInit = torch.zeros(hidden_dim, device=device_gpu)
    init.constant_(bInit, -1)

    model = Net(new_weights, bInit, trainDS.getSparsity(), device_cpu, device_gpu).to(device_gpu)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lossFun = mylo.weightedXor(trainDS.getSparsity(), weight_decay, device_gpu)

    logging.info(f"Start training for{epochs}")
    scheduler = MultiStepLR(optimizer, [5,7], gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device_cpu, device_gpu, train_loader, optimizer, lossFun, epoch, log_interval)

        test(model, device_cpu, device_gpu, test_loader, lossFun)
        scheduler.step()
    logging.info("Training done")
    return model, new_weights, trainDS