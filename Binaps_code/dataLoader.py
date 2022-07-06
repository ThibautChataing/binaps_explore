## BinaPs -- Binary Pattern Networks
## Copyright 2021 Jonas Fischer <fischer@mpi-inf.mpg.de>
## Permission of copy is granted under the GNU General Public License v3.0

import math

import numpy as np
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
from tqdm import tqdm

# TODO add a way to specify the wanted number of row and col in the data, we can add a file wit meta data

def readDatFile(dat_file):
    """
    Read .dat file
    Normal text file where:
        each line is a row of data for training
        each value is the index of a 1 in the associated sparse matrice

    Return : sparse matrice, sparsity
    """
    logging.critical("Read file has been tempered with to make it work with github data. Look for TODO in this file to correct")
    ncol = -1
    nrow = 0
    count_1 = 0

    # Get number of row and column to recreate a sparse matrice
    with open(dat_file) as datF:
        # read .dat format line by line
        l = datF.readline()
        while l:
            # drop newline
            l = l[:-1]
            if l == "":
                continue
            if l[-1] == " ":
                l = l[:-1]
            # get indices as array
            sl = l.split(" ")
            sl = [int(i) for i in sl]
            maxi = max(sl)
            count_1 += len(sl)
            if (ncol < maxi):
                ncol = maxi
            nrow += 1
            l = datF.readline()

    # TODO take out that and make it work
    ncol = 361675 + 1
    sparsity = count_1 / (nrow * ncol)

    logging.info(f"Info of data is row={nrow}, features={ncol}, sparsity={sparsity}")
    data = np.zeros((nrow, ncol), dtype=np.single)

    with open(dat_file) as datF:
        # read .dat format line by line
        l = datF.readline()
        rIdx = 0

        pbar = tqdm(total=nrow + 1, desc="Adding 1 in the sparse matrix")

        while l:
            # drop newline
            # TODO better way
            l = l[:-1]
            if l == "":
                continue
            if l[-1] == " ":
                l = l[:-1]
            # get indices as array
            sl = l.split(" ")
            idxs = np.array(sl, dtype=int)
            # TODO they shift index to obtain a zero, don't do it and find the min in the first part of this algo if it's zero, don't do the shift
            #idxs -= 1
            # assign active cells
            data[rIdx, idxs] = np.repeat(1, idxs.shape[0])
            rIdx += 1
            l = datF.readline()
            pbar.update()
        pbar.close()

    return data, sparsity


## Construct a dataset from a regular csv file
class RegDataset(Dataset):

    def __init__(self, csv_file, train_proportion, is_training):
        data = pd.read_csv(csv_file, sep=",", index_col=0)
        self.data = np.asarray(data)
        ran = np.arange(0, math.ceil(train_proportion * self.data.shape[0]))
        trainmin = self.data[ran, 1].min()
        trainmax = self.data[ran, 1].max()
        if not (is_training):
            ran = np.arange(math.ceil(train_proportion * self.data.shape[0]), self.data.shape[0])
        self.data = self.data[ran, :]
        self.data[:, 1] = self.data[:, 1] - trainmin / (trainmax - trainmin)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, 0], self.data[index, 1]


## Construct a dataset from a .dat file
class DatDataset(Dataset):

    def __init__(self, dat_file, train_proportion, is_training, device_cpu):
        logging.info("Init dataset")
        data, sparsity = readDatFile(dat_file)
        # Next line commented because data is already an ndarray
        self.data = np.asarray(data)

        # change because inefficient
        # self.sparsity = np.count_nonzero(self.data) / np.prod(self.data.shape)
        self.sparsity = sparsity
        logging.info(f"Data sparity = {self.sparsity}")

        logging.critical("Bad bad way to do it !")
        if is_training:
            ran = np.arange(0, math.ceil(train_proportion * self.data.shape[0]))
        else:
            ran = np.arange(math.ceil(train_proportion * self.data.shape[0]), self.data.shape[0])

        logging.info("Input data will be converted to tensor for torch")
        logging.critical("Train test split is BAD ! Make a good random use of it !")

        # old way to do it, that break the code
        #self.data = torch.from_numpy(self.data[ran, :])  # , device=device_cpu)
        row_max = math.ceil(train_proportion * self.data.shape[0])
        self.data = torch.from_numpy(self.data[:row_max, :])  # , device=device_cpu)

        logging.info("Input data converted to tensor for torch")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.data[index, :]

    def matmul(self, other):
        return self.data.matmul(other)

    def nrow(self):
        return self.data.shape[0]

    def ncol(self):
        return self.data.shape[1]

    def getSparsity(self):
        return self.sparsity
