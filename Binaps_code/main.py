from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


import os
import datetime
import numpy as np
import math
import json
import logging
import sys

import dataLoader as mydl
import my_layers as myla
import my_loss as mylo
import network as mynet


def set_logger(file_name=None):

    # create logger for prd_ci
    log = logging.getLogger()
    log.setLevel(level=logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(levelname)s] %(module)s in %(funcName)s at %(lineno)dl : %(message)s')

    if file_name:
        # create file handler for logger.
        fh = logging.FileHandler(file_name + '.log', mode='w')
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
    # reate console handler for logger.
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if file_name:
        log.addHandler(fh)

    log.addHandler(ch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Binary Pattern Network implementation')
    parser.add_argument('-i','--input', required=True,
                        help='Input file to use for training and testing')
    parser.add_argument('--train_set_size', type=float, default=.9,
                        help='proportion of data to be used for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay for L2 norm (default 0)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save the current Model')
    parser.add_argument('-o', '--output_dir', default='.',
                        help='output directory to save things')
    parser.add_argument('--hidden_dim', type=int, default=-1,
                        help='size for the hidden layer (default: #features)')
    parser.add_argument('--thread_num', type=int, default=16,
                        help='number of threads to use (default: 16)')
    parser.add_argument('--no_gpu', action='store_true', default=False,
                        help='Force to use only cpu')
    args = parser.parse_args()
    now =datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    log_file = os.path.join(args.output_dir, os.path.basename(args.input[:-4]).replace(':', '_') + f"_{now}_")
    set_logger(log_file)

    logging.debug(f"Args : {args}")
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.thread_num)

    device_cpu = torch.device("cpu")
    if not torch.cuda.is_available() or args.no_gpu:
        device_gpu = device_cpu
        logging.warning("Working on CPU, SLOW !")
    else:
        device_gpu = torch.device("cuda")
        logging.info("Working on GPU")

    logging.debug("Start")
    model, weights, train_data = mynet.learn(args.input, args.lr, args.gamma, args.weight_decay, args.epochs, args.hidden_dim, args.train_set_size, args.batch_size, args.test_batch_size, args.log_interval, device_cpu, device_gpu)

    if args.save_model:
        file = os.path.join(args.output_dir, os.path.basename(args.input[:-4]) + f"_{now}_ternary_net.pt")
        torch.save(model.state_dict(), file)
        logging.info(f"Model saved to {file}")

    with torch.no_grad():
        file_pat = os.path.join(args.output_dir, os.path.basename(args.input[:-4]) + f"_{now}.binaps.patterns")
        #logging.info("-"*10 + "Patterns:" + "_"*10)
        with open(file_pat, 'w') as patF:
            for hn in myla.BinarizeTensorThresh(weights, .2):   # We take the weight matrice and binarize with the threshold of 0.2
                # On parcours toute les lignes de la matrice de poids après binarization
                pat = torch.squeeze(hn.nonzero())

                if hn.sum() >= 2:
                    #  Compte du nombre de ligne dans lesquelles on trouve le pattern
                    #  train_data.matmul(hn.cpu()) -> vector de int où count sur j de train_data[i,j] = hn[i]
                    supp_full = (train_data.matmul(hn.cpu()) == hn.sum().cpu()).sum().cpu().numpy()

                    # compte du numbre de ligne dans lesquelles on trouve au moins la moitié du pattern
                    supp_half = (train_data.matmul(hn.cpu()) >= hn.sum().cpu() / 2).sum().cpu().numpy()
                    #logging.info(f"{pat.cpu().numpy()}, ({supp_full}/{supp_half})")
                    json.dump(dict(supp_full=supp_full.tolist(), supp_half=supp_half.tolist(), pat=pat.cpu().tolist()), patF, indent=2)
        logging.info(f"Pattern saved to {file_pat}")

    logging.info("Finished.")


if __name__ == '__main__':
    main()
