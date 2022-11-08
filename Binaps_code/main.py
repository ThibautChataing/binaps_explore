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
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(module)s in %(funcName)s at %(lineno)dl : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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


def main(argu=None):
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
    args = parser.parse_args(argu)
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
                    
                    # obtiens le support maximum
                    supp_max = (train_data.matmul(hn.cpu()).div(hn.sum().cpu()) * 100 ).max().cpu().numpy()

                    if supp_full > 0 or supp_half > 0:
                        logging.info(f"({supp_full}/{supp_half}/{supp_max}), {pat.cpu().numpy()}")

                    json.dump(dict(supp_full=supp_full.tolist(), supp_half=supp_half.tolist(), supp_max=supp_max.tolist(), pat=pat.cpu().tolist()), patF)
                    patF.write('\n')
        logging.info(f"Pattern saved to {file_pat}")

    logging.info("Finished.")


if __name__ == '__main__':

    # data is going from one to max
    # pattern is goind from 0 to max-1
    #argument = r"-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_only_versicolor.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10"
    #argument = r"-i C:\Users\Thibaut\Documents\These\code\binaps_explore\Data\accidents.dat -o ./output --epochs 20 --batch_size 1000 --test_batch_size 100"
    
    
#     exp = ['-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_only_setosa.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_only_versicolor.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_only_virginica.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_set_ver.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_set_vir.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_setosa.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\Iris_vir_ver.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 24 --test_batch_size 10',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\credit_card.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 1000 --test_batch_size 100',
#  '-i C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\input\\synthetic_data_1000_1000_10_0.001_INTER_2022-08-29T16h20m33s.dat -o C:\\Users\\Thibaut\\Documents\\These\\code\\experiments\\output\\binaps --epochs 20 --batch_size 100 --test_batch_size 10']
#     for e in exp:
#         main(e.split())
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

    argument = r"-i C:\Users\Thibaut\Documents\These\code\binaps_explore\github_explore\data_scrap\v2\github_binary_event_filter_part.dat -o C:\Users\Thibaut\Documents\These\code\binaps_explore\github_explore\data_scrap\ --epochs 150 --hidden_dim 1000 --log_interval 100"
    #argument = rf'-i C:\Users\Thibaut\Documents\These\code\experiments\synth_simplest\data\synthetic_data_100000_1000_100_0.0_NO_INTER_2022-10-27T14h38m21s.dat -o C:\Users\Thibaut\Documents\These\code\experiments\synth_simplest\output --lr 0.02 --epochs 50 --batch_size 64 --test_batch_size 64 --log_interval 100' 
    argument = r"-i C:\Users\Thibaut\Documents\These\code\experiments\exp2\input\Iris_setosa_v2.dat -o C:\Users\Thibaut\Documents\These\code\experiments\exp2\output\ --epochs 150 --batch_size 25 --test_batch_size 25"
    #argument = r"-i C:\Users\Thibaut\Documents\These\code\experiments\exp2\input\Iris_only_setosa_v2.dat -o C:\Users\Thibaut\Documents\These\code\experiments\exp2\output\ --epochs 150 --batch_size 2 --test_batch_size 25"
    
    main(argument.split())
