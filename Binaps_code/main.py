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

import dataLoader as mydl
import my_layers as myla
import my_loss as mylo
import network as mynet


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
    args = parser.parse_args()
    now = datetime.datetime.today().isoformat(sep='T', timespec='seconds')
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.thread_num)

    device_cpu = torch.device("cpu")
    if not torch.cuda.is_available():
        device_gpu = device_cpu
        print("WARNING: Running purely on CPU. Slow.")
    else:
        device_gpu = torch.device("cuda")

    model, weights, train_data = mynet.learn(args.input, args.lr, args.gamma, args.weight_decay, args.epochs, args.hidden_dim, args.train_set_size, args.batch_size, args.test_batch_size, args.log_interval, device_cpu, device_gpu)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"_{now}_ternary_net.pt"))

    with torch.no_grad():

        print("\n\n\nPatterns:\n")
        with open(os.path.join(args.output_dir, args.input[:-4] + f"_{now}.binaps.patterns"),'w') as patF:
            for hn in myla.BinarizeTensorThresh(weights, .2):
                pat = torch.squeeze(hn.nonzero())
                supp_full = (train_data.matmul(hn.cpu()) == hn.sum().cpu()).sum().cpu().numpy()
                supp_half = (train_data.matmul(hn.cpu()) >= hn.sum().cpu()/2).sum().cpu().numpy()
                if hn.sum() >= 2:
                    print(pat.cpu().numpy(), "(", supp_full, "/", supp_half, ")")
                    print(pat.cpu().numpy(), file=patF)


if __name__ == '__main__':
    main()