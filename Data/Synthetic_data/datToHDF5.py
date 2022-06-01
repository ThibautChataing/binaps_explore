from __future__ import print_function
import argparse
import os
import pandas as pd
import numpy as np
import math

import h5py

def readDatFile(dat_file):

  ncol = -1
  nrow = 0
  with open(dat_file) as datF:
    # read .dat format line by line
    l = datF.readline()
    while l:
      # drop newline
      l = l[:-1]
      # skip empty entry
      if not l:
        continue
      # get indices as array
      sl = l.split(" ")
      sl = [int(i) for i in sl if i]
      maxi = max(sl)
      if (ncol < maxi):
        ncol = maxi
      nrow += 1
      l = datF.readline()
  data = np.zeros((nrow, ncol), dtype=np.single)
  with open(dat_file) as datF:
    # read .dat format line by line
    l = datF.readline()
    rIdx = 0
    while l:
      # drop newline
      l = l[:-1]
      # skip empty entry
      if not l:
        continue
      # get indices as array
      sl = l.split(" ")
      idxs = np.array([int(i) for i in sl if i])
      idxs -= 1
      # assign active cells
      data[rIdx, idxs] = np.repeat(1, idxs.shape[0])
      rIdx += 1
      l = datF.readline()

  return data


def main():
    parser = argparse.ArgumentParser(description='.dat to .hdf5 parser')
    parser.add_argument('-i','--input', required=True,
                        help='Input .dat file to convert')
    args = parser.parse_args()

    d = np.asarray(readDatFile(args.input))

    if args.input[-4:] != '.dat':
        print("Warning: Input does not have .dat file ending!")
        print(args.input[-4:])

    hf = h5py.File(args.input[:-4] + '.h5', 'w')
    hf.create_dataset('data', data=d)
    hf.close()


if __name__ == '__main__':
    main()