#!/usr/bin/env python
# coding: utf-8

# import train data and train label
# run binapsC 
# save pattern and weight
# run binaps on all, class 0, class 1
# save pattern and weight

# File used with Dockerfile_exp

import sys
import os
import argparse
from pathlib import Path



def set_local():
    imp = r"C:\Users\Thibaut\Documents\These\code"
    if imp not in sys.path:
        sys.path.append(imp)

    imp = r"C:\Users\Thibaut\Documents\These\code\binaps_explore\Binaps_code"
    if imp not in sys.path:
        sys.path.append(imp)

    print(sys.path)

    from Binaps_code import main as binapsC


def check_dir(output_dir, dir_name):
    output_dir = os.path.join(output_dir, 'binaps', dir_name)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    return output_dir


def main(args=None):
    import model as binaps
    parser = argparse.ArgumentParser(description='Binary Pattern Network implementation')

    parser.add_argument('-i', '--input', required=True,
                        help='Root input dir')

    parser.add_argument('-o', '--output', required=True,
                        help='output dir')

    parser.add_argument('-d', '--data', required=True,
                        help='data_data')
    args = parser.parse_args(args)

    #output_binapsC = r"C:\Users\Thibaut\Documents\These\code\binaps_contrastive\experiment\output_binapsC"
    basic =   ' --epochs 20000 --save_model --log_interval 1000'

    output = check_dir(args.output, args.data)
    print(f"Output = {output}")

    if args.data == 'adult':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 131 --test_batch_size 127' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 131 --test_batch_size 127' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 131 --test_batch_size 127' + basic 

    elif args.data == 'australian':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 

    elif args.data == 'chess':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 

    elif args.data == "breast-cancer-wisconsin":
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 

    elif args.data == 'german':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 

    elif args.data == 'haberman':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 41 --test_batch_size 31' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 41 --test_batch_size 31' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 41 --test_batch_size 31' + basic 

    elif args.data == 'magic04':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 130 --test_batch_size 128' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 130 --test_batch_size 128' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 130 --test_batch_size 128' + basic 

    elif args.data == 'mushroom':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 128 --test_batch_size 128' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 

    elif args.data == 'sonar':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 66 --test_batch_size 34' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 66 --test_batch_size 34' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 66 --test_batch_size 34' + basic 

    elif args.data == 'tic-tac-toe':
        arg = rf'-i {args.input}_train.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg1 = rf'-i {args.input}_train_class0.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 
        arg2 = rf'-i {args.input}_train_class1.dat -o {output} --batch_size 64 --test_batch_size 64' + basic 

    binaps.main(arg.split(' '))
    binaps.main(arg1.split(' '))
    binaps.main(arg2.split(' '))

if __name__ == "__main__":
    main()