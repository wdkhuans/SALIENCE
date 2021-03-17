# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:38:35 2020

@author: 82045
"""

from __future__ import print_function
import argparse
import torch
from solver import Solver
import warnings
warnings.filterwarnings("ignore")
#
# Training settings
parser = argparse.ArgumentParser(description='PyTorch SALIENCE Implementation')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                    help='how many epochs')
parser.add_argument('--test_user', type=int, default=0, metavar='N',
                    help='user for test')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--ex_num', type=int, default=1, metavar='N',
                    help='the number of experiments')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

def main():
    solver = Solver(args)
    solver.train(args.max_epoch)

if __name__ == '__main__':
    main()
