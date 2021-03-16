from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import autograd

import time
import _pickle as cPickle

import urllib
import matplotlib.pyplot as plt

import os
import sys
import codecs
import re
import numpy as np
import data
from model import NERModel
from train import create_and_train_model

import argparse
plt.rcParams['figure.dpi'] = 80
plt.style.use('seaborn-pastel')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Language Model - Named Entity Recognition')
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
    parser.add_argument('--model_dir', type=str, default='./model/', help='model directory')
    parser.add_argument('--save_path', type=str, default='model.pt', help='path to the model')
    parser.add_argument('--tag', type=str, default='BIOES', help='tag scheme: BIO or BIOES')
    parser.add_argument('--lower', type=bool, default=True, help='lower case or not')
    parser.add_argument('--zeros', type=bool, default=True, help='zero digits or not')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--start', type=str, default='<START>', help='start tag')
    parser.add_argument('--stop', type=str, default='<STOP>', help='stop tag')
    parser.add_argument('--seed', type=str, default=40, help='random seed')
    parser.add_argument('--embedding_dir', type=str, default='./data/embedding/', help='embedding directory')
    parser.add_argument('--embedding_dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=200, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--crf', action='store_true', help='CRF layer')
    parser.add_argument('--pretrain', type=bool, default=True, help='use pretrained word embedding')
    parser.add_argument('--char_encoder', type=str, default='LSTM', help='LSTM, CNN')
    parser.add_argument('--word_encoder', type=str, default='LSTM', help='LSTM, CNN, CNN2, CNN3, CNN_DILATED')
    parser.add_argument('--lr', type=float, default=0.015, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clip')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='decay rate')
    parser.add_argument('--plot_interval', type=int, default=2000, help='plot every # steps')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # =============== Load device ===============
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    # =============== Load data ===============
    cleaner = data.Cleaner(args)
    raw_train_data, raw_dev_data, raw_test_data = cleaner.clean()
    dataset = data.Dataset(raw_train_data, raw_dev_data, raw_test_data, args)
    word2idx, tag2idx, char2idx = dataset.word_to_id, dataset.tag_to_id, dataset.char_to_id
    train_data, dev_data, test_data = dataset.train_data, dataset.dev_data, dataset.test_data
    print("{} / {} / {} sentences in train / dev / test.".format(len(train_data), len(dev_data), len(test_data)))

    # =============== Build the model ===============
    model = NERModel(word2idx, tag2idx, char2idx, args)
    if args.cuda:
        model.to(device)
    print('Model Initialized!!, n_params = {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # =============== Train the model ===============
    all_f1, all_acc = create_and_train_model(model, train_data, dev_data, test_data, tag2idx, args)
    print('f1 = {}'.format(all_f1))
    print('acc = {}'.format(all_acc))
