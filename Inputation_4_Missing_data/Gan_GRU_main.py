# written by LorneM 2020.1.2

import os,sys
sys.path.append('GAN_GRU')

import numpy as np
import pandas as pd
import random
import copy
import argparse
import tensorflow as tf
from ReadTrainData import readTrainData
from ReadTestData import readTestData
from WGAN_GRUI import WGAN

# 设置seed
random.seed(1)

# parse arguments
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--gen-length', type=int, default=96)
parser.add_argument('--impute-iter', type=int, default=400)
parser.add_argument('--pretrain-epoch', type=int, default=1) #不进行PretrainG
parser.add_argument('--lr', type=float, default=0.001)
# lr 0.001的时候 pretrain_loss降的很快，4个epoch就行了
parser.add_argument('--n-inputs', type=int, default=6)
parser.add_argument('--n-hidden-units', type=int, default=64)
parser.add_argument('--n-classes', type=int, default=2)
parser.add_argument('--z-dim', type=int, default=16)

# 0 false 1 true
parser.add_argument('--isBatch-normal', type=int, default=1)
parser.add_argument('--disc-iters', type=int, default=8)
args = parser.parse_args()

if args.isBatch_normal == 0:
    args.isBatch_normal = False
if args.isBatch_normal == 1:
    args.isBatch_normal = True


# make the max step length of two datasett the same
epochs = [1]
g_loss_lambdas = [0.15]
beta1s = [0.5]
for beta1 in beta1s:
    for e in epochs:
        for g_l in g_loss_lambdas:
            args.epoch = e
            args.beta1 = beta1
            args.g_loss_lambda = g_l
            tf.reset_default_graph()
            dt_train = readTrainData(time_steps = 12)
            dt_test = readTestData(standard_scalar = dt_train.standard_scalar,time_steps = 12)
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                gan = WGAN(sess,
                                     args=args,
                                     datasets=dt_train,
                                     )

                # build graph
                gan.build_model()

                # show network architecture
                # show_all_variables()

                # launch the graph in a session
                gan.train()
                print(" [*] Training finished!")

                gan.imputation(dt_test)

                print(gan.loss_pre)

                print(" [*] Test dataset Imputation finished!")

