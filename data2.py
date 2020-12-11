# -*- coding: utf-8 -*-
# coding=utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import random
import scipy.sparse
import pickle
import torch
import pandas as pd

seed = 20200803
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    vowel = 'vowel-context.data'
    a = pd.read_csv(vowel)
    a2 = []
    for i, j in enumerate(a.values):
        a1 = a.values[i][0].split()
        a2.append([float(i) for i in a1])
    a2 = np.array(a2)

    train_data=a2[:,3:13]
    train_labels=a2[:,-1]

    # -------------生成指定大小的样本,并分成训练集和测试集--------------
    bl = [0.2, 0.3, 0.4]
    acc_GRF = [None] * len(bl)
    acc1_GRF = [None] * len(bl)
    acc_LLGC = [None] * len(bl)
    acc1_LLGC = [None] * len(bl)
    for iii, iiii in enumerate(bl):
        test_idx, train_idx = train_test_split(np.arange(len(train_data)), test_size=int(iiii * len(train_data)),
                                               random_state=seed)
        train_in, test_in = train_data[train_idx], train_data[test_idx]
        train_out, test_out = train_labels[train_idx], train_labels[test_idx]
        train_data = train_data[train_idx.tolist() + test_idx.tolist()]
        train_labels = train_labels[train_idx.tolist() + test_idx.tolist()]
        # --------------GRF模型-----------------
        sigma0 = np.array(np.array(list(range(1, 50)))/5).tolist()
        lamda0 = np.array(np.array(list(range(1, 10)))/10).tolist()
        acc_GRF[iii] = [None]*len(sigma0)
        acc1_GRF[iii] = [None]*len(sigma0)
        acc_LLGC[iii] = [None]*len(sigma0)
        acc1_LLGC[iii] = [None]*len(sigma0)
        for kkk,_ in enumerate(sigma0):
            acc_GRF[iii][kkk] = [None] * len(lamda0)
            acc1_GRF[iii][kkk] = [None] * len(lamda0)
            acc_LLGC[iii][kkk] = [None] * len(lamda0)
            acc1_LLGC[iii][kkk] = [None] * len(lamda0)
        # for sigma, lamda in zip(sigma0, lamda0):
        for idx,sigma in enumerate(sigma0):
            for jdx,lamda in enumerate(lamda0):
                # sigma=2**8 # 这里我们认为所有维度上的高斯函数的超参都一样
                num_train = len(train_idx)
                num_test = len(test_idx)
                num_all = len(train_data)

                W = np.zeros(shape=(num_all, num_all))
                for i in range(num_all):
                    for j in range(num_all):
                        wij = np.exp(np.sum(-(train_data[i] - train_data[j]) ** 2 / (sigma ** 2)))
                        W[i, j] = wij
                    # print('处理完', i)
                D = np.sum(W, axis=1)
                D = np.diag(D)

                W_ll = W[0:num_train, 0:num_train]
                W_lu = W[0:num_train, num_train:]
                W_ul = W_lu.T
                W_uu = W[num_train:, num_train:]

                D_ll = D[0:num_train, 0:num_train]
                D_uu = D[num_train:, num_train:]

                f_l = train_out
                f_u = np.dot(np.dot(np.linalg.inv(D_uu - W_uu), W_ul), f_l)
                f_u1 = np.dot(np.linalg.inv(D_uu - W_uu), W_ul)

                f_u_T = test_out
                f_u = np.around(f_u)
                re = f_u - f_u_T
                acc = len(np.where(re == 0)[0])
                acc1 = len(np.where(re == 0)[0]) / len(test_out)
                print('---------------超参数sigma：{}超参数lamda：{}训练集比例：{}-------------------'.format(sigma,lamda, iiii))
                print('【MGF】分类正确个数：{}/{}----------正确率：{}'.format(acc, len(test_out), acc1))
                acc_GRF[iii][idx][jdx]=acc
                acc1_GRF[iii][idx][jdx]=acc1
                # --------------LLGC模型-----------------
                S = np.dot(np.dot(np.linalg.inv(D ** 0.5), W), np.linalg.inv(D ** 0.5))

                alpha = 1 / (1 + lamda)
                beta = 1 - alpha
                Y = np.zeros(shape=(num_all, 11))
                for i in range(num_train):
                    Y[i, int(train_out[i])] = 1
                Y_y = beta * np.dot(np.linalg.inv((np.identity(num_all) - alpha * S)), Y)
                f_u = Y_y[num_train:]
                f_u = np.argmax(f_u, axis=1)

                f_u_T = test_out
                re = f_u - f_u_T
                acc = len(np.where(re == 0)[0])
                acc1 = len(np.where(re == 0)[0]) / len(test_out)
                print('【LLGC】分类正确个数：{}/{}----------正确率：{}'.format(acc, len(test_out), acc1))
                acc_LLGC[iii][idx][jdx]=acc
                acc1_LLGC[iii][idx][jdx]=acc1

    np.savez('re1.npz', acc_GRF=acc_GRF, acc1_GRF=acc1_GRF,
             acc_LLGC=acc_LLGC, acc1_LLGC=acc1_LLGC, sigma=sigma0, lamda=lamda0, bl=bl)


