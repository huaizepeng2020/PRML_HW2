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

from matplotlib import pyplot as plt
# plt.rcParams['font.sans-serif']=['FangSong']
# plt.rcParams['axes.unicode_minus'] = False
import matplotlib

matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D


def plotdata(file, name):
    a = np.load(file)  # MNIST
    print('--------------------{}---------------------'.format(name))

    acc_GRF = a['acc_GRF']
    acc1_GRF = a['acc1_GRF']
    acc_LLGC = a['acc_LLGC']
    acc1_LLGC = a['acc1_LLGC']
    sigma = a['sigma']
    lamda = a['lamda']
    bl = a['bl']

    # ---------------GRF acc-sigma-bl------------------------
    acc1_GRF1 = np.mean(acc1_GRF, axis=2)
    # psigma,pbl = np.meshgrid(sigma,bl)
    pbl, psigma = np.meshgrid(bl, sigma)

    fig1 = plt.figure()
    ax = Axes3D(fig1)
    ax.plot_surface(psigma, pbl, acc1_GRF1.T, cmap=plt.cm.hot)  # 渐变颜色
    # ax.contourf(psigma, pbl, acc1_GRF1.T,
    #             zdir='z',  # 使用数据方向
    #             offset=-1,  # 填充投影轮廓位置
    #             cmap=plt.cm.hot)
    ax.set_zlim(acc1_GRF1.min(), acc1_GRF1.max())
    ax.set_ylabel('training_rate')  # 坐标轴
    ax.set_xlabel('sigma')
    ax.set_zlabel('acc')
    plt.savefig(str('{}1' + ".png").format(name))

    best_acc = []
    best_sigma = []
    for i, j in enumerate(bl):
        best_acc.append(acc1_GRF1[i].max())
        b = np.where(acc1_GRF1[i] == acc1_GRF1[i].max())
        best_sigma.append(psigma[:, i][b[0][0]])
    print('--------------------GRF---------------------')
    print('traning_rate:', bl)
    print('best_acc:', best_acc)
    print('best_sigma:', best_sigma)

    b = np.where(sigma == best_sigma[0])
    pbl1 = acc1_GRF1[:, b[0][0]]
    fig1 = plt.figure()
    # ax.plot(bl, pbl1)  # 渐变颜色
    plt.plot(bl, pbl1)  # 渐变颜色
    plt.xlabel('training_rate')
    plt.ylabel('acc')
    tt = 'sigma={} acc'.format(best_sigma[0])
    plt.title(tt)
    plt.savefig(str('{}2' + ".png").format(name))

    # ---------------LLGC acc-sigma-bl------------------------
    plamda, psigma = np.meshgrid(lamda, sigma)
    p_acc1_LLGC = []
    for i, j in enumerate(bl):
        p_acc1_LLGC.append(acc1_LLGC[i])

    for i, j in enumerate(bl):
        fig1 = plt.figure()
        ax = Axes3D(fig1)
        ax.plot_surface(psigma, plamda, p_acc1_LLGC[i], cmap=plt.cm.hot)  # 渐变颜色
        ax.contourf(psigma, plamda, p_acc1_LLGC[i],
                    zdir='z',  # 使用数据方向
                    offset=p_acc1_LLGC[i].min(),  # 填充投影轮廓位置
                    cmap=plt.cm.hot)
        ax.set_zlim(p_acc1_LLGC[i].min(), p_acc1_LLGC[i].max())
        ax.set_xlabel('sigma')  # 坐标轴
        ax.set_ylabel('lamda')
        ax.set_zlabel('acc')
        tt = 'training_rate={} acc'.format(j)
        plt.savefig(str('{}_L1_{}' + ".png").format(name,j))

    best_acc = []
    best_sigma = []
    best_lamda = []
    for i, j in enumerate(bl):
        best_acc.append(p_acc1_LLGC[i].max())
        b = np.where(p_acc1_LLGC[i] == p_acc1_LLGC[i].max())
        best_sigma.append(psigma[b[0][0], b[1][0]])
        best_lamda.append(plamda[b[0][0], b[1][0]])
    print('--------------------LLGC---------------------')
    print('traning_rate:', bl)
    print('best_acc:', best_acc)
    print('best_sigma:', best_sigma)
    print('best_sigma:', best_lamda)

    psigma1 = p_acc1_LLGC[-1][:, np.where(lamda == best_lamda[0])[0][0]]
    fig1 = plt.figure()
    plt.plot(sigma, psigma1)
    plt.xlabel('sigma')
    plt.ylabel('acc')
    tt = 'training_rate=0.4 lamda={} acc'.format(best_lamda[0])
    plt.title(tt)
    plt.savefig(str('{}_L2' + ".png").format(name))

    plamda1 = p_acc1_LLGC[-1][np.where(sigma == best_sigma[0])[0][0], :]
    # plamda1=p_acc1_LLGC[-1][np.where(sigma==sigma[int(len(sigma)/2)])[0][0],:]
    fig1 = plt.figure()
    plt.plot(lamda, plamda1)
    plt.xlabel('lamda')
    plt.ylabel('acc')
    tt = 'training_rate=0.4 sigma={} acc'.format(best_sigma[0])
    # tt='training_rate=0.4 sigma={} acc'.format(sigma[int(len(sigma)/2)])
    plt.title(tt)
    plt.savefig(str('{}_L3' + ".png").format(name))

    aa = 1


file, name = 're.npz', 'MNIST'
file1, name1 = 're1.npz', 'Vowel'
file2, name2 = 're2.npz', 'Nosar'

plotdata(file, name)
plotdata(file1, name1)
plotdata(file2, name2)
