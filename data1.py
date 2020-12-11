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

vowel='vowel-context.data'
a=pd.read_csv(vowel)
a2=[]
for i,j in enumerate(a.values):
    a1=a.values[i][0].split()
    a2.append([float(i) for i in a1])
a2=np.array(a2)

# 训练集文件
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(
        image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # print(images[i])
        offset += struct.calcsize(fmt_image)
    #        plt.imshow(images[i],'gray')
    #        plt.pause(0.00001)
    #        plt.show()
    # plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


if __name__ == '__main__':
    train_images = load_train_images()
    train_images = train_images.reshape(len(train_images), -1)
    train_labels = load_train_labels()
    train_labels = train_labels.reshape(len(train_labels), -1)
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    # -------------生成指定大小的样本,并分成训练集和测试集--------------
    all_data_idx = []
    for i in range(10):
        a = np.where(train_labels == i)[0]
        all_data_idx += a[0:200].tolist()
    train_data = train_images[all_data_idx]
    train_labels = train_labels[all_data_idx]

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
        sigma0 = list(range(2 ** 8, 2 ** 10, 2 ** 5))  # 这里我们认为所有维度上的高斯函数的超参都一样
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
                Y = np.zeros(shape=(num_all, 10))
                for i in range(num_train):
                    Y[i, int(train_out[i])] = 1
                Y_y = beta * np.dot(np.linalg.inv((np.identity(num_all) - alpha * S)), Y)
                f_u = Y_y[num_train:]
                f_u = np.argmax(f_u, axis=1)
                f_u = np.expand_dims(f_u, axis=1)

                f_u_T = test_out
                re = f_u - f_u_T
                acc = len(np.where(re == 0)[0])
                acc1 = len(np.where(re == 0)[0]) / len(test_out)
                print('【LLGC】分类正确个数：{}/{}----------正确率：{}'.format(acc, len(test_out), acc1))
                acc_LLGC[iii][idx][jdx]=acc
                acc1_LLGC[iii][idx][jdx]=acc1

    np.savez('re.npz', acc_GRF=acc_GRF, acc1_GRF=acc1_GRF,
             acc_LLGC=acc_LLGC, acc1_LLGC=acc1_LLGC, sigma=sigma0, lamda=lamda0, bl=bl)


