#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

Project Name:   CS231n-Note 

File Name:  unpackage.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/8/11 13:13'


def unpickle(file):
    """按字节读取file中的数据，并以字典形式返回"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict


def load_CIFAR10(file):
    """
    输入 CIFAR-10 路径, 返回 CIFAR-10 中的数据，格式为:
    训练集 Xtr: 3072 x 50000, Ytr: 1 x 50000,
    测试集 Xte: 3072 x 10000, Yte: 1 x 10000
    """
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in range(5):
        d = unpickle(file + 'data_batch_' + str(i + 1))
        Xtr = np.append(Xtr, d[b'data'])
        ytr += d[b'labels']

    d = unpickle(file + 'test_batch')
    Xte = d[b'data']
    yte = d[b'labels']
    return Xtr.reshape((50000, 3072)).T, np.array(ytr).reshape((-1, 1)).T, \
           Xte.astype(float).reshape((-1, 3072)).T, np.array(yte).reshape((-1, 1)).T


def sample_training_data(data, num):
    """
    从 训练样本 中随机取 num 条数据返回，返回数据为 data[0]: 3072维, data[1]: 1维 (CIFAR-10为例)

    :param data:
        [X_train, Y_train]。X_train 是3072维 x 50000条数据， Y_train 是1维 x 50000条标签，要求每一列是一项数据
    :param num:
        要随机选取多少项数据，应该大于 1
    :return:
        返回随机选取的 num 条数据和对应的标签
    """
    if num <= 1:
        raise ValueError("'num' should bigger than 1")
    N = data[1].shape[1]  # 3072 x 50000
    import random
    l = random.sample(range(N), num)
    return data[0][:, l], data[1][:, l]


if __name__ == '__main__':
    xtr, ytr, xte, yte = load_CIFAR10("data/cifar10/")
    print()
