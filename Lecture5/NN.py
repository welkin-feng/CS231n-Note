#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  NN.py

"""
from datetime import datetime

import numpy as np
import time
import copy
from scipy.misc import imread, imsave

from Lecture5.module.ReLU import ReLU
from Lecture5.module.module import module
from Lecture5.module.softmax import Softmax

__author__ = 'Welkin'
__date__ = '2017/8/16 02:51'


class NN(object):
    def __init__(self, input, in_num, hidden_nums, out_num, output, step_size, lam):
        self.input = input
        self.in_num = in_num
        self.hidden_nums = hidden_nums
        self.out_num = out_num
        self.output = output
        self.step_size = step_size
        self.lam = lam

        self.hidden_layer = ReLU(name = 'hidden_layer', in_num = in_num, out_num = hidden_num, x = input,
                                 step_size = step_size, lam = lam)
        self.output_layer = module(name = 'output_layer', in_num = hidden_num, out_num = out_num,
                                   x = self.hidden_layer.output, step_size = step_size, lam = lam)
        # self.output_layer = Softmax(name = 'output_layer', in_num = hidden_num, out_num = out_num,
        #                             x = self.hidden_layer.output, step_size = step_size, lam = lam)

    def set(self, hidden_w, hidden_b, out_w, out_b):
        self.hidden_layer.W = np.copy(hidden_w)
        self.hidden_layer.b = np.copy(hidden_b)
        self.output_layer.W = np.copy(out_w)
        self.output_layer.b = np.copy(out_b)

    @property
    def loss(self):
        # data_loss = 0.5 * np.sum(self.gradient ** 2)
        data_loss = self.softmax_loss()
        return (data_loss + self.lam * self.hidden_layer.L2 + self.lam * self.output_layer.L2)

    @property
    def gradient(self):
        # data_gradient = self.output_layer.output - self.output
        data_gradient = self.dsoftmax_loss()
        return data_gradient

    def softmax_loss(self):
        y = self.output
        num = y.shape[1]
        x = self.output_layer.output
        scores = x - np.max(x, axis = 0)
        self.softmax = np.exp(scores) / np.sum(np.exp(scores), axis = 0)
        self.probability = self.softmax[y, range(num)]
        # - ln(x1) - ln(x2)
        return np.sum(- np.log(self.probability)) / num

    def dsoftmax_loss(self):
        y = self.output
        num = y.shape[1]
        dsoftmax = np.copy(self.softmax)
        dsoftmax[y, range(num)] -= 1  # dL/doutput_layer.output
        dsoftmax /= num
        return dsoftmax

    def eval_numerical_gradient(self, layer):
        weights = layer.W
        fw = self.forward()  # 在原点计算函数值
        grad = np.zeros(weights.shape)
        h = 0.00001

        it = np.nditer(weights, flags = ['multi_index'], op_flags = ['readwrite'])
        while not it.finished:
            # 计算 w+h 处的函数值
            iw = it.multi_index
            old_value = weights[iw]
            weights[iw] = old_value + h  # 增加h
            fwh = self.forward()  # 计算f(w + h)
            weights[iw] = old_value  # 存到前一个值中 (非常重要)

            # 计算偏导数
            grad[iw] = (fwh - fw) / h  # 坡度
            it.iternext()  # 到下个维度

        return grad

    def forward(self, data = None):
        if data is not None:
            self.hidden_layer.X = data[0]
            self.output = data[1]
        self.output_layer.X = self.hidden_layer.forward()
        self.output_layer.forward()
        return self.loss

    def save_weights(nn):
        np.save(str(datetime.now()) + "_hidden_W.npy", nn.hidden_layer.W)
        np.save(str(datetime.now()) + "_hidden_b.npy", nn.hidden_layer.b)
        np.save(str(datetime.now()) + "_out_W.npy", nn.output_layer.W)
        np.save(str(datetime.now()) + "_out_b.npy", nn.output_layer.b)

    def load_weights(npyfile):
        return np.load(npyfile)


if __name__ == '__main__':
    from load_data import load_CIFAR10, sample_training_data

    X_train, Y_train, X_test, Y_test = load_CIFAR10('../data/cifar10/')  # 3072 x 50000

    mean_train = np.mean(X_train, axis = 1).reshape((-1, 1))
    std_train = np.std(X_train, axis = 1).reshape((-1, 1))

    X_train -= mean_train  # 0中心化：均值减法
    X_train /= np.std(X_train, axis = 1).reshape((-1, 1))  # 归一化：每个维度都除以其标准差
    X_test -= mean_train
    X_test /= np.std(X_train, axis = 1).reshape((-1, 1))

    data_batch = sample_training_data([X_train, Y_train], 256)  # 256个数据, 3072 x 256
    in_num = 3072
    hidden_num = 100
    out_num = 10
    nn = NN(data_batch[0], in_num, hidden_num, out_num, data_batch[1], 10 ** -3, 0.01)
    loss_original = nn.loss
    print("original loss: %f" % (loss_original,))
    min_loss = loss_original
    for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, ]:
        step_size = 10 ** step_size_log
        n = copy.deepcopy(nn)
        n.hidden_layer.step_size = step_size
        n.output_layer.step_size = step_size

        dh = n.output_layer.backward(n.gradient)
        n.hidden_layer.backward(dh)

        loss_new = n.forward()
        print("step_size: %.10f, loss: %f" % (step_size, loss_new,))
        if loss_new < min_loss:
            min_loss = loss_new
            best_step_size = step_size

    print("best step size %.10f" % (best_step_size,))
    nn.hidden_layer.step_size = best_step_size
    nn.output_layer.step_size = best_step_size
    time.sleep(1)

    for i in range(1000):
        batch_size = 256
        data_batch = sample_training_data([X_train, Y_train], batch_size)
        nn.forward(data_batch)
        h_weights_grad = nn.eval_numerical_gradient(nn.hidden_layer)
        o_weights_grad = nn.eval_numerical_gradient(nn.output_layer)
        loss = nn.loss
        print("i: %d , loss: %f, correct ratio: %f" % (i, loss, np.sum(nn.probability > 0.5) / batch_size,))
        if (loss < 0.00001):
            break

        dh = nn.output_layer.backward(nn.gradient)
        nn.hidden_layer.backward(dh)

        # time.sleep(0.1)
    print("loss: %f" % (loss,))
    data_test = [X_test, Y_test]
    nn.forward(data_batch)
    loss = nn.loss
    print("test loss: %f, correct ratio: %f" % (loss, np.sum(nn.probability > 0.5) / batch_size,))
    # test loss: 2.840334, correct ratio: 0.222656
    select = input("save weights and bias ? (y or n)")
    if select is "y":
        np.save(str(datetime.now()) + "_hidden_W.npy", nn.hidden_layer.W)
        np.save(str(datetime.now()) + "_hidden_b.npy", nn.hidden_layer.b)
        np.save(str(datetime.now()) + "_out_W.npy", nn.output_layer.W)
        np.save(str(datetime.now()) + "_out_b.npy", nn.output_layer.b)
        print("save successfully")
