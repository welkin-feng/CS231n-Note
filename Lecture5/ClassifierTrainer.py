#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  ClassifierTrainer.py

"""
import datetime

import numpy as np
import copy
import time

from load_data import sample_training_data
from Lecture5.NN2 import NN

__author__ = 'Welkin'
__date__ = '2017/8/23 11:29'


class ClassifierTrainer(object):
    def train(self, X_train, Y_train, X_validation, Y_validation, node_num,
              num_epochs, reg, update = 'sgd',
              learning_rate_decay = 1, learning_rate = None,
              sample_batch = False, batch_size = None, model = None):

        # 训练数据集，批量训练
        if sample_batch and batch_size is not None:
            train_data = sample_training_data([X_train, Y_train], batch_size)  # 256个数据, 3072 x 256
            val_data = sample_training_data([X_validation, Y_validation], batch_size // 2)
        elif not sample_batch:
            train_data = [X_train, Y_train]
            val_data = [X_validation, Y_validation]
        else:
            raise Exception("missing 'batch_size")
        train_data_size = train_data[0].shape[1]
        val_data_size = val_data[0].shape[1]

        in_num = node_num[0]  # 3072
        hidden_num = node_num[1]  # 100
        out_num = node_num[2]  # 10

        if model is not None:
            learning_rate = model['lr']

        nn = NN(train_data[0], in_num, hidden_num, out_num, train_data[1], step_size = learning_rate, lam = reg)

        if model is not None:
            nn.set(model['W1'], model['b1'], model['W2'], model['b2'])

        if learning_rate is None:
            # 选取合适步长
            loss_original = nn.loss
            print("original loss: %f" % (loss_original,))
            min_loss = loss_original
            step_size_log_list = [-10 + learning_rate_decay * x for x in range(10 // learning_rate_decay + 1)]

            for step_size_log in step_size_log_list:
                step_size = 10 ** step_size_log
                n = copy.deepcopy(nn)
                n.hidden_layer.step_size = step_size
                n.output_layer.step_size = step_size

                dh = n.output_layer.backward(n.gradient)
                n.hidden_layer.backward(dh)

                loss_new = n.forward()[0]
                print("step_size: %s, loss: %f" % (format(step_size, '.2e'), loss_new,))
                if loss_new < min_loss:
                    min_loss = loss_new
                    best_step_size = step_size
            print("best step size %s" % format(best_step_size, '.2e'))
            # 设置步长
            nn.hidden_layer.step_size = best_step_size
            nn.output_layer.step_size = best_step_size

        best_val_accuracy = 0
        for i in range(num_epochs):
            if sample_batch:
                train_data = sample_training_data([X_train, Y_train], batch_size)  # 256个数据, 3072 x 256
                val_data = sample_training_data([X_validation, Y_validation], batch_size // 2)
            else:
                train_data = None

            val_loss, val_probability = nn.validation(val_data)
            loss, tr_probability = nn.forward(train_data)

            # 按置信度求正确率
            # tr_accuracy = np.sum(tr_probability > 0.9) / train_data_size
            # val_accuracy = np.sum(val_probability > 0.9) / val_data_size

            # 按 softmax 结果的均值求正确率
            tr_accuracy = np.sum(tr_probability) / train_data_size
            val_accuracy = np.sum(val_probability) / val_data_size
            lr = learning_rate if learning_rate is not None else best_step_size
            print("epoch  %d / %d, loss: %f, train: %f, validation: %f, lr: %s"
                  % (i + 1, num_epochs, loss, tr_accuracy, val_accuracy, format(lr, '.2e')))
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            dh = nn.output_layer.backward(nn.gradient)
            nn.hidden_layer.backward(dh)
            # time.sleep(0.1)
        print("finished optimization, best validation accuracy: %f" % best_val_accuracy)
        best_model = {}
        best_model['W1'] = nn.hidden_layer.W
        best_model['b1'] = nn.hidden_layer.b
        best_model['W2'] = nn.output_layer.W
        best_model['b2'] = nn.output_layer.b
        best_model['lr'] = lr
        return best_model


def save_weights(model):
    np.save(str(datetime.now()) + "_hidden_W.npy", model['W1'])
    np.save(str(datetime.now()) + "_hidden_b.npy", model['b1'])
    np.save(str(datetime.now()) + "_out_W.npy", model['W2'])
    np.save(str(datetime.now()) + "_out_b.npy", model['b2'])
    np.save(str(datetime.now()) + "_lr.npy", model['lr'])


if __name__ == '__main__':
    from load_data import load_CIFAR10, sample_training_data

    X_train, Y_train, X_test, Y_test = load_CIFAR10('../data/cifar10/')  # 3072 x 50000
    mean_train = np.mean(X_train, axis = 1).reshape((-1, 1))
    std_train = np.std(X_train, axis = 1).reshape((-1, 1))

    X_train -= mean_train  # 0中心化：均值减法
    X_train /= std_train  # 归一化：每个维度都除以其标准差
    X_test -= mean_train
    X_test /= std_train
    np.linalg.norm()

    trainer = ClassifierTrainer()
    node_num = [3072, 100, 10]
    x_tiny = X_train[:, :20]
    y_tiny = Y_train[:, :20]
    first_model = trainer.train(x_tiny, y_tiny, x_tiny, y_tiny, node_num,
                                num_epochs = 150, reg = 0.01, update = 'sgd',
                                learning_rate_decay = 1, learning_rate = None,
                                sample_batch = False, batch_size = None)

    best_model = trainer.train(X_train[:, :40000], Y_train[:, :40000],
                               X_train[:, 40000:], Y_train[:, 40000:], node_num,
                               num_epochs = 300, reg = 0.01, update = 'sgd',
                               learning_rate_decay = 1, learning_rate = None,
                               sample_batch = True, batch_size = 256, model = first_model)

    select = input("save weights and bias ? (y or n)")
    if select is "y":
        save_weights(best_model)
        print("save successfully")
