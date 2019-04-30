#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  NN.py

"""
import copy
import matplotlib.pyplot as plt

from Lecture6.layer import *
from Lecture6.util import *

__author__ = 'Welkin'
__date__ = '2017/11/20 13:41'


class NN(object):
    """
    用神经网络训练模型步骤：
    1. 构建神经网络框架，确定层数、每层节点数
    2. 确定每个节点的激活函数，以及输出层选用的分类器（SVM 或 softmax）
    3. 根据分类器确定损失函数，正则化函数
    4. 确定反向传播中参数的更新方法，SGD、动量、学习率退火等
    5. 编码实现前向传播、反向传播
    6. 对数据进行预处理：0 均值、归一化
    7. 对神经网络每层的权重进行初始化：针对不同激活函数有不同初始化方法
    8. 可以选择在每层的激活函数前加入BatchNormal层来代替归一化
    9. 用小批量数据进行梯度检查：将神经网络梯度和数值梯度进行比较，检查反向传播是否正确
    10. 检查学习过程：小批量数据能否过拟合，训练集和验证集准确率比较，权重更新比例检查
    11. 加入随机失活
    """

    def __init__(self, layer_num, data_axis):
        self.layer_num = layer_num
        # self.nn = [None for i in range(layer_num)]  # 储存每层神经网络, nn[0] 表示输入层，nn[-1]表示输出层
        # self.nn.insert(-1, None) # 以这种形式依次添加每层神经网络
        self.X_train = None  # 储存所有训练样本集的输入
        self.y_train = None  # 储存所有训练样本集的输出
        self.X_test = None
        self.y_test = None
        self.data_axis = data_axis  # 样本数据轴
        self.learn_rate = 0.0001
        self.X_mean = 0
        self.X_std = 0
        self._cur_batch_data = None
        self._cur_train_data = None
        self._cur_valid_data = None
        self.forward_data = None
        self._loss = 0  # 储存神经网络的损失
        self._data_loss = 0
        self._reg_loss = 0
        self._dloss = 0
        self._accuracy = 0
        '''
        ......
        其它超参数设置
        '''
        pass

    def load_examples(self, X, y, axis):
        self.X_train = X
        self.y_train = y
        self.data_axis = axis
        pass

    def load_test_data(self, X, y):
        self.X_test = X
        self.y_test = y
        pass

    def data_preprocessing(self):
        """
        数据预处理，先进行0均值处理，记录（训练）样本均值；再进行归一化处理，记录（训练）样本标准差
        """
        pass

    def weight_init(self):
        pass

    def forward(self, data):
        pass

    def backward(self):
        pass

    @property
    def output(self):
        return self.nn[-1].output

    @property
    def loss(self):
        self._loss = self.data_loss + self.reg_loss
        return self._loss

    @property
    def data_loss(self):
        return self._data_loss

    @property
    def reg_loss(self):
        return self._reg_loss

    @property
    def dloss(self):
        return self._dloss

    @property
    def accuracy(self):
        return self._accuracy

    def gradient_check(self):
        pass

    def numerical_gradient(self, layer_name):
        pass

    def test_learn_rate(self):
        pass

    def overfit_check(self, times):
        pass

    def train(self, times):
        pass

    def validation(self):
        pass

    def test(self):
        pass


class NN3(NN):
    def __init__(self, layer_num, data_axis = 1, learn_rate = 0.001, reg_lam = 0.01):
        super().__init__(layer_num, data_axis)
        self.learn_rate = learn_rate
        self.reg_lam = reg_lam
        '''
        self.nn[0] = input_layer(3072)
        self.nn[1] = L2_hidden_layer(3072, 500, f = relu, df = drelu, reg_lam = self.reg_lam)
        self.nn[2] = L2_hidden_layer(500, 10, f = softmax, df = dsoftmax, reg_lam = self.reg_lam)
        '''

        l1 = input_layer(3072)
        l2 = L2_hidden_layer(3072, 768, f = relu, df = drelu, reg_lam = self.reg_lam)
        l3 = L2_hidden_layer(768, 48, f = relu, df = drelu, reg_lam = self.reg_lam)
        l4 = L2_hidden_layer(48, 10, f = softmax, df = dsoftmax, reg_lam = self.reg_lam)
        self.nn = [l1, l2, l3, l4]

    def data_preprocessing(self):
        """
        数据预处理，先进行0均值处理，记录（训练）样本均值；再进行归一化处理，记录（训练）样本标准差
        """
        self.X_mean = np.mean(self.X_train, axis = self.data_axis).reshape((-1, 1))
        self.X_std = np.std(self.X_train, axis = self.data_axis).reshape((-1, 1))

        self.X_train -= self.X_mean
        self.X_train /= self.X_std

        self.X_test -= self.X_mean
        self.X_test /= self.X_std

    def weight_init(self):
        """
        初始化每层神经网络权值
        :return:
        """
        for i in self.nn[1:-1]:
            i.W = np.random.randn(i.number_this, i.number_before) / np.sqrt(i.number_before / 2)
        i = self.nn[-1]
        i.W = np.random.randn(i.number_this, i.number_before) / np.sqrt(i.number_before)

    def forward(self, data):
        """
        将当前训练数据集 self._cur_train_data 代入神经网络
        应该先设置当前训练数据
        """
        X, Y = data
        self.nn[0].forward(X)
        for i in range(1, self.layer_num):
            self.nn[i].forward(self.nn[i - 1].output)
        self.forward_data = [self.output, Y]
        return self.forward_data

    @property
    def output(self):
        return super().output

    @property
    def loss(self):
        return super().loss

    @property
    def data_loss(self):
        output, y_data = self.forward_data
        data_size = y_data.shape[-1]
        self._data_loss = np.mean(- np.log(output[y_data, range(data_size)]))
        return super().data_loss

    @property
    def reg_loss(self):
        _reg_loss = 0
        for i in self.nn:
            if hasattr(i, 'reg_loss'):
                _reg_loss += i.reg_loss()
        self._reg_loss = _reg_loss
        return super().reg_loss

    @property
    def dloss(self):
        y_train = self._cur_train_data[1]
        data_size = y_train.shape[-1]
        _dloss = np.zeros(self.output.shape)
        _dloss[y_train, range(data_size)] = - 1 / (data_size * self.output[y_train, range(data_size)])
        self._dloss = _dloss
        return super().dloss

    @property
    def accuracy(self):
        output, y_data = self.forward_data
        data_size = y_data.shape[-1]
        self._accuracy = np.mean(output[y_data, range(data_size)])
        return self._accuracy

    def backward(self):
        """
        根据当前梯度 self._dloss 更新每层神经网络的权值
        :return:
        """
        _dloss = self.dloss
        for i in self.nn[::-1]:
            _dloss = i.backward(_dloss, self.learn_rate)

    def gradient_check(self):
        self.sample_training_data([self.X_train, self.y_train], 256)
        self.forward(self._cur_train_data)
        print("loss: %f, data loss: %f, regular loss: %f, accuracy: %f" % (
            self.loss, self._data_loss, self._reg_loss, self.accuracy))

        # 从第 l 层开始的梯度检查
        begin_layer = 2
        ndw = [self.numerical_gradient(self.nn[i]) for i in range(begin_layer, self.layer_num)]
        self.backward()
        adw = [self.nn[i]._dW for i in range(begin_layer, self.layer_num)]

        relative_error = [np.mean(np.abs((ndw[i] - adw[i]) / (ndw[i] * (ndw[i] > adw[i]) +
                                                              adw[i] * (adw[i] >= ndw[i]))))
                          for i in range(ndw.__len__())]

        for i in range(relative_error.__len__()):
            print('layer %d, relative error: %.7f' % (begin_layer + i, relative_error[i]))

        '''
        # 最后一层梯度检查
        ndw = self.numerical_gradient(self.nn[-1])
        self.backward()
        adw = self.nn[-1]._dW
        relative_error = np.mean(np.abs((ndw - adw) / (ndw * (ndw > adw) +
                                                       adw * (adw >= ndw))))
        print('layer %d, relative error: %.7f' % (2, relative_error))
        '''

    def numerical_gradient(self, layer_name):
        """
        计算数值梯度，按照 f'(x) = lim (h->0) [ f(x+h) - f(x) ] / h
        """
        weights = layer_name.W
        self.forward(self._cur_train_data)  # 在原点计算函数值
        fw = self.loss
        grad = np.zeros(weights.shape)
        h = 10 ** -6

        it = np.nditer(weights, flags = ['multi_index'], op_flags = ['readwrite'])
        while not it.finished:
            # 计算 w+h 处的函数值
            iw = it.multi_index

            old_value = weights[iw]
            weights[iw] = old_value + h  # 增加h
            self.forward(self._cur_train_data)  # 计算f(w + h)
            fwh = self.loss
            # weights[iw] = old_value - h  # 减少h
            # self.forward(self._cur_train_data)  # 计算f(w - h)
            # fwh_ = self.loss
            weights[iw] = old_value  # 存到前一个值中 (非常重要)

            # 计算偏导数
            # grad[iw] = (fwh - fwh_) / (2 * h)  # 坡度
            grad[iw] = (fwh - fw) / h  # 坡度

            it.iternext()  # 到下个维度
        return grad

    def test_learn_rate(self):
        if self._cur_batch_data is None:
            self.sample_training_data([self.X_train, self.y_train], 256)
        nn = copy.deepcopy(self.nn)
        best_i = 0
        best_loss = 0
        for i in range(10):
            self.learn_rate = 10 ** -i
            self.forward(self._cur_train_data)  # self.output, Y_train
            last_loss = self.loss
            last_accuracy = self.accuracy
            self.backward()
            self.forward(self._cur_train_data)  # self.output, Y_train
            this_loss = self.loss
            this_accuracy = self.accuracy
            print("learn rate: %.10e, last loss: %f, this loss: %f, last accuracy: %f, this accuracy: %f" % (
                self.learn_rate, last_loss, this_loss, last_accuracy, this_accuracy))
            if this_loss - last_loss < best_loss:
                best_loss = this_loss - last_loss
                best_i = -i
            self.nn = copy.deepcopy(nn)
        self.learn_rate = 10 ** best_i
        print("best learn rate: %.10e" % self.learn_rate)

    def overfit_check(self, times):
        if self._cur_batch_data is None:
            self.sample_training_data([self.X_train, self.y_train], 32)

        self.forward(self._cur_train_data)  # self.output, Y_train
        print("loss: %f, data loss: %f, regular loss: %f, train accuracy: %f" % (
            self.loss, self._data_loss, self._reg_loss, self.accuracy))

        for i in range(times):
            self.backward()
            self.forward(self._cur_train_data)  # self.output, Y_train
            print("i: %d, loss: %f, data loss: %f, regular loss: %f, train accuracy: %f" % (
                i, self.loss, self._data_loss, self._reg_loss, self.accuracy))

    def sample_training_data(self, data, num):
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
        random_list = random.sample(range(N), num)
        self._cur_batch_data = [data[0][:, random_list], data[1][:, random_list]]
        t_num = int(num * 0.8)
        self._cur_train_data = [self._cur_batch_data[0][:, :t_num], self._cur_batch_data[1][:, :t_num]]
        self._cur_valid_data = [self._cur_batch_data[0][:, t_num:], self._cur_batch_data[1][:, t_num:]]
        return self._cur_batch_data

    def train(self, times):
        for i in range(times):
            self.sample_training_data([self.X_train, self.y_train], 256)
            self.forward(self._cur_train_data)  # self.output, Y_train
            print("i: {:d}".format(i))
            # print("\t train, loss: {:f}, data loss: {:f}, regular loss: {:f}, accuracy: {:%}".format(
            #     i, self.loss, self._data_loss, self._reg_loss, self.accuracy))
            self.backward()
            self.validation()
        pass

    def validation(self):
        if self._cur_valid_data is None:
            return
        self.forward(self._cur_valid_data)  # self.output, Y_train
        validation_loss = self.loss
        validation_data_loss = self._data_loss
        validation_reg_loss = self._reg_loss
        validation_accuracy = self.accuracy
        self.forward(self._cur_train_data)
        print("\t train, loss: {:f}, data loss: {:f}, regular loss: {:f}, accuracy: {:%}".format(
            self.loss, self._data_loss, self._reg_loss, self.accuracy))
        print("\t validation, loss: {:f}, data loss: {:f}, accuracy: {:%}".format(
            validation_loss, validation_data_loss, validation_accuracy))

    def test(self):
        test_data = [self.X_test, self.y_test]
        self.forward(test_data)
        print("test, loss: {:f}, data loss: {:f}, regular loss: {:f}, accuracy: {:%}".format(
            self.loss, self._data_loss, self._reg_loss, self.accuracy))

    def visualize(self):
        # Visualize some examples from the dataset.
        # We show a few examples of training images from each class.
        classes = ['plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(self.y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace = False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(self.X_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()



def CIFAR10_test():
    from load_data import load_CIFAR10

    X_train, Y_train, X_test, Y_test = load_CIFAR10('../data/cifar10/')  # 3072 x 50000

    nn = NN3(4, reg_lam = 0.01)
    nn.load_examples(X_train, Y_train, axis = 1)
    nn.load_test_data(X_test, Y_test)
    nn.data_preprocessing()
    nn.weight_init()
    nn.visualize()
    # nn.gradient_check()
    # nn.test_learn_rate()
    # nn.overfit_check(500)
    # nn.train(500)
    # nn.test()


if __name__ == '__main__':
    CIFAR10_test()
