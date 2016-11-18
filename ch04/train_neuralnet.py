#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layers_net import TwoLayersNet

(x_train, y_train), (x_test, y_test) =\
    load_mnist(normalize=True, one_hot_label=True)

iter_num = 2
batch_size = 100
learning_rate = 0.1

net = TwoLayersNet(784, 10, 10)

losses_train = []
acc_train, acc_test = [], []
iter_per_epoch = int(x_train.shape[0] / batch_size)
for i in range(iter_num):
    print(i, iter_per_epoch)
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]

    grads = net.numerical_gradient(x_batch, y_batch)
    for k in net.params.keys():
        net.params[k] -= learning_rate * grads[k]

    losses_train.append(net.loss(x_batch, y_batch))
    if i % iter_per_epoch == 0:
        acc_train.append(net.accuracy(x_train, y_train))
        acc_test.append(net.accuracy(x_test, y_test))
        print('acc_train: %s,  acc_test: %s' % (acc_train[-1], acc_test[-1]))
plt.plot(losses_train)
plt.show()
