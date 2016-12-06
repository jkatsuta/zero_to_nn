#!/usr/bin/env python
import os
import sys
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from twolayers import TwoLayerNet

(x_train, t_train), (x_test, t_test) =\
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(x_train.shape[1], 50, t_train.shape[1])

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

n_epoch = max(int(train_size / batch_size), 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % n_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)