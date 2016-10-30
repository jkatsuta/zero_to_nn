#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.)


def identity_function(x):
    return x


def softmax(x):
    c = np.max(np.exp(x))
    return np.exp(x - c) / np.sum(np.exp(x - c))


def init_network(fn_weights):
    with open(fn_weights, 'rb') as f:
        network = pickle.load(f)
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


def get_data():
    (x_train, t_train), (x_test, t_test) =\
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def calc_accuracy(y, label):
    pred = np.argmax(y, axis=1)
    acc = np.sum(pred == label) / pred.size
    return acc


def cross_entropy_error(y, label):
    if y.dim == 1:
        y = y.reshape(1, y.size)
        label = label.reshape(1, label.size)
    batch_size = y.shape[0]
    return -np.sum(label * np.log(y)) / batch_size


if __name__ == '__main__':
    fn_weights = 'sample_weight.pkl'
    x, t = get_data()
    network = init_network(fn_weights)
    y = forward(network, x)
    acc = calc_accuracy(y, t)
    print('Accuracy: %f' % acc)
