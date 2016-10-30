#!/usr/bin/env python
import os
import sys
import numpy as np
sys.path.append(os.pardir)


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


def init_network():
    network = {}
    network['W1'] = np.arange(0.1, 0.7, 0.1).reshape(2, 3)
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.arange(0.1, 0.7, 0.1).reshape(3, 2)
    network['B2'] = np.array([0.1, 0.2])
    network['W3'] = np.arange(0.1, 0.5, 0.1).reshape(2, 2)
    network['B3'] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
