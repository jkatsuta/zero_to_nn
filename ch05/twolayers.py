#!/usr/bin/env python
import os
import sys
import numpy as np
from collections import OrderedDict
sys.path.append(os.pardir)
import common.layers as clay
import common.gradient as cgrad


class TwoLayerNet:
    def __init__(self, size_in, size_hidden, size_out,
                 std_init_weight=0.01):
        self.params = {}
        self.params['W1'] = std_init_weight * np.random.randn(size_in, size_hidden)
        self.params['b1'] = np.zeros(size_hidden)
        self.params['W2'] = std_init_weight * np.random.randn(size_hidden, size_out)
        self.params['b2'] = np.zeros(size_out)

        self.layers = OrderedDict()
        self.layers['Affine1'] = clay.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = clay.Relu()
        self.layers['Affine2'] = clay.Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = clay.SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        pred = y.argmax(axis=1)
        if t.ndim != 1:
            t = t.argmax(axis=1)
        return np.sum(pred == t) / float(pred.size)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = cgrad.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = cgrad.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = cgrad.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = cgrad.numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
