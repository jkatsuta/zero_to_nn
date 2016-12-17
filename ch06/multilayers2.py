#!/usr/bin/env python
import os
import sys
import numpy as np
from collections import OrderedDict
sys.path.append(os.pardir)
import common.layers as clay
import common.gradient as cgrad


class MultiLayerNet:
    def __init__(self, size_in, list_size_hidden, size_out,
                 lambda_l2=0., ratio_dropout=None):
        if isinstance(list_size_hidden, int):
            list_size_hidden = [list_size_hidden]
        self.size_units = [size_in] + list_size_hidden + [size_out]
        self.lambda_l2 = lambda_l2

        self.params = {}
        self.layers = OrderedDict()
        for i in range(len(self.size_units) - 1):
            affine, weight, bias, relu, dropout =\
                [k + str(i + 1) for k in ('Affine', 'W', 'b', 'Relu', 'Dropout')]
            scale = np.sqrt(2.0 / self.size_units[i])
            self.params[weight] =\
                scale * np.random.randn(self.size_units[i], self.size_units[i+1])
            self.params[bias] = np.zeros(self.size_units[i+1])

            self.layers[affine] = clay.Affine(self.params[weight], self.params[bias])
            if ratio_dropout is not None:
                self.layers[dropout] = clay.Dropout(ratio_dropout)
            if i < len(self.size_units) - 2:
                self.layers[relu] = clay.Relu()
            else:
                self.lastLayer = clay.SoftmaxWithLoss()

    def predict(self, x, flg_train=True):
        for k, layer in self.layers.items():
            if k.startswith('Dropout'):
                x = layer.forward(x, flg_train)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        w = 0.
        for i in range(len(self.size_units) - 1):
            w += np.sum(self.params['W%d' % (i + 1)]**2)
        return self.lastLayer.forward(y, t) + 0.5 * self.lambda_l2 * w

    def accuracy(self, x, t):
        y = self.predict(x)
        pred = y.argmax(axis=1)
        if t.ndim != 1:
            t = t.argmax(axis=1)
        return np.sum(pred == t) / float(pred.size)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        for i in range(len(self.size_units) - 1):
            affine, weight, bias = [k + str(i + 1) for k in ('Affine', 'W', 'b')]
            grads[weight] =\
                self.layers[affine].dW + self.lambda_l2 * self.params[weight]
            grads[bias] = self.layers[affine].db
        return grads
