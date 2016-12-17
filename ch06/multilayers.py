#!/usr/bin/env python
import os
import sys
import numpy as np
from collections import OrderedDict
sys.path.append(os.pardir)
import common.layers as clay
import common.gradient as cgrad


class MultiLayerNet:
    def __init__(self, size_in, list_size_hidden, size_out):
        if isinstance(list_size_hidden, int):
            list_size_hidden = [list_size_hidden]
        self.size_units = [size_in] + list_size_hidden + [size_out]
        self.params = {}
        self.layers = OrderedDict()

        for i in range(len(self.size_units) - 1):
            affine, weight, bias, relu =\
                [k + str(i + 1) for k in ('Affine', 'W', 'b', 'Relu')]
            scale = np.sqrt(2.0 / self.size_units[i])
            self.params[weight] =\
                scale * np.random.randn(self.size_units[i], self.size_units[i+1])
            self.params[bias] = np.zeros(self.size_units[i+1])

            self.layers[affine] = clay.Affine(self.params[weight], self.params[bias])
            if i < len(self.size_units) - 2:
                self.layers[relu] = clay.Relu()
            else:
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
            grads[weight] = self.layers[affine].dW
            grads[bias] = self.layers[affine].db
        return grads
