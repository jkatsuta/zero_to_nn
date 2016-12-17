import numpy as np
import os
import sys
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.where(self.x > 0, self.x, 0.)

    def backward(self, dout):
        return np.where(self.x > 0, dout, 0.)


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1. / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1. - self.out)
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.t = None
        self.y = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.ndim == 1:
            dx = self.y.copy()
            dx[:, self.t] -= 1
        elif self.t.ndim == 2:
            dx = self.y - self.t
        return dx / batch_size


class Dropout:
    def __init__(self, ratio_dropout=0.):
        self.mask = None
        self.r_do = ratio_dropout

    def forward(self, x, flg_train):
        if flg_train:
            self.mask = np.random.rand(*x.shape) >= self.r_do
            return x * self.mask
        else:
            return x * (1. - self.r_do)

    def backward(self, dout):
        return dout * self.mask
