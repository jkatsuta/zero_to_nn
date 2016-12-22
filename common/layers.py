import numpy as np
import os
import sys
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.util import im2col, col2im


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
        # for convolution net
        self.shape_x_org = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx.reshape(*self.shape_x_org)


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
            dx[np.arange(batch_size), self.t] -= 1
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


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        self.x = x
        N, C, H, W = self.x.shape
        FN, C, FH, FW = self.W.shape
        h_out = int((H + 2 * self.pad - FH) / self.stride) + 1
        w_out = int((W + 2 * self.pad - FW) / self.stride) + 1

        self.x_col = im2col(x, FH, FW, self.stride, self.pad)
        self.w_col = self.W.reshape(FN, -1).T
        out = np.dot(self.x_col, self.w_col) + self.b  # numpy broadcast
        out = out.reshape(N, h_out, w_out, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x_col.T, dout).T.reshape(*self.W.shape)
        dcol = np.dot(dout, self.w_col.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


class Pooling:
    def __init__(self, w_p, h_p, stride=1, pad=0):
        self.w_p = w_p
        self.h_p = h_p
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        self.x = x
        N, C, H, W = self.x.shape
        h_out = int((H + 2 * self.pad - self.h_p) / self.stride) + 1
        w_out = int((W + 2 * self.pad - self.w_p) / self.stride) + 1

        x_col = im2col(x, self.h_p, self.w_p, self.stride, self.pad)\
            .reshape(-1, self.h_p * self.w_p)
        x_pool = x_col.max(axis=1)
        out = x_pool.reshape(N, h_out, w_out, C).transpose(0, 3, 1, 2)

        self.arg_max = np.argmax(x_col, axis=1)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.h_p * self.w_p
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.h_p, self.w_p, self.stride, self.pad)
        return dx
