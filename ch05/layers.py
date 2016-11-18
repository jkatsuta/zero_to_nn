import numpy as np


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
