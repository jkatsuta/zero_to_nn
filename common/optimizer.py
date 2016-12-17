import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update(self, params, grads):
        if not self.vs:
            for key, val in params.items():
                self.vs[key] = np.zeros_like(val)

        for key in params.keys():
            self.vs[key] = self.momentum * self.vs[key] - self.lr * grads[key]
            params[key] += self.vs[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.hs = {}

    def update(self, params, grads):
        if not self.hs:
            for key, val in params.items():
                self.hs[key] = np.zeros_like(val)

        for key in params.keys():
            self.hs[key] += grads[key]**2
            params[key] -= self.lr / (np.sqrt(self.hs[key]) + 1e-7) * grads[key]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.ms, self.vs = {}, {}
        self.iter = 0

    def update(self, params, grads):
        if not self.ms:
            for key, val in params.items():
                self.ms[key] = np.zeros_like(val)
                self.vs[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1. - self.beta2**self.iter) / (1. - self.beta1**self.iter)
        for key in params.keys():
            self.ms[key] = self.beta1 * self.ms[key] + (1. - self.beta1) * grads[key]
            self.vs[key] = self.beta2 * self.vs[key] + (1. - self.beta2) * grads[key]**2
            params[key] -= lr_t * self.ms[key] / (np.sqrt(self.vs[key]) + 1e-7)
