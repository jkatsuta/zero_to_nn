import os
import sys
import numpy as np
sys.path.append(os.pardir)
from common.functions import sigmoid, cross_entropy_error
from common.gradient import numerical_gradient


class TwoLayersNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std *\
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std *\
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)
        return y

    def loss(self, x, label):
        y = self.predict(x)
        return cross_entropy_error(y, label)

    def accuracy(self, x, label):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        label = np.argmax(label, axis=1)
        return np.sum(y == label) / float(y.size)

    def numerical_gradient(self, x, label):
        f = lambda v: self.loss(x, label)
        grads = {}
        for k in self.params.keys():
            grads[k] = numerical_gradient(f, self.params[k])
        return grads
