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

    def numerical_gradient(self, x, label):
        def f_loss(k, v):
            def f(v):
                self.params[k] = v
                return self.loss(x, label)
            return f

        grads = {}
        for k, v in self.params.items():
            grads[k] = numerical_gradient(f_loss(k, v), v)
        return grads
