import os
import sys
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, label):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, label)
        return loss
