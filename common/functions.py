import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.)


def identity_function(x):
    return x


def softmax(x):
    c = np.max(np.exp(x))
    return np.exp(x - c) / np.sum(np.exp(x - c))


def cross_entropy_error(y, label):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        label = label.reshape(1, label.size)
    batch_size = y.shape[0]
    return -np.sum(label * np.log(y)) / batch_size
