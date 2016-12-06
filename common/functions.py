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
    if x.ndim == 1:
        xmod = x - np.max(x)
        v = np.exp(xmod) / np.sum(np.exp(xmod))
    elif x.ndim == 2:
        xmod = x - np.max(x, axis=1).reshape(x.shape[0], 1)
        v = np.exp(xmod) / np.sum(np.exp(xmod), axis=1).reshape(x.shape[0], 1)
    return v


def cross_entropy_error(y, label):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        label = label.reshape(1, label.size)
    if label.ndim == 2:
        label = np.argmax(label, axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), label])) / batch_size
