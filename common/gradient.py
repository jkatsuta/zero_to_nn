import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2*h)


def diff_tangent(f, x0):
    alpha = numerical_diff(f, x0)
    return lambda x: alpha * (x - x0) + f(x0)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        hs = np.zeros_like(x)
        hs[it.multi_index] = h
        grad[it.multi_index] = (f(x + hs) - f(x - hs)) / (2 * h)
        it.iternext()
    return grad