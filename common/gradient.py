import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def diff_tangent(f, x0):
    alpha = numerical_diff(f, x0)
    return lambda x: alpha * (x - x0) + f(x0)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fx_p_h = f(x)
        x[idx] = tmp_val - h
        fx_m_h = f(x)

        grad[idx] = (fx_p_h - fx_m_h) / (2 * h)
        x[idx] = tmp_val
        it.iternext()
    return grad
