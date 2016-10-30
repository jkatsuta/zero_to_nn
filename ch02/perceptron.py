#!/usr/bin/env python
import numpy as np


def get_weights(typ):
    network = {}
    network['AND'] = [0.5, 0.5, -0.7]
    network['NAND'] = [-0.5, -0.5, 0.7]
    network['OR'] = [0.5, 0.5, -0.1]
    return np.array(network[typ])


def step_function(v):
    return int(v > 0)


def perceptron(typ, x1, x2):
    xs = np.array([x1, x2, 1.])
    weights = get_weights(typ)
    val = sum(weights * xs)
    return step_function(val)


def XOR(x1, x2):
    c1 = perceptron('NAND', x1, x2)
    c2 = perceptron('OR', x1, x2)
    return perceptron('AND', c1, c2)


if __name__ == '__main__':
    typs = 'AND', 'NAND', 'OR', 'XOR'
    cases = (0, 0), (0, 1), (1, 0), (1, 1)
    for typ in typs:
        print('\n', typ)
        for case in cases:
            if typ != 'XOR':
                print(case, perceptron(typ, *case))
            else:
                print(case, XOR(*case))