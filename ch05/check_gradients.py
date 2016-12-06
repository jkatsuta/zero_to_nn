#!/usr/bin/env python
import os
import sys
import numpy as np
import twolayers
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


(x_train, y_train), (_, _) = load_mnist(normalize=True, one_hot_label=True)
x_batch = x_train[:3]
y_batch = y_train[:3]
print(y_batch)

network = twolayers.TwoLayerNet(x_batch.shape[1], 50, 10)
grad_numerical = network.numerical_gradient(x_batch, y_batch)
grad_backprop = network.gradient(x_batch, y_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
    print(key + " : " + str(diff))
