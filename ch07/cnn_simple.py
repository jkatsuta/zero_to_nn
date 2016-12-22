import numpy as np
from collections import OrderedDict
from common.layers import Convolution, Pooling, Relu, Affine, SoftmaxWithLoss

class SimpleConvNet:
    def __init__(self, dim_in=(1, 28, 28),
                 par={'num_filter': 30, 'size_filter': 5, 'pad': 0, 'stride': 1},
                 s_hidden=100, s_out=10, std_w_init=0.01):
        n_f = par['num_filter']
        s_f = par['size_filter']
        pad = par['pad']
        stride = par['stride']
        size_in = dim_in[1]

        size_out_conv = int((size_in + 2 * pad - s_f) / stride) + 1
        size_out_pool = int(n_f * (size_out_conv / 2) ** 2)

        self.params = {}
        self.params['W1'] =\
            std_w_init * np.random.randn(n_f, dim_in[0], s_f, s_f)
        self.params['b1'] = np.zeros(n_f)
        self.params['W2'] = std_w_init * np.random.randn(size_out_pool, s_hidden)
        self.params['b2'] = np.zeros(s_hidden)
        self.params['W3'] = std_w_init * np.random.randn(s_hidden, s_out)
        self.params['b3'] = np.zeros(s_out)

        self.layers = OrderedDict()
        self.layers['Conv'] = Convolution(self.params['W1'], self.params['b1'],
                                          stride, pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool'] = Pooling(2, 2, 2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.last_layer.forward(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        pred = y.argmax(axis=1)
        if t.ndim != 1:
            t = t.argmax(axis=1)
        return np.sum(pred == t) / float(pred.size)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv'].dW
        grads['b1'] = self.layers['Conv'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads
