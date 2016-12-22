import os
import gzip
import pickle
import urllib.request
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist'
dic_files = {
    'img_train': 'train-images-idx3-ubyte.gz',
    'label_train': 'train-labels-idx1-ubyte.gz',
    'img_test': 't10k-images-idx3-ubyte.gz',
    'label_test': 't10k-labels-idx1-ubyte.gz'
}

dir_dataset = os.path.dirname(os.path.abspath(__file__))
dic_files = {k: os.path.join(dir_dataset, imgfile)
             for k, imgfile in dic_files.items()}
mnistfile = dir_dataset + "/mnist.pkl"

n_train = 60000
n_test = 10000
dim_img = (1, 28, 28)
size_img = 784


def init_mnist():
    download_mnist()
    dataset = _to_numpy()
    print('saving the mnist file in the pickle format ...')
    with open(mnistfile, 'wb') as g:
        pickle.dump(dataset, g, -1)


def download_mnist():
    for imgfile in dic_files.values():
        if not os.path.exists(imgfile):
            print('downloading %s ...' % os.path.basename(imgfile))
            src = os.path.join(url_base, os.path.basename(imgfile))
            urllib.request.urlretrieve(src, imgfile)


def _to_numpy():
    dataset = {}
    for k, imgfile in dic_files.items():
        with gzip.open(imgfile, 'rb') as f:
            if k.startswith('img_'):
                data = np.frombuffer(f.read(), np.uint8, offset=16)
                data = data.reshape(-1, size_img)
            elif k.startswith('label_'):
                data = np.frombuffer(f.read(), np.uint8, offset=8)
        dataset[k] = data
    return dataset


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(mnistfile):
        init_mnist()

    with open(mnistfile, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for k, img in dataset.items():
            if k.startswith('img_'):
                dataset[k] = img.astype(np.float32) / 255.
    if not flatten:
        for k, img in dataset.items():
            if k.startswith('img_'):
                dataset[k] = img.reshape(-1, 1, 28, 28)
    if one_hot_label:
        for k, label in dataset.items():
            if k.startswith('label_'):
                one_hot = np.zeros((label.size, 10))
                one_hot[np.arange(label.size), label] = 1
                dataset[k] = one_hot
    return ((dataset['img_train'], dataset['label_train']),
            (dataset['img_test'], dataset['label_test']))
