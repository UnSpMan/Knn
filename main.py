import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import KNearestNeighbor
import matplotlib.pyplot as plt

def SVM(X_train, y_train, X_test, y_test):
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    from cs231n.classifiers.linear_svm import svm_loss_naive
    import time

    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    print('loss: %f' % (loss,))
cifar10_dir = 'cs231n/datasets/cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
SVM(X_train,y_train,X_test,y_test)

