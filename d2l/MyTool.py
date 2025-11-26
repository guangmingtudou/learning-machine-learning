'''
这个文件包含以下功能
画三位散点图
-------------------
To be done
训练train

'''
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

def plot_points_3d(xdata, ydata, zdata):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(xdata, ydata, s=np.ones(len(zdata)) * 5, zs=zdata)
    plt.show()

import torch
from torch import nn

def train_epoch(net, training_set, loss, updater):
    if isinstance(net, nn.Module):
        net.train()
    for X, y in training_set:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()

def test_accuracy(net, test_set):
    for X, y in test_set:
        y_hat = net(X)
        y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y 

    return 

def train(net, training_set, test_set, loss, epoch_number, updater, lr=None):
    for epoch in range(epoch_number):
        train_epoch(net, training_set, loss, updater)

