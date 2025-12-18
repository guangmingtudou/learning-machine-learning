#This is a homework program that I wrote, and I want to use it as an example.
#hyperparameters: patch_size, lr

import numpy as np
import copy
import sys

import os

def housepricepredict(patch_size, lr):
    np.random.seed(0)

    # data about houses
    import pandas as pd
    house = pd.read_csv('D:\chw\programming\learning_machine_learning\BayesianOptimization\house.csv', header=0)
    house_filtered = house[(house['Bldg Type']=='1Fam') & (house['Sale Condition']=='Normal')]
    house_filtered = house_filtered[['SalePrice','1st Flr SF', '2nd Flr SF', 'Total Bsmt SF', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Lot Area', 'Year Built']]
    #Yr sold deleted
    #print(house_filtered)

    # get features and targets
    house_matrix = house_filtered.to_numpy()
    X = house_matrix[:, 1:].astype(float)
    y = house_matrix[:, 0].astype(float)
    y = np.log(y+1).reshape(-1, 1)

    #normalize
    def normalize(x):
        x_mean = np.mean(x,axis=0)
        x_std = np.std(x,axis=0)
        x_std = np.where(x_std == 0, 1, x_std)
        return (x-x_mean)/x_std, x_mean, x_std
    X, X_mean, X_std = normalize(X)
    y, y_mean, y_std = normalize(y)

    # initialize parameters
    w = np.random.randn(X.shape[1], 1)
    b = 0.0

    def MSE_loss(X, y, w, b):
        return np.mean((np.dot(X, w) + b - y) ** 2)

    def para_update(X, y, w, b):
        n = X.shape[0]
        error = np.dot(X, w) + b - y.reshape(-1, 1)
        dw = (2.0 / n) * np.dot(X.T, error) 
        db = (2.0 / n) * np.sum(error) 
        return dw, db

    patch_size = patch_size
    lr = lr
    theta = 1e-5
    max_epoch = 1000

    def train_epoch(X, y, w, b):
        for i in range(0, y.shape[0], patch_size):
            ed = i + patch_size
            if (ed > y.shape[0]):
                ed = y.shape[0]
            X_0 = X[i:ed, :]
            y_0 = y[i:ed]
            l = MSE_loss(X_0, y_0, w, b)
            dw, db = para_update(X_0, y_0, w, b)
            w -= dw * lr
            b -= db * lr
        return w, b
        
    epoch=0
    old_loss = 0
    for epoch in range(max_epoch):
        epoch += 1
        indice = np.arange(X.shape[0])
        np.random.shuffle(indice)
        X_shuffle = X[indice]
        y_shuffle = y[indice]
        w, b = train_epoch(X_shuffle, y_shuffle, w, b)
        loss = MSE_loss(X, y, w, b)
        #loss下降到theta以下停止
        if loss > old_loss and epoch != 1: break
        old_loss = copy.deepcopy(loss)

        #if epoch % 10 ==0: print(f'第{epoch}轮, loss={MSE_loss(X, y, w, b)}\n')    
        
    return epoch, MSE_loss(X, y, w, b)