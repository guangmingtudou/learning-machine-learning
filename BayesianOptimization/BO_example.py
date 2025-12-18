import pandas as pd
import numpy as np

import time
import os 

import sklearn
from sklearn.ensemble import RandomForestRegressor
import housepricepredict
from sklearn.model_selection import KFold, cross_validate

#优化器
from bayes_opt import BayesianOptimization

param_grid_opt = {'patch_size': (10,60)
                     , 'lr':(0.001, 0.1)
                    }

def bayesopt_objective(patch_size, lr):
    epoch, loss = housepricepredict.housepricepredict(int(patch_size), lr)
    return -epoch * 5e-4 - loss

def param_bayes_opt(init_points,n_iter):

    #定义优化器，先实例化优化器
    opt = BayesianOptimization(bayesopt_objective #需要优化的目标函数
                               ,param_grid_opt #备选参数空间
                               ,random_state=24 
                              )

    #使用优化器，bayes_opt只支持最大化
    opt.maximize(init_points = init_points #抽取多少个初始观测值
                 , n_iter=n_iter #一共观测/迭代多少次
                )

    #优化完成，取出最佳参数与最佳分数
    params_best = opt.max["params"]
    score_best = opt.max["target"]

    #打印最佳参数与最佳分数
    print("\n","\n","best params: ", params_best,
          "\n","\n","best cvscore: ", score_best)

    #返回最佳参数与最佳分数
    return params_best, score_best

def bayes_opt_validation(params_best):

    epoch, loss = housepricepredict.housepricepredict(int(params_best['patch_size']), params_best['lr'])
    return -epoch * 5e-4 - loss

start = time.time()
#初始看10个观测值，后面迭代290次
params_best, score_best = param_bayes_opt(10,200) 
print('It takes %s minutes' % ((time.time() - start)/60))
validation_score = bayes_opt_validation(int(params_best['patch_size']), params_best['lr'])
print("\n","\n","validation_score: ",validation_score)