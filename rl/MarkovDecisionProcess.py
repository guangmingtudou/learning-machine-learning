import numpy as np
np.random.seed(0)

def compute(P, rewards, gamma, states_num): #贝尔曼方程的矩阵形式计算解析式
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value