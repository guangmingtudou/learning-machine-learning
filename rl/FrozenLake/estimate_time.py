import FrozenLake
import time
import matplotlib.pyplot as plt
import try_reward_iteration


nrow, ncol, st, ed, holes =5, 5, 0, 24, [4, 6, 13, 16, 18]
'''
nrow, ncol, st, ed, holes = 4, 4, 0, 15, [5, 7, 11, 12] # standard form
'''
env = FrozenLake.FrozenLake(nrow, ncol, startp=st, endp=ed, holes=holes)
'''
for i in range(env.ncol * env.nrow):
    for j in range(4):
        print(env.P[i][j])
'''

t1 = time.time()
analytical_solution = FrozenLake.Analytical_Solution(env, 0.9)
analytical_solution.train()

t2 = time.time()
monte_carlo = FrozenLake.Monte_Carlo(env, 200, 0.9)
monte_carlo.train()

t3 = time.time()
policyiteration = FrozenLake.PolicyIteration(env, 1e-4, 0.9)
policyiteration.train()


t4 = time.time()
rewarditeration = try_reward_iteration.ValueIteration(env, 1e-4, 0.9)
rewarditeration.value_iteration()
t5 = time.time()
print("时间消耗: ")
print(f'解析解{t2-t1}, 蒙特卡洛{t3-t2}, 策略迭代{t4-t3}, 价值迭代{t5-t4}')
'''
#增加维度
dimension_list = []
analytical_solution_time_list = []
policyiteration_time_list = []
rewarditeration_list = []
for i in range(4, 20):
    dimension_list.append(i)
    env = FrozenLake.FrozenLake(i, i, endp=i*i-1)
    analytical_solution = FrozenLake.Analytical_Solution(env, 0.9)
    policyiteration = FrozenLake.PolicyIteration(env, 1e-4, 0.9)
    rewarditeration = try_reward_iteration.ValueIteration(env, 1e-4, 0.9)

    ts = time.time()
    analytical_solution.train()
    tm = time.time()
    policyiteration.train()
    te = time.time()
    rewarditeration.value_iteration()
    tp = time.time()
    
    analytical_solution_time_list.append(tm-ts)
    policyiteration_time_list.append(te-tm)
    rewarditeration_list.append(tp-te)

plt.plot(dimension_list, analytical_solution_time_list, label='analytical soluion', color='blue', linestyle='-')
plt.plot(dimension_list, policyiteration_time_list, label='policy iteration', color='red', linestyle='--')

plt.plot(dimension_list, rewarditeration_list, label='value iteration', color='green', linestyle='-')
# 添加标签和图例
plt.xlabel('dimension')
plt.ylabel('time cost')
plt.title('comparison')
plt.legend()

# 显示图表
plt.show()
'''