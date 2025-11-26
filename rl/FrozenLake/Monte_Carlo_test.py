import FrozenLake
import numpy as np
from matplotlib import pyplot as plt

'''
nrow, ncol, st, ed, holes =5, 5, 0, 24, [4, 6, 13, 16, 18]
'''
nrow, ncol, st, ed, holes = 4, 4, 0, 15, [5, 7, 11, 12] # standard form

env = FrozenLake.FrozenLake(nrow, ncol, startp=st, endp=ed, holes=holes)

sample_number_list = []
max_diff_list = []

analytical_solution = FrozenLake.Analytical_Solution(env, 0.9)
analytical_solution.compute_V()
print(analytical_solution.v)

np.random.seed(1)
epochs, gap = 100, 100
for epoch in range(1, epochs):
    sample_number_list.append(epoch * gap)
    monte_carlo = FrozenLake.Monte_Carlo(env, epoch * gap, 0.9)
    V = monte_carlo.compute_V()
    max_diff = 0
    for s in range(env.ncol * env.nrow):
        max_diff = max(max_diff, abs(V[s]-analytical_solution.v[s]))
    max_diff_list.append(max_diff)


fig, ax = plt.subplots()
ax.plot(sample_number_list[1:], max_diff_list[1:])

ax.set(xlabel='sample number list', ylabel='max diff',
       title='finding the best sampling number')
ax.grid()
plt.show()
