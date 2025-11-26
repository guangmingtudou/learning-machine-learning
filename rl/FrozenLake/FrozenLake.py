import copy
import numpy as np
import debug_tool


class FrozenLake:
    def __init__(self, nrow=4, ncol=4, startp=0, endp=15, holes=None):
        self.nrow = nrow
        self.ncol = ncol
        self.startp = startp
        self.endp = endp
        self.holes = holes
        self.a_std = [[0,-1], [0,1], [-1,0], [1,0]]
        self.a_l = [[-1,0], [1,0], [0,1], [0,-1]]
        self.a_r = [[1,0], [-1,0], [0,-1], [0,1]]
        self.P = self.initP(nrow, ncol)
    
    def ishole(self, idx): #是洞就True
        if self.holes==None:
            return False
        if idx in self.holes:
            return True
        else:
            return False
        
    def initP(self, nrow=4, ncol=4): #(p, next_state, r, done)
        P = [[[] for _ in range(4)] for _ in range(nrow*ncol)]
        for i in range(nrow*ncol): #state
            if self.ishole(i): 
                r = -100
                done = True
                for j in range(4): #action
                    P[i][j] = [(1, i, r, done)]
                continue
            elif i == self.endp:
                r = 100
                done = True
                for j in range(4): #action
                        P[i][j] = [(1, i, r, done)]
                continue
            else: 
                r = -1
                done = False
            for j in range(4): #action
                next_state_std, next_state_r, next_state_l = self.next_state(i, self.a_std[j]), self.next_state(i, self.a_r[j]), self.next_state(i, self.a_l[j])
                P[i][j] = [(0.33333333, next_state_std, r, done), (0.33333333, next_state_l, r, done), (0.33333333, next_state_r, r, done)]
        return P
    
    def next_state(self, state, action):
        # 边界条件检验
        if (state < self.ncol and action[1] == -1): return state
        elif (state >= (self.nrow-1)*self.ncol and action[1] == 1): return state
        elif (state % self.ncol == 0 and action[0] == -1): return state
        elif (state % self.ncol == self.ncol-1 and action[0] == 1): return state
        #真走
        return state + action[0] + action[1] * self.ncol
    
def print_strategy(solution):
    st = ['^', 'v', '<', '>']
    for i in range(solution.env.nrow):
        for j in range(solution.env.ncol):
            if (solution.env.ishole(i * solution.env.ncol + j)):
                print("hole ", end='')
                continue
            elif (solution.env.endp == i * solution.env.ncol + j):
                print("end! ", end='')
                continue
            for a in range(4):
                if (solution.pi[i * solution.env.ncol + j][a] == 0):
                    print('o', end='')
                else:
                    print(st[a], end='')
            print(' ', end='')
        print(end='\n')
'''
nrow, ncol, st, ed, holes =5, 5, 0, 24, [4, 6, 13, 16, 18]
'''
nrow, ncol, st, ed, holes = 4, 4, 0, 15, [5, 7, 11, 12] # standard form

env = FrozenLake(nrow, ncol, startp=st, endp=ed, holes=holes)
'''
for i in range(env.ncol * env.nrow):
    for j in range(4):
        print(env.P[i][j])
'''
class Analytical_Solution:
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(env.ncol * env.nrow)]
        self.v = [.0] * env.ncol * env.nrow
    
    def degrade_P_R(self):
        n = self.env.ncol * self.env.nrow
        P = np.zeros(shape=(n, n))
        R = np.zeros(n)
        for s in range(n):
            if (self.env.ishole(s)): 
                R[s] = -100
                continue
            elif (self.env.endp == s):
                R[s] = 100
                continue
            else: R[s] = -1
            for a in range(4):
                pi = self.pi[s][a]
                for tmp in self.env.P[s][a]:
                    p, ns= tmp[0], tmp[1]
                    P[s][ns] += p*pi
                    #r(s,a)
        return P, R
    
    def compute_V(self):
        P, R = self.degrade_P_R()
        self.v = np.dot(np.linalg.inv(np.eye(self.env.ncol * self.env.nrow) - self.gamma * P), R) #V = (I - gammaP)^(-1)*R
    
    def policy_improve(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for tmp in self.env.P[s][a]:
                    p, ns, r, done = tmp
                    qsa += p * (r + self.gamma * self.v[ns] * (1 - done))
                qsa_list.append(int(qsa * 1000000) / 1000000) 
            qsa_max = max(qsa_list)
            qsa_n = qsa_list.count(qsa_max)
            self.pi[s] = [1 / qsa_n if q==qsa_max else 0 for q in qsa_list]
        return self.pi
    
    def train(self):
        cnt = 0
        while 1:
            self.compute_V()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improve()
            cnt += 1
            #debug_tool.print_V(self.v, 4, 4) #打印Value function
            #print_strategy(self)
            if old_pi == new_pi: break
            if cnt>200: break
        print(f'计算V解析解, 训练{cnt}次')
        
'''
analytical_solution = Analytical_Solution(env, 0.9)
analytical_solution.train()

print_strategy(analytical_solution)
'''
np.random.seed(1)
class Monte_Carlo:
    def __init__(self, env, run_turns, gamma):
        self.env = env
        self.run_turns = run_turns
        self.gamma = gamma
        self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(env.ncol * env.nrow)]
    
    def run_one_time(self, s):
        r_list = []
        for i in range(43):
            if (self.env.ishole(s)) or (self.env.endp == s): break
            random_a = np.random.rand()
            tmp = 0
            for a in range(4):
                tmp += self.pi[s][a]
                if tmp >= random_a: break
            random_wind = np.random.rand()
            tmp = 0
            for complex in self.env.P[s][a]:
                tmp += complex[0]
                if tmp >= random_wind: break
            r_list.append(complex[2])
            s = complex[1]
            
        r_list.append(self.env.P[s][0][0][2])
        #计算回报
        g = 0
        for i in reversed(range(len(r_list))):
            g = g * self.gamma + r_list[i]
        return g
    
    def compute_V(self):
        n_s = self.env.ncol * self.env.nrow
        V = np.zeros(n_s)
        for s in range(n_s):
            g_sum=0
            for _ in range(self.run_turns):
                g_sum += self.run_one_time(s)
            V[s] = g_sum / self.run_turns
        #print(f'蒙特卡洛估计V结束')
        return V

    def policy_improve(self, V):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for tmp in self.env.P[s][a]:
                    p, ns, r, done = tmp
                    qsa += p * (r + self.gamma * V[ns] * (1 - done))
                qsa_list.append(qsa)
            qsa_max = max(qsa_list)
            qsa_n = qsa_list.count(qsa_max)
            self.pi[s] = [1 / qsa_n if qsa_list[i]==qsa_max else 0 for i in range(4)]
        return self.pi
    
    
    def train(self):
        cnt = 0
        while 1:
            V = self.compute_V()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improve(V)
            cnt += 1
            if old_pi == new_pi: break
            if cnt>200: break
            #print(V)
        print(f'蒙特卡洛估算V, 训练{cnt}次')
'''
monte_carlo = Monte_Carlo(env, 100, 0.9)
monte_carlo.train()
print_strategy(monte_carlo)

'''
class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = np.zeros(self.env.ncol * self.env.nrow, dtype=np.float32)
        self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(env.ncol * env.nrow)]

    def compute_V(self):
        cnt=0
        while 1:
            new_v = [.0] * self.env.ncol * self.env.nrow
            max_diff = 0
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for tmp in self.env.P[s][a]:
                        p, ns, r, done = tmp
                        qsa += p * (r + self.gamma * self.v[ns] * (1 - done))
                    qsa_list.append(qsa * self.pi[s][a])
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        self.v = new_v
    
    def policy_improve(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for tmp in self.env.P[s][a]:
                    p, next_state, r, done = tmp
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(int(qsa * 1000000) / 1000000)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        return self.pi
    
    def train(self):
        cnt = 0
        while 1:
            self.compute_V()
            #debug_tool.print_V(self.v, self.env.nrow, self.env.ncol)
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improve()
            cnt += 1
            if old_pi == new_pi: break
        print(f'策略迭代, 训练{cnt}次')
'''   
policyiteration = PolicyIteration(env, 1e-5, 0.9)
policyiteration.train()
print_strategy(policyiteration)
'''