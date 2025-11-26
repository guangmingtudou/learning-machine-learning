import numpy as np

class bandit:
    def __init__(self, arm_n):
        self.arm_n = arm_n
        self.probs = np.random.rand(arm_n)
        self.max_idx = np.argmax(self.probs)
        self.max_prob = self.probs[self.max_idx]
    
    def pull(self, k):
        return np.random.rand() < self.probs[k]

class EGreedy:
    def __init__(self, bandit):
        self.bandit = bandit
        self.assume_probs=[0.] * bandit.arm_n
        self.use_arms=[0] * bandit.arm_n
        self.att = 0
        self.max_idx = 0
        self.regrets = []
        self.regret = 0

    def attempt(self):
        self.att += 1
        if np.random.rand() < (1./self.att):
            k = np.random.randint(0, self.bandit.arm_n)
            self.pull(k)
        else:
            self.pull(self.max_idx)
    
    def pull(self, k):
        self.use_arms[k] += 1
        self.assume_probs[k] += (self.bandit.pull(k) - self.assume_probs[k]) / self.use_arms[k]
        self.max_idx = np.argmax(self.assume_probs)
        self.regret += self.bandit.max_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

bandit_10_arms = bandit(10)
solve = EGreedy(bandit_10_arms)
for i in range(1000):
    solve.attempt()

print(solve.regrets)