import math

class UnivarientGaussianDistribution:
    def __init__ (self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def function(self, x):
        p = 1 / (self.sigma * math.sqrt(2*math.pi)) * math.exp(-(x-self.mu)**2/(2*self.sigma**2))