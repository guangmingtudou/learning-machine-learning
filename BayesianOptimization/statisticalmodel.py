import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_line(x, y):
    plt.plot(x, y)
    plt.show()

def plot_3d(x, y):
    X, Y = np.meshgrid(x[0], x[1])
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, y, cmap='coolwarm', alpha=0.9, edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')
    ax.set_title('probs')
    ax.view_init(elev=30, azim=45)  # 仰角30度，方位角45度

    plt.show()

class UnivarientGaussianDistribution:
    def __init__ (self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def P(self, x):
        return 1 / (self.sigma * np.sqrt(2*np.pi)) * np.exp(-(x-self.mu)**2/(2*self.sigma**2))
    
    def sampling(self, n):
        x = np.linspace(self.mu-self.sigma*3, self.mu+self.sigma*3, n)
        y = self.P(x)
        return x, y

class MultivarientGaussianDistribution:
    def __init__ (self, mu, sigma):
        """
        Docstring for __init__
        param n: the number of variables
        param mu: Average numbers. Should be np.ndarray(-1, 1)
        param sigma: should be np.ndarray(-1, 1)
        """
        self.n = mu.shape[0]
        self.mu = mu
        self.sigma = sigma
        self.K = np.diag((sigma ** 2).reshape(-1))
    
    def generate_mu_sigma(n):
        mu = (np.random.rand(n) * 10).reshape(-1,1)
        sigma = np.random.rand(n).reshape(-1,1)
        return mu, sigma
    
    def P(self, x):
        return np.exp(-0.5*np.dot(np.dot((x-self.mu).T, self.K), (x-self.mu))) / (((2*np.pi) ** (self.n/2)) * np.sqrt(np.linalg.det(self.K)))
    
    def calculate(self, x):
        l = x.shape[1]
        y = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                x_tmp = np.array([x[0][i],x[1][j]]).reshape(-1,1)
                y[i][j] = self.P(x_tmp)
                #print(x_tmp)
            #print(i,"row")
        return y
    
    def sampling(self, num):
        x = np.linspace(self.mu[0]-self.sigma[0]*3, self.mu[0]+self.sigma[0]*3, num).reshape(1, -1)
        for i in range(1, self.n):
            tmp = np.linspace(self.mu[i]-self.sigma[i]*3, self.mu[i]+self.sigma[i]*3, num).reshape(1, -1)
            x = np.concatenate((x, tmp), axis=0)
        y = self.calculate(x)
        return x, y

if __name__ == "__main__":
    '''
    uvgd = UnivarientGaussianDistribution(0, 1)
    x, y = uvgd.sampling(100)
    plot_line(x, y)
    '''

    mu, sigma = MultivarientGaussianDistribution.generate_mu_sigma(2)
    sigma = np.array([0.4, 0.5])
    mvgd = MultivarientGaussianDistribution(mu, sigma)
    x, y = mvgd.sampling(100)
    plot_3d(x, y)