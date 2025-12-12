# Gaussian Process
learning from https://borgwang.github.io/ml/2019/07/28/gaussian-processes.html
## Univarient Gaussian Distribution
It is also known as normal distribution
$p(x) = \frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$

It has parameters $\mu$ (mean value) and $\sigma$ (standard variation)

## Multivarient Gaussian Distribution
assume that $X_1, X_2, ..., X_n$ are independent variables, then

$p(x_1, x_2, ..., x_n) = \prod{p(x_i)} = \frac{1}{(2\pi)^{\frac{n}{2}}\sigma_1\sigma_2...\sigma_n}exp(-\frac{1}{2}[\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}+...+\frac{(x_n-\mu_n)^2}{\sigma_n^2}])$

let $\mathbf{x} - \boldsymbol{\mu} = [x_1-\mu_1, x_2-\mu_2, ..., x_n-\mu_n]^T$


$$ \Sigma = 
\begin{bmatrix}
\sigma_1^2&0&...&0\\
0&\sigma_2^2&...&0\\
...&...&...&...\\
0&0&...&\sigma_n^2\\
\end{bmatrix}
$$
$\Sigma$ is the correlation matrix, $\sigma_{xy}=E[(x-\mu_x)(y-\mu_y)]$


then $\sigma_1\sigma_2...\sigma_n = |\Sigma|^{\frac{1}{2}}$

$\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}+...+\frac{(x_n-\mu_n)^2}{\sigma_n^2} = (\mathbf{x} - \boldsymbol{\mu})^T\Sigma(\mathbf{x} - \boldsymbol{\mu})$

Using these equations, we get $p(x_1, x_2, ..., x_n) = \frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\Sigma(\mathbf{x} - \boldsymbol{\mu}))$ 

logogram: x ~ N($\boldsymbol{\mu}$, $\Sigma$)

It still works if $X_1, X_2, ..., X_n$ are not independent, then
$$ K = 
\begin{bmatrix}
\sigma_1^2&\sigma_{12}&...&\sigma_{1n}\\
\sigma_{21}&\sigma_2^2&...&\sigma_{2n}\\
...&...&...&...\\
\sigma_{n1}&\sigma_{n2}&...&\sigma_n^2\\
\end{bmatrix}
$$
the matrix is Diagonally symmetric.

多元高斯分布的协方差矩阵必须是正定的，且任意变量的子集，其联合分布都服从多元高斯分布

## Infinite Dimensional Gaussian Distribution

Consider about 1-dimensional, 2-dimensional and multidimensional gaussian process

## Marginalization and conditioning 边缘概率与条件概率
These are two useful tools when trying to understand Gaussian Process.</br>
For $P_{X,Y}=$


## Gaussian Process
for $\textbf{x} = [x_1, x_2, ..., x_n]^T$, if $f(\textbf{x}) = [f(x_1), f(x_2), ..., f(x_n)]^T$ all obey Gaussian Distribution, then $f$ is a Gaussian Process.

logogram: $f(\textbf{x})$ ~ $N(\boldsymbol{\mu}(\textbf{x}), \kappa(\textbf{x},\textbf{x}))$

$\boldsymbol{\mu}(\textbf{x})$: Mean Function

$\kappa(\textbf{x},\textbf{x})$: Covariance Function(or Kernel Function)

a Gaussian Process is distinctively defined by one mean function and one covariance function.

通过定义的核函数和观测数值，我们可以估计函数取值范围

对于特定x，有一定取值范围

上式其实就是高斯过程回归的基本公式，首先有一个高斯过程先验分布，观测到一些数据（机器学习中的训练数据），基于先验和一定的假设（联合高斯分布）计算得到高斯过程后验分布的均值和协方差。

先验分布x, f(x), 观测数据x*, y* 
### Kernel Function
one typical kernel function is the Gaussian Kernel Function:
$K(x_i, x_j) = \sigma^2exp(-\frac{\Vert x_i-x_j\Vert_2^2 }{2l^2})$, 
where $\sigma$ and $l$ are hyperparameters.

To find the best $\sigma$ and $l$, we can maximize the **Marginal Log-Likelihood**, which is give as $\log p(\boldsymbol{y}|\sigma,l)=\log\boldsymbol{N}(\boldsymbol{0},K_{yy}(\sigma, l))=-\frac{1}{2}\boldsymbol{y}^TK_{yy}^{-1}\boldsymbol{y}-\frac{1}{2}\log|K_{yy}|-\frac{N}{2}\log(2\pi)$

## Reference
https://distill.pub/2019/visual-exploration-gaussian-processes/</br>
https://zhuanlan.zhihu.com/p/349600542</br>
https://zhuanlan.zhihu.com/p/75589452</br>
https://blog.csdn.net/shizheng_Li/article/details/144154902</br>