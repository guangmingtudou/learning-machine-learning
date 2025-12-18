# Gaussian Process
<font color='red'>Note</font>: To be honest, due to my weak understanding in Possibility, I can not really understand all the concepts, and it's hard to justify which concept is right since I've seen some different definitions on Gaussian Process. After a systematic learning in machine learning, I'll come back to fix some of the problems.
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
$$ \Sigma = 
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

Consider a multidimensional gaussian distribution with infinite variabless.

## Marginalization and conditioning 边缘概率与条件概率
These are two useful tools when trying to understand Gaussian Process.</br></br>
For $P_{X,Y}=\begin{bmatrix} X \\ Y \\\end{bmatrix}
$ ~ $N(\mu, \Sigma)=N(\begin{bmatrix} \mu_X \\ \mu_Y \\\end{bmatrix},\begin{bmatrix} \Sigma_{XX}\ \Sigma_{XY} \\ \Sigma_{YX}\ \Sigma_{YY} \\\end{bmatrix})$, where X and Y representing subsets of original random variables, with the properties $X$ ~ $N(\mu_X,\Sigma_{XX})$, $Y$ ~ $N(\mu_Y,\Sigma_{YY})$, which means that X and Y only depends on its own $\boldsymbol{\mu}$ and $\Sigma$</br></br>
**Marginalization**: $p_X(x) = \int_yp_{X,Y}(x,y)dy=\int_yp_{X|Y}(x|y)p_Y(y)dy$ due to Bayesian's equation.</br></br>
**Conditioning**: $X|Y$ ~ $N(\boldsymbol{\mu}_X+\Sigma_{XY}\Sigma_{YY}^{-1}(Y-\boldsymbol{\mu}_Y), \Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX})$ detailed explanation is not given, since I have little understanding of that right now

## Gaussian Process
**Definition**:  a Gaussian process is a collection of random variables, any finite number of which have (consistent) Gaussian distributions.

logogram: $f(\textbf{x})$ ~ $N(\boldsymbol{\mu}(\textbf{x}), \kappa(\textbf{x},\textbf{x}))$

$\boldsymbol{\mu}(\textbf{x})$: Mean Function

$\kappa(\textbf{x},\textbf{x})$: Covariance Function(or Kernel Function)

a Gaussian Process is distinctively defined by one mean function and one covariance function.

通过定义的核函数和观测数值，我们可以估计函数取值范围

对于特定x，有一定取值范围

上式其实就是高斯过程回归的基本公式，首先有一个高斯过程先验分布，观测到一些数据（机器学习中的训练数据），基于先验和一定的假设（联合高斯分布）计算得到高斯过程后验分布的均值和协方差。

先验分布x, f(x), 观测数据x*, y* 
### Kernel Function
Some typical kernel functions:</br>
1. Radical Basis Function: $K(x_i, x_j) = \sigma^2exp(-\frac{\Vert x_i-x_j\Vert_2^2 }{2l^2})$
2. Periodic Function: $K(x_i, x_j) = \sigma^2exp(-\frac{2sin^2(\pi|x_i-x_j|/p)}{l^2})$
3. Linear Function: $K(x_i, x_j) = \sigma^2_{b}+\sigma^2(x_i-c)(x_j-c)$
   
The things other than $x_i$ and $x_j$ are hyperparameters, like $\sigma, l, \sigma_b$ and $c$.</br></br>
The kernel functions can be roughly divided into stationary kernels and non-stationary kernels. For stationary ones, $K$ only relates to the relative position of $x_i$ and $x_j$, and non-stationary ones depend on the absolute location.
| stationary kernel | non-stationary kernel |
|---|---|
|RBF|Linear Function|
|Periodic Functoin||

### How to predict
1. we get the prior distribution.
2. from the prior distribution, we get the conditional probability of each point.

### Optimization
To find the best $\sigma$ and $l$ for RBF, we can maximize the **Marginal Log-Likelihood**, which is give as $\log p(\boldsymbol{y}|\sigma,l)=\log\boldsymbol{N}(\boldsymbol{0},K_{yy}(\sigma, l))=-\frac{1}{2}\boldsymbol{y}^TK_{yy}^{-1}\boldsymbol{y}-\frac{1}{2}\log|K_{yy}|-\frac{N}{2}\log(2\pi)$

## Reference
https://distill.pub/2019/visual-exploration-gaussian-processes/</br>
https://zhuanlan.zhihu.com/p/349600542</br>
https://zhuanlan.zhihu.com/p/75589452</br>
https://blog.csdn.net/shizheng_Li/article/details/144154902</br>
https://mlg.eng.cam.ac.uk/teaching/4f13/2425/gp.pdf</br>