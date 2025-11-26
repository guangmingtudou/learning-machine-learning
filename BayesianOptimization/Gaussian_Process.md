# Gaussian Process
learning from https://borgwang.github.io/ml/2019/07/28/gaussian-processes.html
## Univarient Gaussian Distribution
$p(x) = \frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$

the same as normal distribution

## Multivarient Gaussian Distribution
assume that $X_1, X_2, ..., X_n$ are independent variables, then

$p(x_1, x_2, ..., x_n) = \prod{p(x_i)} = \frac{1}{(2\pi)^{\frac{n}{2}}\sigma_1\sigma_2...\sigma_n}exp(-\frac{1}{2}[\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}+...+\frac{(x_n-\mu_n)^2}{\sigma_n^2}])$

let $\mathbf{x} - \boldsymbol{\mu} = [x_1-\mu_1, x_2-\mu_2, ..., x_n-\mu_n]^T$


$$ K = 
\begin{bmatrix}
\sigma_1^2&0&...&0\\
0&\sigma_2^2&...&0\\
...&...&...&...\\
0&0&...&\sigma_n^2\\
\end{bmatrix}
$$

then $\sigma_1\sigma_2...\sigma_n = |K|^{\frac{1}{2}}$

$\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}+...+\frac{(x_n-\mu_n)^2}{\sigma_n^2} = (\mathbf{x} - \boldsymbol{\mu})^TK(\mathbf{x} - \boldsymbol{\mu})$

Using these equations, we get $p(x_1, x_2, ..., x_n) = \frac{1}{(2\pi)^{\frac{n}{2}}|K|^{\frac{1}{2}}}exp(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^TK(\mathbf{x} - \boldsymbol{\mu}))$ 

logogram: x ~ N(\boldsymbol{\mu}, K)

## Gaussian Process
for $\textbf{x} = [x_1, x_2, ..., x_n]^T$, if $f(\textbf{x}) = [f(x_1), f(x_2), ..., f(x_n)]^T$ all obey Gaussian Distribution, then f is a Gaussian Process.

logogram: $f(\textbf{x})$ ~ $N(\boldsymbol{\mu}(\textbf{x}), \kappa(\textbf{x},\textbf{x}))$

$\boldsymbol{\mu}(\textbf{x})$: Mean Function

$\kappa(\textbf{x},\textbf{x})$: Covariance Function(or Kernel Function)

a Gaussian Process is distinctively defined by one mean function and one covariance function.

### Kernel Function
one typical kernel function is the Gaussian Kernel Function:
$K(x_i, x_j) = \sigma^2exp(-\frac{\Vert x_i-x_j\Vert_2^2 }{2l^2})$