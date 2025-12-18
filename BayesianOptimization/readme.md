# Learning Bayesian Optimization

## why using Bayesian Optimization
1. The target function is not explicitly given.
2. Each sampling process costs a lot.

## the structure of Bayesian Optimization
Bayesian Optimization contains two parts: a Bayesian statistical model and an
acquisition function.

### Bayesian statistical model (or surrogate function)
It is used to estimate the target function based on the sampled points.

Possible model:
1. Gaussian Process

For Gaussina Process, please read Gaussian_Process.md.
### Acquisition Function
It is used to determine which point to explore.</br>
Possible funtions:
1. Expected Improvement (EI)

## how Bayesian Optimization works
The process can be roughly discribed as follows:

1. initiate the statistical model
2. sample
3. update the statistical model with the values got in step 2
4. get back to step 2 before reaching a terminal condition

# Reference
https://zhuanlan.zhihu.com/p/358606341
https://krasserm.github.io/2018/03/21/bayesian-optimization/