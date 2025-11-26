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

### acquisition function
It is used to determine which point to explore.

## how Bayesian Optimization works
The process can be roughly discribed as follows:

1. initiate the statistical model
2. sample
3. update the statistical model with the values got in step 2
4. get back to step 2 before reaching a terminal condition
