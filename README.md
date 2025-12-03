"Gaussian processes don't scale."

***GraphGP***... \
✅ generates Gaussian process realizations with approximately stationary, decaying kernels \
✅ scales to billions of parameters with linear time and memory requirements \
✅ effortlessly handles arbitrary point distributions with large dynamic range \
✅ uses JAX, with a faster custom CUDA extension that supports derivatives \
✅ has an exact inverse and determinant available

The underlying theory and implementation is described in two upcoming papers. It is an evolution of Iterative Charted Refinement [[1](https://arxiv.org/abs/2206.10634)], which was first implemented in the [NIFTy](https://pypi.org/project/nifty/) package. The tree algorithms are inspired by two GPU-friendly approaches [[2](https://arxiv.org/abs/2211.00120), [3](https://arxiv.org/abs/2210.12859)] originally implemented in the [cudaKDTree](https://github.com/ingowald/cudaKDTree) library.

We wrote this software for applications in astrophysics, but we hope others across the physical sciences will find it useful! Please do not hesitate to open an issue or discussion for questions and feedback :)

Authors: Benjamin Dodge, Philipp Frank


## Usage

```python
import jax
import graphgp as gp

kp, kx = jax.random.split(jax.random.key(99))
points = jax.random.normal(kp, shape=(100_000, 2))
xi = jax.random.normal(kx, shape=(100_000,))

graph = gp.build_graph(points, n0=100, k=10)
covariance = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=1e1, n_bins=1_000, jitter=1e-4)
values = gp.generate(graph, covariance, xi)
```

## Installation
To install, use pip. The only dependency is JAX.

```python -m pip install graphgp```

For large problems, consider installing the custom CUDA extension as shown below, which will require CMake and the CUDA compiler (nvcc) installed on your system. It will take a moment to build and there may be rough edges, but memory and runtime requirements will be substantially improved. Please let us know if you encounter issues!

```python -m pip install graphgp[cuda]```


## Q&A
*How does it work?* \
The most straightforward way to generate a Gaussian Process realization at N arbitrary points is to construct a dense N x N covariance matrix, compute a matrix square root via Cholesky decomposition, and then apply it to a vector of white noise. This is equivalent to sequential generation of values, where each value is conditioned on all previous values using the Gaussian conditioning formulas. The main approximation made in GraphGP is to condition only on the values of k previously generated nearest neighbors. More details to come!

*Why am I getting NaNs?* \
Just as with a dense Cholesky decomposition, GraphGP can fail if the covariance matrix becomes singular due to finite precision arithmetic. For example, two points are so close together that their covariance is indistinguishable from their variance. A practical solution it to add "jitter" to the diagonal, as shown in the demo. Other options include reducing ``n0`` (singularity usually manifests in the dense Cholesky first), using 64-bit arithmetic, verifying that the covariance of the closest-spaced points can be represented for your choice of kernel, or increasing the number of bins for the discretized covariance. We are working to make this more user-friendly in the future.

*What is the difference between the pure JAX and custom CUDA versions?* \
The JAX version must store a (k+1) x (k+1) conditioning matrix for each point. The CUDA version generates these matrices on the fly and must only store the indices of k neighbors for each point. So we can expect roughly a factor of k better memory usage and runtime performance, depending on the exact setup.

@Philipp want to write a short paragraph explaining the context? Feel free to rephrase the question etc
*How does this work with Nifty?* \
GraphGP is not an inference package and will not help you fit your GP model to data. We encourage users to take advantage of Nifty's inference tools and GraphGP can serve as a drop-in replaement for ICR.