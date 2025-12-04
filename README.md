"Gaussian processes don't scale."

***GraphGP***... \
✅ generates Gaussian process realizations with approximately stationary, decaying kernels \
✅ scales to billions of parameters with linear time and memory requirements \
✅ effortlessly handles arbitrary point distributions with large dynamic range \
✅ uses JAX, with a faster custom CUDA extension that supports derivatives \
✅ has an exact inverse and determinant available

The underlying theory and implementation is described in two upcoming papers. It is an evolution of Iterative Charted Refinement [[1](https://arxiv.org/abs/2206.10634)], which was first implemented in the [NIFTy](https://ift.pages.mpcdf.de/nifty/) package. The tree algorithms are inspired by two GPU-friendly approaches [[2](https://arxiv.org/abs/2211.00120), [3](https://arxiv.org/abs/2210.12859)] originally implemented in the [cudaKDTree](https://github.com/ingowald/cudaKDTree) library.

This software was written by [Benjamin Dodge](https://github.com/dodgebc) and [Philipp Frank](https://ph-frank.de) for applications in astrophysics, but we hope others across the physical sciences will find it useful! We thank [Susan Clark](https://clarkgroup.stanford.edu/) for guidance and support in developing the package, and are grateful for feedback from other members of the ISM group at Stanford. Please do not hesitate to open an issue or discussion for questions or problems :)

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
The most straightforward way to generate a Gaussian Process realization at $N$ arbitrary points is to construct a dense $N \times N$ covariance matrix, compute a matrix square root via Cholesky decomposition, and then apply it to a vector of white noise. This is equivalent to sequential generation of values, where each value is conditioned on all previous values using the Gaussian conditioning formulas. The main approximation made in GraphGP is to condition only on the values of $k$ previously generated nearest neighbors. More details to come!

*What is the graph?* \
GraphGP requires an array of `points`, an array of preceding `neighbors` for all but the first `n0` points for conditioning, and a tuple of `offsets` which specifies the batches of points that can be generated in parallel (i.e. no preceding neighbors must be within the same batch of their point). The `Graph` object is just a dataclass with these fields, plus an optional `indices` field specifying a permutation to apply to input white noise parameters `xi` and output `values`. Most users can just use the default `build_graph` and `generate` functions as shown above, but see the documentation for more options.

*How to I specify the covariance kernel?* \
GraphGP accepts a discretized covariance function $k(r)$, provided as a tuple of distance values and corresponding covariance values. These will be linearly interpolated when generating a GP realization. It is common to include `r=0` followed by logarithmic bins covering the minimum to the maximum distance between points, as demonstrated in `extras`. We use this discretized form for interoperability with the custom CUDA extension, though we may add more options in the future. Let us know what would be useful for you!

*The sharp bits -- Why am I getting NaNs?* \
Just as with a dense Cholesky decomposition, GraphGP can fail if the covariance matrix becomes singular due to finite precision arithmetic. For example, two points are so close together that their covariance is indistinguishable from their variance. A practical solution it to add "jitter" to the diagonal, as shown in the demo. Other options include reducing ``n0`` (singularity usually manifests in the dense Cholesky first), using 64-bit arithmetic, verifying that the covariance of the closest-spaced points can be represented for your choice of kernel, or increasing the number of bins for the discretized covariance. We are working to make this more user-friendly in the future.

*What is the difference between the pure JAX and custom CUDA versions?* \
The JAX version must store a $(k+1) \times (k+1)$ conditioning matrix for each point. The CUDA version generates these matrices on the fly and must only store the indices of $k$ neighbors for each point. We hence roughly expect a factor of $k$ better memory usage and runtime performance, depending on the exact setup.

*How do I `gp.fit` my model in GraphGP?* \
GraphGP is not an inference package on its own and hence will not fit your GP model to data. But GraphGP includes all necessary ingredients to do GP inference and regression: a fast Cholesky application, its inverse, and log-determinant. Hence it is straightforward to combine it with JAX-based optimization frameworks like [jaxopt](https://jaxopt.github.io/stable/) or [optax](https://optax.readthedocs.io/en/latest/). For advanced inference capabilities and Bayesian modeling we encourage users to take advantage of the inference tools available in [NIFTy](https://ift.pages.mpcdf.de/nifty/). Here GraphGP can serve as a drop-in replacement for ICR. Stay tuned for full `ift.Model` integration of GraphGP!