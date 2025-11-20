"Gaussian processes don't scale."

***GraphGP***... \
✅ generates Gaussian process realizations with approximately stationary, decaying kernels \
✅ scales to billions of parameters with linear time and memory requirements \
✅ effortlessly handles arbitrary point distributions with large dynamic range \
✅ uses JAX, with a faster CUDA extension that supports derivatives \
✅ has an exact inverse and determinant available


## Usage

```python
import jax
import graphgp as gp

kp, kx = jax.random.split(jax.random.key(99))
points = jax.random.normal(kp, shape=(100_000, 2))
xi = jax.random.normal(kx, shape=(100_000,))

graph = gp.build_graph(points, n0=1000, k=10)
covariance = gp.compute_matern_covariance_discrete(p=0, r_min=1e-3, r_max=1e3, n_bins=1_000)
values = gp.generate(graph, covariance, xi)
```

## Installation
To install, use pip. The only dependency is JAX.

```python -m pip install graphgp```

For large problems, it is recommended to install the custom CUDA extension as shown below, which will require CMake and the CUDA compiler (nvcc) installed on your system. It will take a moment to build and there may be rough edges.

```python -m pip install graphgp[cuda]```


## Q&A
