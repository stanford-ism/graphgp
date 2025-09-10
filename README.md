"Gaussian processes don't scale," they said.

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

graph = gp.build_graph(points, n_initial=1000, k=10)
covariance = gp.MaternCovariance(p=0)
values = jax.jit(gp.generate)(graph, covariance, xi)
```

## Q&A
