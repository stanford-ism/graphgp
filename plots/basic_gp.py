#%%
import numpy as np
import matplotlib.pyplot as plt

import graphgp as gp

import jax
from jax.tree_util import Partial
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

rng = jr.key(123)

#%%
n_points = 1000
points = jnp.linspace(0, 1, n_points)[:,None]
graph = gp.build_graph(points, n0=100, k=16, cuda=True)
covariance = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=0.1, r_min=1e-4, r_max=1.0, n_bins=1000, jitter=1e-6)

plt.figure(figsize=(4,3), dpi=300)
rng, key = jr.split(rng)
xi = jr.normal(key, (n_points,))
values = gp.generate(graph, covariance, xi, cuda=True)
plt.plot(points, values, c='k')
plt.scatter(points[100], values[100], c='r')
plt.scatter(points[150], values[150], c='r')
plt.gca().set(xlabel='Space', ylabel='Density', xticks=[], yticks=[])

#%%
rng, key = jr.split(rng)
xi = jr.normal(key, (1000, n_points))
values = jax.vmap(Partial(gp.generate, graph, covariance, cuda=True))(xi)

#%%
plt.scatter(values[:,300], values[:,500], c='b', s=1)

plt.scatter(values[:,100], values[:,120], c='r', s=1)
plt.show()