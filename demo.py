# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import graphgp as gp

rng = jr.key(99)

# %%
# Generate some random points, build graph, and define covariance
n_points = 10_000
rng, key = jr.split(rng)
points = jr.normal(key, (n_points, 2))
graph = gp.build_graph(points, n0=500, k=16)
covariance = (0.3, 1.0)  # RBF: a single Gaussian (scales, amplitudes)

# Generate a random GP realization and plot!
rng, key = jr.split(rng)
xi = jr.normal(key, (n_points,))
values = gp.generate(graph, covariance, xi)

plt.scatter(*points.T, c=values, s=1)
plt.gca().set(aspect="equal", xlim=(-4, 4), ylim=(-4, 4))
plt.colorbar()
plt.show()

# %%
