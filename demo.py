# %%
import jax

jax.config.update("jax_enable_x64", True)
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
graph = gp.build_graph(points, n0=100, k=10)
covariance = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=10.0, n_bins=1000, jitter=1e-5)

# %%
# Generate a random GP realization and plot!
rng, key = jr.split(rng)
xi = jr.normal(key, (n_points,))
values = gp.generate(graph, covariance, xi)
assert jnp.sum(jnp.isnan(values)) == 0, "Generated values contain NaNs"

plt.scatter(*points.T, c=values, s=1)
plt.gca().set(aspect="equal", xlim=(-4, 4), ylim=(-4, 4))
plt.colorbar()
plt.show()
