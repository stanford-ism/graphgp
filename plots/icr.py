#%%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial

import graphgp as gp
import jaxkd as jk

import matplotlib.pyplot as plt

rng = jr.key(123)

def compute_offsets(depths):
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
    return offsets

#%% Dense covariance, sparse precision
k = 3
n0 = 4
points = jnp.logspace(0, 1, 256)[:,None]
graph = gp.build_graph(points, n0=n0, k=k, cuda=True)

covariance = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=1.0, r_min=1e-4, r_max=1e3, n_bins=1000, jitter=1e-6)

J = jax.jacfwd(Partial(gp.generate, graph, covariance))(jnp.zeros((len(points),)))
K = J @ J.T

plt.imshow(K, vmin=0, vmax=1, cmap='inferno')
plt.gca().set(title=f'GraphGP covariance (k={k})')
plt.colorbar()
plt.show()

plt.imshow(jnp.abs(jnp.linalg.inv(K)), vmin=0, vmax=0.1, cmap='gray_r')
plt.gca().set(title=f'GraphGP precision (k={k})')
# plt.colorbar()
plt.show()

#%% GraphGP
k = 3
n0 = 4
points = jnp.logspace(0, 1, 256)[:,None]
graph = gp.build_graph(points, n0=n0, k=k, cuda=True)

covariance = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=1.0, r_min=1e-4, r_max=1e3, n_bins=1000, jitter=1e-6)

J = jax.jacfwd(Partial(gp.generate, graph, covariance))(jnp.zeros((len(points),)))
K = J @ J.T

plt.imshow(K, vmin=0, vmax=1, cmap='inferno')
plt.gca().set(title=f'GraphGP covariance (k={k})')
plt.colorbar()
plt.show()

K_dense = gp.compute_cov_matrix(covariance, points, points)
plt.imshow(K - K_dense, vmin=-0.1, vmax=0.1, cmap='PiYG')
plt.gca().set(title=f'|Truth - GraphGP| covariance (k={k})')
plt.colorbar()
plt.show()

solved = jnp.linalg.solve(K, K_dense)
kl_per_dof = 1/2 * (jnp.trace(solved) - jnp.linalg.slogdet(solved)[1] - len(points))/len(points)
print(kl_per_dof)

# %% ICR analogue
k = 3
n0 = 4
tree = jk.build_tree(points)
tree_points = tree.points[tree.indices]
neighbors = []

for level in [2, 3, 4, 5, 6, 7]:
    coarse_points = tree_points[: 2**int(level)]
    fine_points = tree_points[2**int(level) : 2**int(level + 1)]
    neighbors.append(jk.build_and_query(coarse_points, fine_points, k=k)[0])

neighbors = jnp.vstack(neighbors)
depths = gp.graph.compute_depths(neighbors, n0=n0)
offsets = compute_offsets(depths)
icr_graph = gp.Graph(tree_points, neighbors, offsets, tree.indices)

J = jax.jacfwd(Partial(gp.generate, icr_graph, covariance))(jnp.zeros((len(points),)))
K = J @ J.T

plt.imshow(K, vmin=0, vmax=1, cmap='inferno')
plt.gca().set(title=f'ICR covariance (k={k})')
plt.colorbar()
plt.show()

K_dense = gp.compute_cov_matrix(covariance, points, points)
plt.imshow(K - K_dense, vmin=-0.1, vmax=0.1, cmap='PiYG')
plt.gca().set(title=f'|Truth - ICR| covariance (k={k})')
plt.colorbar()
plt.show()

solved = jnp.linalg.solve(K, K_dense)
kl_per_dof = 1/2 * (jnp.trace(solved) - jnp.linalg.slogdet(solved)[1] - len(points))/len(points)
print(kl_per_dof)

# %%
