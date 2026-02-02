#%%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial

import graphgp as gp

import matplotlib.pyplot as plt

rng = jr.key(123)

# %%
d_scan = jnp.array([1,2,3,4,5,6])
k_scan = 2 ** jnp.linspace(0, 5, 6).astype(int)
print(k_scan)

#%%
kl_matrix = jnp.zeros((len(d_scan), len(k_scan)))

for i, d in enumerate(d_scan):
    for j, k in enumerate(k_scan):
        n_points = 1024

        points = jr.normal(rng, (n_points, d)) / jnp.sqrt(d)
        graph = gp.build_graph(points, n0=int(max(k_scan)), k=int(k), cuda=True)
        covariance = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=10.0, r_min=1e-5, r_max=1e2, n_bins=1000, jitter=1e-4)

        K_true = gp.compute_cov_matrix(covariance, points, points)
        J_approx = jax.jacfwd(Partial(gp.generate, graph, covariance))(jnp.zeros((n_points,)))
        K_approx = J_approx @ J_approx.T

        solved = jnp.linalg.solve(K_approx, K_true)
        kl_per_dof = 1/2 * (jnp.trace(solved) - jnp.linalg.slogdet(solved)[1] - n_points)/n_points

        # print(jnp.linalg.slogdet(K_true)[1], jnp.linalg.slogdet(K_approx)[1])

        kl_matrix = kl_matrix.at[i, j].set(kl_per_dof)
        print(f"d={d}, k={k}, KL/dof={kl_per_dof:.3e}")

# %% Plot
fig, ax = plt.subplots(figsize=(4,4), dpi=300)

for i, d in enumerate(d_scan):
    ax.plot(jnp.log(k_scan), kl_matrix[i,:], marker='o', label=f'd={d}', c=plt.get_cmap('inferno')(i / len(d_scan)))

ax.set(xlabel='k (number of neighbors)', ylabel='KL / dof', yscale='log', xticks=jnp.log(k_scan), xticklabels=k_scan, title='Unit normal points, Matern (p=1)')
ax.legend()

# plt.savefig("kl_by_d_k.png", bbox_inches='tight')
plt.show()
# %%
