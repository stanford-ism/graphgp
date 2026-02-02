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

# %%
rng, key = jr.split(rng)

n_points = 1_000
d = 3
k = 8
n0 = k
points = jr.normal(key, (n_points, d))

def get_preceding_neighbors(points, k):
    pairwise_dists = jnp.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    mask = jnp.arange(n_points)[None, :] >= jnp.arange(n_points)[:, None]
    pairwise_dists = jnp.where(mask, jnp.inf, pairwise_dists)
    preceding_neighbors = jnp.argsort(pairwise_dists, axis=-1)[:, :k]
    return preceding_neighbors

def compute_offsets(depths):
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
    return offsets


# %% GraphGP example
graph = gp.build_graph(points, n0=n0, k=k, cuda=True)
graphgp_depths = gp.graph.compute_depths(graph.neighbors, n0=n0)
graphgp_batches = jnp.unique(graphgp_depths, return_counts=True)[1]


# %% Tree example, preceding neighbors
tree = jk.build_tree(points)
tree_points = tree.points[tree.indices]
tree_neighbors = get_preceding_neighbors(tree_points, k)[n0:]
tree_depths = gp.graph.compute_depths(tree_neighbors, n0=n0)
tree_batches = jnp.unique(tree_depths, return_counts=True)[1]

#%% Random order
random_neighbors = get_preceding_neighbors(points, k)[n0:]
random_depths = gp.graph.compute_depths(random_neighbors, n0=n0)
random_batches = jnp.unique(random_depths, return_counts=True)[1]

#%% Lexicographic order
lex_order = jnp.argsort(points[:,0])
lex_points = points[lex_order]
lex_order = jnp.argsort(lex_points[:,1])
lex_points = lex_points[lex_order]
lex_neighbors = get_preceding_neighbors(lex_points, k)[n0:]
lex_depths = gp.graph.compute_depths(lex_neighbors, n0=n0)
lex_batches = jnp.unique(lex_depths, return_counts=True)[1]

# %% mega graphgp example
points = jr.normal(rng, (10_000_000, d))
mega_graph = gp.build_graph(points, n0=n0, k=k, cuda=True)
mega_graphgp_depths = gp.graph.compute_depths(mega_graph.neighbors, n0=n0, cuda=True)
mega_graphgp_batches = jnp.unique(mega_graphgp_depths, return_counts=True)[1]

# %%
fig, ax = plt.subplots(figsize=(6,4), dpi=300)
ax.plot(graphgp_batches, label='GraphGP', c='k')
ax.plot(random_batches, label='Random')
ax.plot(tree_batches, label='Tree')
ax.plot(lex_batches, label='Lexicographic')
ax.plot(mega_graphgp_batches, label='GraphGP (10M points)', c='gray', zorder=-1)
ax.plot(jnp.arange(jnp.log2(n_points)), 2**jnp.arange(jnp.log2(n_points)), '--', label='Doubling', c='gray')
ax.set(yscale='log', xlabel='Level', ylabel='Size of level', title=f'Gaussian distribution of points (n={n_points}, d={d}, k={k})', xlim=(0, len(tree_batches)), ylim=(1, 1e3))
ax.legend()

# plt.savefig("level_sizes_by_order.png", bbox_inches='tight')
plt.show()


#%% Compare KL for different orders
k_scan = 2 ** jnp.linspace(0, 5, 6).astype(int)
# k_scan = jnp.array([2,4,8])

kl_matrix = jnp.zeros((len(k_scan), 4))

for order in ['graphgp', 'random', 'tree', 'lex']:
    for j, k in enumerate(k_scan):
        n0 = int(max(k_scan))
        if order == 'graphgp':
            graph = gp.build_graph(points, n0=n0, k=int(k), cuda=True)
            pts = graph.points
            neighbors = graph.neighbors
            offsets = graph.offsets
            indices = graph.indices
        elif order == 'tree':
            neighbors = get_preceding_neighbors(tree_points, int(k))[n0:]
            depths = gp.graph.compute_depths(neighbors, n0=n0)
            pts, indices, neighbors, depths = gp.graph.order_by_depth(tree_points, jnp.arange(n_points), neighbors, depths)
            offsets = compute_offsets(depths)
        elif order == 'lex':
            neighbors = get_preceding_neighbors(lex_points, int(k))[n0:]
            depths = gp.graph.compute_depths(neighbors, n0=n0)
            pts, indices, neighbors, depths = gp.graph.order_by_depth(lex_points, jnp.arange(n_points), neighbors, depths)
            offsets = compute_offsets(depths)
        elif order == 'random':
            neighbors = get_preceding_neighbors(points, int(k))[n0:]
            depths = gp.graph.compute_depths(neighbors, n0=n0)
            pts, indices, neighbors, depths = gp.graph.order_by_depth(points, jnp.arange(n_points), neighbors, depths)
            offsets = compute_offsets(depths)

        new_graph = gp.Graph(
            pts,
            neighbors,
            offsets,
        )

        covariance = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=10.0, r_min=1e-4, r_max=10, n_bins=1000, jitter=1e-4)

        K_true = gp.compute_cov_matrix(covariance, pts, pts)
        J_approx = jax.jacfwd(Partial(gp.generate, new_graph, covariance))(jnp.zeros((n_points,)))
        K_approx = J_approx @ J_approx.T

        solved = jnp.linalg.solve(K_approx, K_true)
        kl_per_dof = 1/2 * (jnp.trace(solved) - jnp.linalg.slogdet(solved)[1] - n_points)/n_points
        kl_matrix = kl_matrix.at[j, ['graphgp', 'tree', 'lex', 'random'].index(order)].set(kl_per_dof)
        print(f"Order={order}, k={k}, KL/dof={kl_per_dof:.3e}")

#%%
fig, ax = plt.subplots(figsize=(4,4), dpi=300)
for i, order in enumerate(['graphgp', 'tree', 'lex', 'random']):
    ax.plot(jnp.log(k_scan), kl_matrix[:,i], marker='o', label=f'{order}', c={'graphgp':'k', 'tree':'C1', 'lex':'C2', 'random':'C0'}[order])
ax.set(xlabel='k', ylabel='KL per DOF', title='Unit normal points, Matern (p=1)', xticks=jnp.log(k_scan), xticklabels=k_scan, yscale='log')
ax.legend()

# plt.savefig("kl_by_order_k.png", bbox_inches='tight')
plt.show()

# %% Compare KL for these orders
# new_graph = gp.build_graph(points, n0=n0, k=k, cuda=True)

tree_neighbors = get_preceding_neighbors(tree_points, 8)[n0:]
tree_depths = gp.graph.compute_depths(tree_neighbors, n0=n0)

new_graph = gp.Graph(
    tree_points,
    tree_neighbors,
    compute_offsets(tree_depths)
)
# new_graph = gp.Graph(
#     lex_points,
#     lex_neighbors,
#     compute_offsets(lex_depths)
# )
# new_graph = gp.Graph(
#     points,
#     random_neighbors,
#     compute_offsets(random_depths)
# )

covariance = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=10.0, r_min=1e-4, r_max=10, n_bins=1000, jitter=1e-4)

J_true = jax.jacfwd(Partial(gp.generate_dense, points, covariance))(jnp.zeros((n_points,)))
K_true = J_true @ J_true.T
J_approx = jax.jacfwd(Partial(gp.generate, new_graph, covariance))(jnp.zeros((n_points,)))
K_approx = J_approx @ J_approx.T

solved = jnp.linalg.solve(K_approx, K_true)
kl_per_dof = 1/2 * (jnp.trace(solved) - jnp.linalg.slogdet(solved)[1] - n_points)/n_points
print(kl_per_dof)
# %%
