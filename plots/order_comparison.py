#%%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial

import json
import numpy as np

import graphgp as gp
import jaxkd as jk

import matplotlib.pyplot as plt

rng = jr.key(123)

plt.rcParams["font.family"] = "serif"

#%%
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

# %%
rng, key = jr.split(rng)

n_points = 10_000
d = 3
k = 16
n0 = 50
points = jr.normal(key, (n_points, d))
# points = jr.uniform(key, (n_points, d))

pairwise_dists = jnp.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
print(f"Minimum pairwise distance: {jnp.min(pairwise_dists[pairwise_dists > 0]):.3e}")
print(f"Maximum pairwise distance: {jnp.max(pairwise_dists):.3e}")

#%% Visualize field 
covariance = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=1.0, r_min=1e-5, r_max=300, n_bins=3000, jitter=1e-6)
xi = jr.normal(rng, (n_points,))
graph = gp.build_graph(points, n0=50, k=16, cuda=False)
values = gp.generate(graph, covariance, xi)

plt.scatter(points[:,0], points[:,1], c=values, cmap='viridis', s=4)
plt.colorbar()
plt.show()


# %% GraphGP example
graph = gp.build_graph(points, n0=n0, k=k, cuda=False)
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

#%% Maxmin order
points_np = np.array(points)
start_idx = int(np.argmin(np.linalg.norm(points_np - points_np.mean(axis=0), axis=-1)))
min_dists = np.linalg.norm(points_np - points_np[start_idx], axis=-1)
min_dists[start_idx] = -np.inf
maxmin_order = [start_idx]
for _ in range(n_points - 1):
    next_idx = int(np.argmax(min_dists))
    maxmin_order.append(next_idx)
    min_dists = np.minimum(min_dists, np.linalg.norm(points_np - points_np[next_idx], axis=-1))
    min_dists[next_idx] = -np.inf
maxmin_order = jnp.array(maxmin_order)
maxmin_points = points[maxmin_order]
maxmin_neighbors = get_preceding_neighbors(maxmin_points, k)[n0:]
maxmin_depths = gp.graph.compute_depths(maxmin_neighbors, n0=n0)
maxmin_batches = jnp.unique(maxmin_depths, return_counts=True)[1]

#%% Coordinate order
coord_order = jnp.argsort(points[:,0])
coord_points = points[coord_order]
coord_neighbors = get_preceding_neighbors(coord_points, k)[n0:]
coord_depths = gp.graph.compute_depths(coord_neighbors, n0=n0)
coord_batches = jnp.unique(coord_depths, return_counts=True)[1]

#%% Diagnostic: ordering up to index
# diag_order = 'maxmin'  # 'graphgp', 'tree', 'random', 'maxmin', 'coord'
# n_show = 50
# diag_pts = {'graphgp': np.array(graph.points), 'tree': np.array(tree_points),
#             'random': points_np, 'maxmin': points_np[np.array(maxmin_order)],
#             'coord': np.array(coord_points)}[diag_order]
# fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
# ax.scatter(points_np[:, 0], points_np[:, 1], c='lightgray', s=4, zorder=0)
# sc = ax.scatter(diag_pts[:n_show, 0], diag_pts[:n_show, 1],
#                 c=np.arange(n_show), cmap='viridis', s=12, zorder=1)
# plt.colorbar(sc, ax=ax, label='Order index')
# ax.set(title=f'{diag_order} order (first {n_show} points)', aspect='equal')
# plt.show()

#%% Compare KL for different orders
k_scan = jnp.array([1,2,4,8,16,32])
# k_scan = jnp.array([2,4,8])
p_scan = [0, 1]
orders = ['graphgp', 'tree', 'random', 'coord', 'maxmin']

kl_matrix = jnp.zeros((len(p_scan), len(k_scan), len(orders)))

for pi, p in enumerate(p_scan):
    for order in orders:
        for j, k in enumerate(k_scan):
            n0 = 50
            if order == 'graphgp':
                graph = gp.build_graph(points, n0=n0, k=int(k), cuda=False)
                pts = graph.points
                neighbors = graph.neighbors
                offsets = graph.offsets
                indices = graph.indices
            elif order == 'tree':
                neighbors = get_preceding_neighbors(tree_points, int(k))[n0:]
                depths = gp.graph.compute_depths(neighbors, n0=n0)
                pts, indices, neighbors, depths = gp.graph.order_by_depth(tree_points, jnp.arange(n_points), neighbors, depths)
                offsets = compute_offsets(depths)
            elif order == 'maxmin':
                neighbors = get_preceding_neighbors(maxmin_points, int(k))[n0:]
                depths = gp.graph.compute_depths(neighbors, n0=n0)
                pts, indices, neighbors, depths = gp.graph.order_by_depth(maxmin_points, jnp.arange(n_points), neighbors, depths)
                offsets = compute_offsets(depths)
            elif order == 'coord':
                neighbors = get_preceding_neighbors(coord_points, int(k))[n0:]
                depths = gp.graph.compute_depths(neighbors, n0=n0)
                pts, indices, neighbors, depths = gp.graph.order_by_depth(coord_points, jnp.arange(n_points), neighbors, depths)
                offsets = compute_offsets(depths)
            elif order == 'random':
                neighbors = get_preceding_neighbors(points, int(k))[n0:]
                depths = gp.graph.compute_depths(neighbors, n0=n0)
                pts, indices, neighbors, depths = gp.graph.order_by_depth(points, jnp.arange(n_points), neighbors, depths)
                offsets = compute_offsets(depths)

            covariance = gp.extras.matern_kernel(p=p, variance=1.0, cutoff=10.0, r_min=1e-5, r_max=10, n_bins=3000, jitter=1e-6)

            new_graph = gp.Graph(pts, neighbors, offsets)
            K_true = gp.compute_cov_matrix(covariance, pts, pts)
            J_approx = jax.jacfwd(Partial(gp.generate, new_graph, covariance))(jnp.zeros((n_points,)))
            K_approx = J_approx @ J_approx.T

            solved = jnp.linalg.solve(K_approx, K_true)
            kl_per_dof = 1/2 * (jnp.trace(solved) - jnp.linalg.slogdet(solved)[1] - n_points)/n_points
            kl_matrix = kl_matrix.at[pi, j, orders.index(order)].set(kl_per_dof)
            print(f"p={p}, order={order}, k={k}, KL/dof={kl_per_dof:.3e}")

#%% Save results
_out = '/users/bendodge/graphgp/output/order_comparison.json'
with open(_out, 'w') as _f:
    json.dump({
        'graphgp_batches': np.array(graphgp_batches).tolist(),
        'tree_batches':    np.array(tree_batches).tolist(),
        'random_batches':  np.array(random_batches).tolist(),
        'coord_batches':   np.array(coord_batches).tolist(),
        'maxmin_batches':  np.array(maxmin_batches).tolist(),
        'kl_matrix': np.array(kl_matrix).tolist(),
        'k_scan': np.array(k_scan).tolist(),
        'p_scan': list(p_scan),
        'orders': list(orders),
    }, _f)
print(f"Saved to {_out}")
raise SystemExit(0)

#%% Combined figure
def draw_tree(ax, positions, edges, color):
    ax.axis('off')
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2], c=color, lw=1.5, zorder=1)
    ax.scatter([positions[n][0] for n in positions],
               [positions[n][1] for n in positions],
               s=150, c=color, zorder=2, linewidths=0)
    for node, (x, y) in positions.items():
        ax.text(x, y, str(node), ha='center', va='center',
                color='white', fontsize=7, fontweight='bold', zorder=3)

tree_edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(3,7),(3,8),(4,9)]
graphgp_edges = [(0,1),(0,2),(1,3),(1,5),(2,4),(2,6),(3,7),(5,9),(4,8)]

_out = '../output/order_comparison.json'
with open(_out) as _f:
    _r = json.load(_f)
graphgp_batches = np.array(_r['graphgp_batches'])
tree_batches    = np.array(_r['tree_batches'])
random_batches  = np.array(_r['random_batches'])
coord_batches   = np.array(_r['coord_batches'])
maxmin_batches  = np.array(_r['maxmin_batches'])
kl_matrix = jnp.array(_r['kl_matrix'])
k_scan    = jnp.array(_r['k_scan'])
p_scan    = _r['p_scan']
orders    = _r['orders']

plot_order  = ['graphgp', 'tree', 'random', 'coord', 'maxmin']
plot_colors = {'graphgp': '#4477AA', 'tree': '#228833', 'random': '#BBBBBB', 'coord': '#CCBB44', 'maxmin': '#AA3377'}
plot_labels = {'graphgp': 'GraphGP', 'tree': 'Tree', 'random': 'Random', 'coord': 'Coordinate', 'maxmin': 'Maxmin'}
plot_zorders = {'graphgp': 5, 'tree': 4, 'maxmin': 3, 'random': 2, 'coord': 1}

# Compact positions: level-2 spacing 0.55
L2c = [0.40, 0.95, 1.50, 2.05]
tree_pos_c = {
    0: (1.225, 3.0),
    1: (0.675, 2.0), 2: (1.775, 2.0),
    3: (0.40, 1.0), 4: (0.95, 1.0), 5: (1.50, 1.0), 6: (2.05, 1.0),
    7: (0.26, 0.0), 8: (0.54, 0.0), 9: (0.81, 0.0),
}
graphgp_pos_c = {
    0: (1.225, 3.0),
    1: (0.675, 2.0), 2: (1.775, 2.0),
    3: (0.40, 1.0), 5: (0.95, 1.0), 4: (1.50, 1.0), 6: (2.05, 1.0),
    7: (0.26, 0.0), 9: (0.81, 0.0), 8: (1.36, 0.0),
}

fig = plt.figure(figsize=(8, 3), dpi=300)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 2, 1.1],
                      left=0.07, right=0.98, top=0.91, bottom=0.16,
                      hspace=0.45, wspace=0.25)
ax_g  = fig.add_subplot(gs[0, 0])
ax_t  = fig.add_subplot(gs[1, 0])
for _ax in [ax_g, ax_t]:
    _p = _ax.get_position()
    _ax.set_position([_p.x0 + 0.03, _p.y0, _p.width, _p.height])
ax_b  = fig.add_subplot(gs[:, 1])
ax_kl = fig.add_subplot(gs[:, 2])

for ax, pos, edges, color, title in [
    (ax_g, graphgp_pos_c, graphgp_edges, plot_colors['graphgp'], 'GraphGP order'),
    (ax_t, tree_pos_c, tree_edges, plot_colors['tree'], 'Tree order'),
]:
    draw_tree(ax, pos, edges, color)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = (min(xs) + max(xs)) / 2
    hw = (max(xs) - min(xs)) / 2 + 0.22
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(min(ys)-0.45, max(ys)+0.45)
    ax.text(0.5, 1.0, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=plt.rcParams['axes.titlesize'])

ax_b.plot(graphgp_batches, label='GraphGP', c=plot_colors['graphgp'], zorder=plot_zorders['graphgp'])
ax_b.plot(tree_batches, label='Tree', c=plot_colors['tree'], zorder=plot_zorders['tree'])
ax_b.plot(random_batches, label='Random', c=plot_colors['random'], zorder=plot_zorders['random'])
ax_b.plot(coord_batches, label='Coord', c=plot_colors['coord'], zorder=plot_zorders['coord'])
ax_b.plot(maxmin_batches, label='Max-min', c=plot_colors['maxmin'], zorder=plot_zorders['maxmin'])
ax_b.set_box_aspect(1)
ax_b.set(yscale='log', xlabel='Batch', ylabel='Batch size',
         xlim=(0, 320), ylim=(1, 1e3), title='Parallelism by order')
ax_b.legend(fontsize=8, loc='upper right', frameon=False, bbox_to_anchor=(1.02, 1.01))

for order in plot_order:
    i = orders.index(order)
    for pi, p in enumerate(p_scan):
        ls = '-' if p == 0 else '--'
        ax_kl.plot(jnp.log(k_scan), kl_matrix[pi, :, i], marker='o', ms=3, ls=ls,
                   #label=plot_labels[order] if pi == 0 else None,
                   c=plot_colors[order], zorder=plot_zorders[order])
ax_kl.plot([], [], color='gray', ls='-', label=r'$\nu=1/2$')
ax_kl.plot([], [], color='gray', ls='--', label=r'$\nu=3/2$')
ax_kl.set(xlabel='Neighbors', ylabel='KL / dof', title='Accuracy by order',
          xticks=jnp.log(k_scan), xticklabels=k_scan, yscale='log', ylim=(1e-3, 3))
ax_kl.legend(fontsize=8, frameon=False, loc='lower left')

plt.savefig("../output/order_comparison.pdf", bbox_inches='tight')
# plt.savefig("../../Overleaf/graphgp-pai26/figures/order_comparison.pdf", bbox_inches='tight')
plt.show()