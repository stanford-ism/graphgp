#%%
import json
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

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

plot_colors  = {'graphgp': '#4477AA', 'tree': '#228833', 'random': '#BBBBBB', 'coord': '#CCBB44', 'maxmin': '#AA3377'}
plot_zorders = {'graphgp': 5, 'tree': 4, 'maxmin': 3, 'random': 2, 'coord': 1}

#%% Parallelism by order
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

ax.plot(graphgp_batches, label='GraphGP',    c=plot_colors['graphgp'], zorder=plot_zorders['graphgp'])
ax.plot(tree_batches,    label='Tree',        c=plot_colors['tree'],    zorder=plot_zorders['tree'])
ax.plot(random_batches,  label='Random',      c=plot_colors['random'],  zorder=plot_zorders['random'])
ax.plot(coord_batches,   label='Coord',       c=plot_colors['coord'],   zorder=plot_zorders['coord'])
ax.plot(maxmin_batches,  label='Max-min',     c=plot_colors['maxmin'],  zorder=plot_zorders['maxmin'])

ax.set(yscale='log', xlabel='Batch', ylabel='Batch size',
       xlim=(0, 320), ylim=(1, 1e3))
ax.legend(fontsize=8, loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('../output/parallelism_by_order.pdf', bbox_inches='tight')
plt.show()

#%% Tree comparison
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

tree_edges    = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(3,7),(3,8),(4,9)]
graphgp_edges = [(0,1),(0,2),(1,3),(1,5),(2,4),(2,6),(3,7),(5,9),(4,8)]

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

fig, (ax_t, ax_g) = plt.subplots(2, 1, figsize=(2,3), dpi=300)

for ax, pos, edges, color, title in [
    (ax_t, tree_pos_c,    tree_edges,    plot_colors['tree'],    'Tree order'),
    (ax_g, graphgp_pos_c, graphgp_edges, plot_colors['graphgp'], 'GraphGP order'),
]:
    draw_tree(ax, pos, edges, color)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = (min(xs) + max(xs)) / 2
    hw = (max(xs) - min(xs)) / 2 + 0.22
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(min(ys) - 0.45, max(ys) + 0.45)
    ax.text(0.5, 1.0, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=plt.rcParams['axes.titlesize'])

plt.tight_layout()
plt.savefig('../output/tree_comparison.pdf', bbox_inches='tight')
plt.show()

#%% Accuracy by order
plot_order = ['graphgp', 'tree', 'random', 'coord', 'maxmin']

fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

for order in plot_order:
    i = orders.index(order)
    for pi, p in enumerate(p_scan):
        ls = '-' if p == 0 else '--'
        ax.plot(jnp.log(k_scan), kl_matrix[pi, :, i], marker='o', ms=3, ls=ls,
                c=plot_colors[order], zorder=plot_zorders[order])
plot_labels = {'graphgp': 'GraphGP', 'tree': 'Tree', 'random': 'Random', 'coord': 'Coord', 'maxmin': 'Max-min'}
for order in plot_order:
    ax.plot([], [], color=plot_colors[order], label=plot_labels[order])
ax.plot([], [], color='gray', ls='-', label=r'$\nu=1/2$')
ax.plot([], [], color='gray', ls='--', label=r'$\nu=3/2$')
ax.set(box_aspect=1, xlabel='Neighbors', ylabel='KL / dof',
       xticks=jnp.log(k_scan), xticklabels=k_scan, yscale='log', ylim=(1e-3, 3))
ax.legend(fontsize=8, frameon=False, loc='lower left')

plt.tight_layout()
plt.savefig('../output/accuracy_by_order.pdf', bbox_inches='tight')
plt.show()

# %%
