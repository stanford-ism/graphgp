import jax
import jax.numpy as jnp
from jax.tree_util import Partial

import jaxkd as jk
import hugegp_cuda


def verify_topological_order(neighbors, n_initial):
    max_neighbors = jnp.max(neighbors, axis=1)
    indices = jnp.arange(len(neighbors))
    ok = (max_neighbors < indices) | (indices < n_initial)
    return jnp.all(ok)


def verify_valid_offsets(neighbors, offsets):
    offsets = offsets + (len(neighbors),)
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i + 1]
        level_neighbors = neighbors[start:end]
        if jnp.any(level_neighbors >= start):
            return False
    return True

def build_graph(points, *, n_initial, k):
    # Build k-d tree
    tree = jk.build_tree(points, cuda=True)
    points = points[tree.indices]

    # Query preceding neighbors in alternating order
    neighbors = hugegp_cuda.query_preceding_neighbors_alt(points, tree.split_dims, k=k)

    # Compute alternating order and reorder everything
    alt_order = jnp.argsort(jax.vmap(alt_index)(jnp.arange(len(points))))
    points = points[alt_order]
    neighbors = jnp.asarray(jnp.argsort(alt_order)[neighbors[alt_order]], dtype=jnp.uint32)
    # assert verify_topological_order(neighbors, n_initial), "Not topologically ordered"

    # Compute depth of each point in the graph for levels
    levels = hugegp_cuda.compute_levels(neighbors, n_initial=n_initial)

    # Sort by increasing level and compute level offsets
    level_order = jnp.argsort(levels)
    levels = levels[level_order]
    points = points[level_order]
    neighbors = jnp.asarray(jnp.argsort(level_order)[neighbors[level_order]], dtype=jnp.uint32)
    offsets = jnp.searchsorted(levels, jnp.arange(1, jnp.max(levels) + 1))
    offsets = tuple(int(o) for o in offsets)
    # assert verify_valid_offsets(neighbors, offsets), "Invalid level offsets"

    # Compute indices so points[indices] = original points
    indices = jnp.argsort(jnp.arange(len(points))[tree.indices][alt_order][level_order])
    return (points, neighbors, offsets), indices


def make_binary_offsets(start_level, n_points):
    n_levels = n_points.bit_length() - 1
    offsets = (1 << jnp.arange(start_level, n_levels + 1, dtype=jnp.uint32)) - 1
    offsets = tuple(int(o) for o in offsets)
    return offsets


@Partial(jax.jit, static_argnames=("n_initial",))
def compute_levels(neighbors, *, n_initial):
    levels = jnp.zeros(len(neighbors), dtype=jnp.int32)

    def update(i, levels):
        levels = levels.at[i].set(1 + jnp.max(levels[neighbors[i]]))
        return levels

    levels = jax.lax.fori_loop(n_initial, len(neighbors), update, levels)
    return levels


def alt_index(idx):
    level = jnp.frexp(idx + 1)[1] - 1
    level_idx = idx - ((1 << level) - 1)
    return bit_reverse(level_idx, level) + ((1 << level) - 1)


def bit_reverse(n, b):
    n = jnp.asarray(n, dtype=jnp.uint32)
    b = jnp.asarray(b, dtype=jnp.uint32)
    n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1)
    n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2)
    n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4)
    n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8)
    n = (n >> 16) | (n << 16)
    n = n >> (32 - b)
    return n


# def make_graph(points, *, n_initial, k):

#     # build k-d tree order
#     tree = jk.build_tree(points)
#     points = points[tree.indices]
#     neighbors = hugegp_cuda.query_previous_neighbors(points, tree.split_dims, k=k)

#     # fill levels sequentially
#     levels = compute_dag_levels(neighbors, n_initial=n_initial)

#     # sort by level
#     order = jnp.argsort(levels)
#     levels = levels[order]
#     points = points[order]
#     neighbors = jnp.argsort(order)[neighbors[order]]

#     # determine level offsets
#     offsets = jnp.searchsorted(levels, jnp.arange(1, jnp.max(levels) + 1))
#     offsets = tuple(int(o) for o in offsets)

#     return (points, offsets, neighbors)


# def make_valid_offsets(neighbors, n_initial):
#     max_neighbors = jnp.max(neighbors, axis=1)
#     order = jnp.argsort(max_neighbors)
#     max_neighbors = max_neighbors[order]

#     offsets = [n_initial]
#     while offsets[-1] < len(neighbors):
#         next = jnp.searchsorted(max_neighbors, offsets[-1])
#         offsets.append(int(next))
#     offsets = offsets[:-1]  # remove last offset which is always len(neighbors)
#     offsets = tuple(int(o) for o in offsets)
#     return offsets, order


# def build_kd_graph(points, *, start_level, k):
#     assert (1 << start_level) - 1 <= len(points), (
#         "initial points must be less than or equal to number of points"
#     )
#     assert (1 << start_level) - 1 > k, (
#         "k must be less than or equal to the number of initial points"
#     )
#     _, indices, split_dims = jk.build_tree(points)
#     points = points[indices]
#     neighbors = hugegp_cuda.query_coarse_neighbors(points, split_dims, k=k)
#     offsets = make_binary_offsets(start_level, len(points))
#     return points, offsets, neighbors


# def build_custom_graph(points, offsets, *, k):
#     _, indices, split_dims = jk.build_tree(points, cuda=True)
#     points = points[indices]
#     order = jnp.argsort(tree_reorder(jnp.arange(len(indices), dtype=jnp.uint32)))
#     points = points[order]
#     neighbors = query_coarse_neighbors_general(points, offsets, k=k)
#     assert verify_graph(points, offsets, neighbors)
#     return (points, offsets, jnp.asarray(neighbors, dtype=jnp.uint32)), order


# def query_coarse_neighbors_general(points, offsets, k):
#     neighbors = [jnp.zeros((offsets[0], k), dtype=jnp.int32)]
#     offsets = offsets + (len(points),)
#     for i in range(len(offsets) - 1):
#         start = offsets[i]
#         end = offsets[i + 1]
#         level_neighbors, _ = jk.build_and_query(points[:start], points[start:end], k=k, cuda=True)
#         neighbors.append(level_neighbors)
#     neighbors = jnp.concatenate(neighbors)
#     return neighbors
