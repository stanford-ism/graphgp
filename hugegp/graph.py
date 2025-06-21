import jax.numpy as jnp

import jaxkd as jk
import hugegp_cuda


def build_kd_graph(points, *, start_level, k):
    assert (1 << start_level) - 1 <= len(points), "initial points must be less than or equal to number of points"
    assert (1 << start_level) - 1 > k, "k must be less than or equal to the number of initial points"
    _, indices, split_dims = jk.build_tree(points)
    points = points[indices]
    n_levels = len(points).bit_length() - 1
    offsets = (1 << jnp.arange(start_level, n_levels + 1, dtype=jnp.uint32)) - 1
    offsets = tuple(int(o) for o in offsets)
    neighbors = hugegp_cuda.query_coarse_neighbors(points, split_dims, k=k)
    return points, offsets, neighbors
