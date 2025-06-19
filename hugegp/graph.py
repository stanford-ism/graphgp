import jax.numpy as jnp

import jaxkd as jk
import hugegp_cuda


def build_kd_graph(points, *, start_level, k):
    _, indices, split_dims = jk.build_tree(points, cuda=True)
    points = points[indices]
    neighbors = hugegp_cuda.query_coarse_neighbors(points, split_dims, k=k)
    n_levels = len(points).bit_length() - 1
    level_offsets = (1 << jnp.arange(start_level, n_levels + 1, dtype=jnp.uint32)) - 1
    level_offsets = tuple(int(o) for o in level_offsets)
    return points, neighbors, level_offsets
