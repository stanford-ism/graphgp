import jax.numpy as jnp

import jaxkd as jk
import hugegp_cuda


def build_graph(points, *, start_level, k):
    tree = jk.build_tree(points, cuda=True)
    indices = tree.indices
    neighbors = hugegp_cuda.query_coarse_neighbors(tree, k=k)
    n_levels = len(points).bit_length() - 1
    level_offsets = 1 << jnp.arange(start_level, n_levels + 1, dtype=jnp.uint32)
    return points, indices, neighbors, level_offsets
