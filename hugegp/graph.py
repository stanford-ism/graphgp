from dataclasses import dataclass, field
from typing import Callable, Tuple, Any, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial, register_dataclass
from jax import Array
from jax import lax

from .tree import build_tree, query_preceding_neighbors, query_offset_neighbors

try:
    import hugegp_cuda
    has_cuda = True
except ImportError:
    has_cuda = False


@register_dataclass
@dataclass
class Graph:
    """
    Nearest-neighbor dependency graph for Gaussian process generation.

    This object is provided for convenient use with ``generate`` and to easily mark ``offsets`` as static for JIT compilation.
    Users who are comfortable with the GraphGP algorithm should feel free to construct their own graphs however they like.
    We provide a function ``check_graph`` to verify that the graph can be used with other GraphGP components.
    The ``generate_dense`` and ``refine`` functions take raw arrays as arguments if users do not wish to use this object.

    Fields:
        points: Modeled points in tree order of shape ``(N, d)``.
        neighbors: Indices of the neighbors of shape ``(N - offsets[0], k)``.
        offsets: Tuple of length ``B`` representing the end index of each batch.
        indices: Original indices of the points of shape ``(N,)``. Can be ``None``.
    """

    points: Array
    neighbors: Array
    offsets: Tuple[int, ...] = field(metadata=dict(static=True))
    indices: Array | None = None


def check_graph(graph: Graph):
    """
    Verify the graph is valid for use with GraphGP.

    Requirements:
        1. Points are in topological order (i.e. neighbors come before in the order).
        2. Length of ``neighbors`` is less than length of ``points`` by ``offsets[0]``.
        3. Neighbors are not in the same batch as defined by ``offsets``.
        4. Last offset does not exceed the number of points.
    """
    points, neighbors, offsets = graph.points, graph.neighbors, graph.offsets
    offsets = jnp.asarray(offsets)

    # Ensure offsets are valid
    assert offsets[0] == len(points) - len(neighbors), "Neighbors should start from first offset"
    assert jnp.all(offsets[1:] > offsets[:-1]), "Offsets must be strictly increasing"
    assert offsets[-1] <= len(points), "Last offset must be less than or equal to the number of points"

    # Ensure topological order
    max_neighbors = jnp.max(neighbors, axis=1)
    index = jnp.arange(len(neighbors)) + offsets[0]
    ok = max_neighbors < index
    assert jnp.all(ok), "Points are not in topological order"

    # Ensure only coarse points
    offsets_index = jnp.searchsorted(offsets, index, side="right") - 1
    assert jnp.all(max_neighbors < offsets[offsets_index]), "Neighbors must not be in the same batch"


def build_graph(points: Array, *, n0: int, k: int, cuda: bool = False) -> Graph:
    """
    Build a graph such that all preceding neighbors are strictly included. This is the most accurate
    but is slower to build due to a single required serial pass through the points. The depth of the
    graph depends on the number of points and may impact parallel performance if too deep.

    Args:
        points: The input points of shape ``(N, d)``.
        n0: The number of initial points.
        k: The number of neighbors to include.
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.

    Returns:
        A ``Graph`` dataclass containing ``points``, ``neighbors``, ``offsets``, and ``indices``.
    """
    points, split_dims, indices = build_tree(points)
    neighbors = query_preceding_neighbors(points, split_dims, n0=n0, k=k, cuda=cuda)
    depths = compute_depths(neighbors, n0=n0, cuda=cuda)
    points, indices, neighbors, depths = order_by_depth(points, indices, neighbors, depths)
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(o) for o in offsets)
    return Graph(points, neighbors[:,::-1], offsets, indices)


def build_lazy_graph(points: Array, *, n0: int, k: int, factor: float = 1.5, max_batch: int | None = None) -> Graph:
    """
    Build a graph where preceding neighbors are skipped if they would fall inside the same batch.
    This is less accurate but faster to build and in some cases faster to use. It has the advantage
    that the depth of the graph is determined by the number of points and not the distribution.

    Args:
        points: The input points of shape ``(N, d)``.
        n0: The number of initial points.
        k: The number of neighbors to include.
        factor: The multiplicative factor by which to increase the number of points in each batch.
        max_batch: The maximum batch size. For accuracy set as small as possible without sacrificing performance.

    Returns:
        A ``Graph`` dataclass containing ``points``, ``neighbors``, ``offsets``, and ``indices``.
    """
    offsets = [n0]
    while offsets[-1] < len(points):
        next = offsets[-1] * factor
        if max_batch is not None:
            next = min(next, offsets[-1] + max_batch)
        offsets.append(int(min(next, len(points))))
    offsets = tuple(offsets)

    points, split_dims, indices = build_tree(points)
    neighbors, _ = query_offset_neighbors(points, split_dims, offsets=offsets, k=k)
    return Graph(points, neighbors[:,::-1], offsets, indices)


@Partial(jax.jit, static_argnames=("n0", "cuda"))
def compute_depths(neighbors, *, n0, cuda=False):
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        depths = hugegp_cuda.compute_depths(neighbors, n0=n0)
    else:
        depths = jnp.zeros(n0 + len(neighbors), dtype=jnp.int32)

        def update(i, depths):
            depths = depths.at[i].set(1 + jnp.max(depths[neighbors[i - n0]]))
            return depths

        depths = jax.lax.fori_loop(n0, n0 + len(neighbors), update, depths)
    return depths

@jax.jit
def order_by_depth(points, indices, neighbors, depths):
    n0 = len(points) - len(neighbors)
    order = jnp.argsort(depths)
    points, indices, depths = points[order], indices[order], depths[order]
    neighbors = neighbors[order[n0:] - n0]  # first n0 should stay in order
    inv_order = jnp.arange(len(points), dtype=int)
    inv_order = inv_order.at[order].set(inv_order)
    neighbors = inv_order[neighbors]
    return points, indices, neighbors, depths


# @jax.jit
# def order_by_depth_slow(points, indices, neighbors, depths):
#     n0 = len(points) - len(neighbors)
#     depth_order = jnp.argsort(depths)
#     points = points[depth_order]
#     indices = indices[depth_order]
#     depths = depths[depth_order]
#     neighbors = jnp.argsort(depth_order)[neighbors[depth_order[n0:] - n0]]  # first n0 stay in order
    # return points, indices, neighbors, depths

# def build_old_strict_graph(points, *, n_initial, k):
#     n_points = len(points)

#     # Build k-d tree
#     tree = jk.build_tree(points, cuda=True)
#     points = points[tree.indices]

#     # Query preceding neighbors in alternating order
#     neighbors = hugegp_cuda.query_preceding_neighbors_alt(points, tree.split_dims, k=k)

#     # Compute alternating order and reorder everything
#     alt_order = jnp.argsort(jax.vmap(compute_alt_index)(jnp.arange(n_points)))
#     points = points[alt_order]
#     neighbors = jnp.asarray(jnp.argsort(alt_order)[neighbors[alt_order]], dtype=jnp.uint32)

#     # Compute depth of each point in the graph for levels
#     levels = hugegp_cuda.compute_levels(neighbors, n_initial=n_initial)

#     # Sort by increasing level and compute level offsets
#     level_order = jnp.argsort(levels)
#     levels = levels[level_order]
#     points = points[level_order]
#     neighbors = jnp.asarray(jnp.argsort(level_order)[neighbors[level_order]], dtype=jnp.uint32)
#     neighbors = jnp.sort(neighbors, axis=1)
#     offsets = jnp.searchsorted(levels, jnp.arange(1, jnp.max(levels) + 1))
#     offsets = tuple(int(o) for o in offsets)

#     # Compute indices
#     indices = tree.indices[alt_order][level_order]
#     return (points, neighbors, offsets, indices)

# def build_old_graph(points, *, offsets, k, cuda=False):
#     n_points = len(points)

#     # Order points into alternating tree order
#     tree = jk.build_tree(points)
#     alt_order = jnp.argsort(jax.vmap(compute_alt_index)(jnp.arange(n_points)))
#     indices = tree.indices[alt_order]
#     points = points[indices]

#     # Construct neighbors array
#     neighbors = []
#     for i in range(len(offsets) - 1):
#         ni, _ = jk.build_and_query(points[: offsets[i]], points[offsets[i] : offsets[i + 1]], k=k, cuda=cuda)
#         neighbors.append(ni)
#     neighbors = jnp.concatenate(neighbors, axis=0)

#     return (points, neighbors, indices)


# def compute_alt_index(idx):
#     level = jnp.frexp(idx + 1)[1] - 1
#     level_idx = idx - ((1 << level) - 1)
#     return bit_reverse(level_idx, level) + ((1 << level) - 1)


# def bit_reverse(n, b):
#     n = jnp.asarray(n, dtype=jnp.uint32)
#     b = jnp.asarray(b, dtype=jnp.uint32)
#     n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1)
#     n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2)
#     n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4)
#     n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8)
#     n = (n >> 16) | (n << 16)
#     n = n >> (32 - b)
#     return n
