from dataclasses import dataclass, field
from typing import Callable, Tuple, Any, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial, register_dataclass
from jax import Array
from jax import lax

import numpy as np

from .tree import build_tree, query_preceding_neighbors

try:
    import graphgp_cuda

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
        4. Offsets are increasing and do not exceed the number of points.
    """
    points, neighbors, offsets = graph.points, graph.neighbors, graph.offsets
    offsets = jnp.asarray(offsets)

    # Ensure offsets are valid
    assert offsets[0] == len(points) - len(neighbors), "Neighbors should start from first offset"
    assert jnp.all(offsets[1:] >= offsets[:-1]), "Offsets must be non-decreasing"
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
    Build a graph where each point depends on its ``k`` nearest neighbors which precede it in a k-d tree ordering (the original point order does not matter).

    Args:
        points: The input points of shape ``(N, d)``.
        n0: The number of initial points.
        k: The number of neighbors to include.
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.

    Returns:
        A ``Graph`` dataclass containing ``points``, ``neighbors``, ``offsets``, and ``indices``.
    """
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        points, indices, neighbors, depths = graphgp_cuda.build_graph(points, n0=n0, k=k)
    else:
        points, split_dims, indices = build_tree(points)
        neighbors = query_preceding_neighbors(points, split_dims, n0=n0, k=k)
        depths = compute_depths(neighbors, n0=n0)
        points, indices, neighbors, depths = order_by_depth(points, indices, neighbors, depths)
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
    # TODO: neighbors[:, ::-1] from far to close feels more stable but not sure if it matters
    return Graph(points, neighbors, offsets, indices)


@Partial(jax.jit, static_argnames=("n0", "cuda"))
def compute_depths(neighbors, *, n0, cuda=False):
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        depths = graphgp_cuda.compute_depths_parallel(neighbors, n0=n0)
    else:
        depths = jnp.zeros(n0 + len(neighbors), dtype=jnp.int32)

        def update(carry):
            old_depths, depths = carry
            new_depths = depths.at[jnp.arange(n0, len(depths))].set(1 + jnp.max(depths[neighbors], axis=1))
            return depths, new_depths

        def cond(carry):
            old_depths, depths = carry
            return jnp.any(old_depths != depths)

        depths = jax.lax.while_loop(cond, update, (depths - 1, depths))[1]
    return depths


@Partial(jax.jit, static_argnames=("cuda",))
def order_by_depth(points, indices, neighbors, depths, *, cuda=False):
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        points, indices, neighbors, depths = graphgp_cuda.order_by_depth(points, indices, neighbors, depths)
    else:
        n0 = len(points) - len(neighbors)
        order = jnp.argsort(depths)
        points, indices, depths = points[order], indices[order], depths[order]
        neighbors = neighbors[order[n0:] - n0]  # first n0 should stay in order
        inv_order = jnp.arange(len(points), dtype=int)
        inv_order = inv_order.at[order].set(inv_order)
        neighbors = inv_order[neighbors]
    return points, indices, neighbors, depths
