from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import Partial, register_dataclass
from jax import Array


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


def build_graph(
    points: Array,
    *,
    n0: int,
    k: int,
    cuda: bool = False,
    periodic: bool = False,
    boxsize: Array | float | Tuple[float, ...] | None = None,
) -> Graph:
    """
    Build a graph where each point depends on its ``k`` nearest neighbors which precede it in a k-d tree ordering (the original point order does not matter).

    Args:
        points: The input points of shape ``(N, d)``.
        n0: The number of initial points.
        k: The number of neighbors to include.
        cuda: Whether to use optional CUDA extension, if installed.
        periodic: If ``True``, choose nearest predecessors with the periodic minimum-image metric.
        boxsize: Side lengths of the periodic box. Required when ``periodic=True``.

    Returns:
        A ``Graph`` dataclass containing ``points``, ``neighbors``, ``offsets``, and ``indices``.
    """
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        if periodic:
            raise NotImplementedError("Periodic nearest-neighbor queries are currently only implemented on the CPU path.")
        points, indices, neighbors, depths = graphgp_cuda.build_graph(points, n0=n0, k=k)
    else:
        points, split_dims, indices = build_tree(points)
        neighbors = query_preceding_neighbors(
            points,
            split_dims,
            n0=n0,
            k=k,
            periodic=periodic,
            boxsize=boxsize,
        )
        depths = compute_depths(neighbors, n0=n0)
        points, indices, neighbors, depths = order_by_depth(points, indices, neighbors, depths)
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
    return Graph(points, neighbors, offsets, indices)


def build_interpolation_graph(
    source_points: Array,
    target_points: Array,
    *,
    k: int,
    periodic: bool = False,
    boxsize: Array | float | Tuple[float, ...] | None = None,
) -> Graph:
    """Build a dependency graph for conditional refinement from known source points to new target points."""
    source_points = jnp.asarray(source_points)
    target_points = jnp.asarray(target_points)
    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("source_points and target_points must have shape (N, d).")
    if source_points.shape[1] != target_points.shape[1]:
        raise ValueError("source_points and target_points must have the same dimensionality.")
    if k <= 0 or len(source_points) < k:
        raise ValueError("Need at least k source points and k>0.")

    source_np = np.asarray(source_points)
    target_tree, _, target_indices = build_tree(target_points)
    target_np = np.asarray(target_tree)
    n0 = len(source_np)
    nt = len(target_np)
    points_np = np.concatenate([source_np, target_np], axis=0)
    indices_np = np.concatenate([np.arange(n0, dtype=np.int32), n0 + np.asarray(target_indices, dtype=np.int32)], axis=0)
    neighbors_np = np.empty((nt, k), dtype=np.int32)

    if periodic:
        if boxsize is None:
            raise ValueError("boxsize must be provided when periodic=True.")
        box = np.asarray(boxsize, dtype=np.float64)
        if box.ndim == 0:
            box = np.full(points_np.shape[1], float(box), dtype=np.float64)
    else:
        box = None

    for i in range(nt):
        idx = n0 + i
        deltas = points_np[:idx] - points_np[idx]
        if periodic:
            deltas = deltas - box[None, :] * np.round(deltas / box[None, :])
        dist2 = np.einsum('ij,ij->i', deltas, deltas)
        nbr = np.argpartition(dist2, kth=k-1)[:k]
        nbr = nbr[np.argsort(dist2[nbr])]
        neighbors_np[i] = nbr

    points = jnp.asarray(points_np)
    indices = jnp.asarray(indices_np)
    neighbors = jnp.asarray(neighbors_np)
    depths = compute_depths(neighbors, n0=n0)
    points, indices, neighbors, depths = order_by_depth(points, indices, neighbors, depths)
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
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
