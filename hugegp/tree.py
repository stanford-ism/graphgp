from __future__ import annotations
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial
from jax import Array

@jax.jit
def build_tree(points: Array) -> Tuple[Array, Array, Array]:
    """
    Build k-d tree in special order.

    Args:
        points: Input points of shape ``(N, d)``.

    Returns:
        tuple:
            - points: Reordered points of shape ``(N, d)``.
            - split_dims: Split dimension for each point of shape ``(N,)``.
            - indices: Original indices of the points of shape ``(N,)``.
    """
    if points.ndim != 2:
        raise ValueError(f"Points must have shape (N, d). Got shape {points.shape}.")
    return _build_tree(points)

@Partial(jax.jit, static_argnames=("n0", "k"))
def query_preceding_neighbors(points: Array, split_dims: Array, *, n0: int, k: int) -> tuple[Array, Array]:
    """
    Query the k-nearest neighbors of each point among the preceding points, starting from point n0.

    Args:
        points: Input points in tree order of shape ```(N, d)```.
        split_dims: Split dimension for each point of shape ```(N,)```.
        k: Number of neighbors to query.
        n0: Starting point for the query.

    Returns:
        tuple:
            - neighbors: Indices of the neighbors of shape ``(N - n0, k)``.
            - distances: Distances to the neighbors of shape ``(N - n0, k)``.
    """
    if n0 < k:
        raise ValueError(f"n0 must be at least k. Got n0={n0}, k={k}.")
    query_indices = jnp.arange(n0, len(points))
    query_func = Partial(_single_query_neighbors, points, split_dims, k=k)
    neighbors, distances = jax.vmap(query_func)(query_indices, query_indices)
    return neighbors, distances


@Partial(jax.jit, static_argnames=("offsets", "k"))
def query_offset_neighbors(
    points: Array, split_dims: Array, *, offsets: Tuple[int, ...], k: int
) -> tuple[Array, Array]:
    """
    Query the k-nearest neighbors of each point among points in preceding batches.

    Args:
        points: Input points in tree order of shape ``(N, d)``.
        split_dims: Split dimension for each point of shape ``(N,)``.
        offsets: Tuple of length ``B`` representing the end index of each batch.
        k: Number of neighbors to query.

    Returns:
        tuple:
            - neighbors: Indices of the neighbors of shape ``(N - offsets[0], k)``.
            - distances: Distances to the neighbors of shape ``(N - offsets[0], k)``.
    """
    query_indices = jnp.arange(offsets[0], len(points))
    offsets = jnp.asarray(offsets)
    max_indices = offsets[jnp.searchsorted(offsets, query_indices, side="right") - 1]
    query_func = Partial(_single_query_neighbors, points, split_dims, k=k)
    neighbors, distances = jax.vmap(query_func)(query_indices, max_indices)
    return neighbors, distances


def _single_query_neighbors(points, split_dims, query_index, max_index, *, k):
    # rule for updating neighbors
    def update_func(node, state, _):
        neighbors, square_distances = state
        square_distance = jnp.sum(jnp.square(points[query_index] - points[node]), axis=-1)
        max_neighbor = jnp.argmax(square_distances)
        neighbors, square_distances = lax.cond(
            # if the node is closer than the farthest neighbor, replace
            square_distance < square_distances[max_neighbor],
            lambda _: (
                neighbors.at[max_neighbor].set(node),
                square_distances.at[max_neighbor].set(square_distance),
            ),
            lambda _: (neighbors, square_distances),
            None,
        )
        return (neighbors, square_distances), jnp.max(square_distances)

    neighbors = -1 * jnp.ones(k, dtype=int)
    square_distances = jnp.inf * jnp.ones(k, dtype=points.dtype)
    neighbors, _ = _traverse_tree(
        points, split_dims, query_index, max_index, update_func, (neighbors, square_distances), jnp.asarray(jnp.inf)
    )

    distances = jnp.linalg.norm(points[neighbors] - points[query_index], axis=-1)  # recompute distances to enable VJP
    distances, neighbors = lax.sort((distances, neighbors), dimension=0, num_keys=2)
    return neighbors, distances


def _traverse_tree(
    points,
    split_dims,
    query_index,
    max_index,
    update_func,  # takes (current node, state, square radius) -> (new state, new square radius)
    initial_state,
    initial_square_radius,
):
    def step(carry):
        # Update neighbors with the current node if necessary
        current, previous, state, square_radius = carry
        parent = _compute_parent(current)
        state, square_radius = lax.cond(
            previous == parent, update_func, lambda _, s, r: (s, r), current, state, square_radius
        )

        # Locate children and determine if far child is in range
        split_distance = points[query_index, split_dims[current]] - points[current, split_dims[current]]
        near_child = jnp.where(split_distance < 0, _compute_left(current), _compute_right(current))
        far_child = jnp.where(split_distance < 0, _compute_right(current), _compute_left(current))
        far_in_range = jnp.square(split_distance) <= square_radius

        # Determine next node to traverse
        next = lax.select(
            # go to the far child if we came from near child or near child doesn't exist
            (previous == near_child) | ((previous == parent) & (near_child >= max_index)),
            # only go to the far child if it exists and is in range
            lax.select((far_child < max_index) & far_in_range, far_child, parent),
            # go to the near child if it exists and we came from the parent
            lax.select(previous == parent, near_child, parent),
        )
        return next, current, state, square_radius

    # Loop until we return to root
    current = jnp.array(0, dtype=int)
    previous = jnp.array(-1, dtype=int)
    _, _, state, _ = lax.while_loop(
        lambda carry: carry[0] != jnp.array(-1, dtype=int),
        step,
        (current, previous, initial_state, initial_square_radius),
    )
    return state


def _compute_left(current):
    level = jnp.frexp(current + 1)[1] - 1
    # level = 32 - lax.clz(current + 1) - 1
    n_level = 1 << level
    return current + n_level


def _compute_right(current):
    level = jnp.frexp(current + 1)[1] - 1
    n_level = 1 << level
    return current + 2 * n_level


def _compute_parent(current):
    level = jnp.frexp(current + 1)[1] - 1
    n_above = (1 << level) - 1
    n_parent_level = 1 << (level - 1)
    parent = jnp.where(current < n_above + n_parent_level, current - n_parent_level, current - 2 * n_parent_level)
    parent = jnp.where(current == 0, -1, parent)  # root has no parent
    return parent


def _build_tree(points):
    n_points = len(points)
    n_levels = n_points.bit_length()
    array_index = jnp.arange(n_points, dtype=int)  # needed at various points

    def step(carry, level):
        nodes, points, indices, split_dims = carry

        # Compute split dimension and extract values along that dimension
        dim_max = jax.ops.segment_max(points, nodes, num_segments=n_points)
        dim_min = jax.ops.segment_min(points, nodes, num_segments=n_points)
        new_split_dims = jnp.argmax(dim_max - dim_min, axis=-1).astype(jnp.int8)
        split_dims = jnp.where(array_index < (1 << level) - 1, split_dims, new_split_dims)
        points_along_dim = jnp.squeeze(
            jnp.take_along_axis(points, split_dims[nodes][:, jnp.newaxis], axis=-1),
            axis=-1,
        )

        # Sort the points in each node segment along the splitting dimension
        nodes, _, indices, perm = lax.sort((nodes, points_along_dim, indices, array_index), dimension=0, num_keys=2)
        points = points[perm]

        # Update nodes
        nodes = _update_nodes(nodes, array_index, level)
        return (nodes, points, indices, split_dims), None

    # Start all points at root and sort into tree at each level
    nodes = jnp.zeros(n_points, dtype=int)
    indices = jnp.arange(n_points, dtype=int)
    split_dims = -1 * jnp.ones(n_points, dtype=jnp.int8)
    (nodes, points, indices, split_dims), _ = lax.scan(step, (nodes, points, indices, split_dims), jnp.arange(n_levels))
    return points, split_dims, indices


def _update_nodes(nodes, index, level):
    # Calculate numbers for the level
    n_above = (1 << level) - 1
    n_level = 1 << level
    n_remaining = len(nodes) - n_above
    q = n_remaining // n_level
    r = n_remaining % n_level

    # Compute the global index of the midpoint for each node
    i = nodes - n_above
    midpoint = jnp.where(i < r, i * (q + 1) + (q + 1) // 2, r * (q + 1) + (i - r) * q + q // 2) + n_above

    # Assign left or right child
    nodes = jnp.where(
        (index < n_above) | (index == midpoint),
        nodes,
        jnp.where(
            index < midpoint,
            nodes + n_level,  # left
            nodes + 2 * n_level,  # right
        ),
    )

    return nodes
