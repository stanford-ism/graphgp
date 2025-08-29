from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import Array

from .covariance import compute_cov_matrix
from .graph import Graph

# try:
#     import hugegp_cuda

#     has_cuda = True
# except ImportError:
#     has_cuda = False


def generate(graph: Graph, covariance: Callable, xi: Array, *, reorder: bool = True, cuda: bool = False) -> Array:
    """
    Generate a GP with dense Cholesky for the first layer followed by conditional refinement.
    It is recommended to JIT compile before use.

    Args:
        graph: An instance of ``Graph``, can be checked for validity with ``check_graph``.
        covariance: A callable which takes a distance and returns a covariance. Distance zero should be inflated by a small amount to ensure positive definiteness.
        xi: Unit normal distributed parameters of shape ``(N,).``
        reorder: Whether to reorder parameters and values according to the original order of the points. Default is ``True``.
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.

    Returns:
        The generated values of shape ``(N,).``
    """
    n0 = len(graph.points) - len(graph.neighbors)
    if reorder:
        if graph.indices is None:
            raise ValueError("Graph must have indices to reorder.")
        xi = xi[graph.indices]
    initial_values = generate_dense(graph.points[:n0], covariance, xi[:n0])
    values = refine(graph.points, graph.neighbors, graph.offsets, covariance, initial_values, xi[n0:], cuda=cuda)
    if reorder:
        if graph.reverse_indices is None:
            raise ValueError("Graph must have reverse_indices to reorder.")
        values = values[graph.reverse_indices]
    return values


def generate_inv(graph: Graph, covariance: Callable, values: Array, *, reorder: bool = True, cuda: bool = False) -> Array:
    """
    Inverse of ``generate``. Ensure that the choice for ``reorder`` is the same. Recommended to JIT compile.
    """
    n0 = len(graph.points) - len(graph.neighbors)
    if reorder:
        if graph.indices is None:
            raise ValueError("Graph must have indices to reorder.")
        values = values[graph.indices]
    initial_values, xi = refine_inv(graph.points, graph.neighbors, graph.offsets, covariance, values, cuda=cuda)
    initial_xi = generate_dense_inv(graph.points[:n0], covariance, initial_values)
    xi = jnp.concatenate([initial_xi, xi], axis=0)
    if reorder:
        if graph.reverse_indices is None:
            raise ValueError("Graph must have reverse_indices to reorder.")
        xi = xi[graph.reverse_indices]
    return xi


def generate_logdet(graph: Graph, covariance: Callable, *, cuda: bool = False) -> Array:
    """
    Log determinant of ``generate``. Recommended to JIT compile.
    """
    n0 = len(graph.points) - len(graph.neighbors)
    dense_logdet = generate_dense_logdet(graph.points[:n0], covariance)
    return dense_logdet + refine_logdet(graph.points, graph.neighbors, graph.offsets, covariance, cuda=cuda)


def generate_dense(points: Array, covariance: Callable, xi: Array) -> Array:
    """
    Generate a GP with a dense Cholesky decomposition. Note that to compare with the GraphGP values,
    the points must be provided in tree order.

    Args:
        points: Locations of points to model of shape ``(N, d)``
        covariance: A callable which takes a distance and returns a covariance. Distance zero should be inflated by a small amount to ensure positive definiteness.
        xi: Unit normal distributed parameters of shape ``(N,).``
    Returns:
        The generated values of shape ``(N,).``
    """
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    values = L @ xi
    return values


def generate_dense_inv(points: Array, covariance: Callable, values: Array) -> Array:
    """
    Inverse of ``generate_dense``.
    """
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    xi = jnp.linalg.solve(L, values)
    return xi


def generate_dense_logdet(points: Array, covariance: Callable) -> Array:
    """
    Log determinant of ``generate_dense``.
    """
    K = compute_cov_matrix(covariance, points, points)
    return jnp.linalg.slogdet(K)[1] / 2


def refine(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Callable,
    initial_values: Array,
    xi: Array,
    *,
    cuda: bool = False,
) -> Array:
    """
    Conditionally generate using initial values according to GraphGP algorithm. Most users can use ``generate``, which
    automatically generates the initial values and accepts a ``Graph`` object as input. This function is provided
    if more flexibility is needed, for example to conditionally upsample already generated values.

    It is recommended to JIT compile before use due to internal for loops.

    Args:
        points: Modeled points in tree order of shape ``(N, d)``.
        neighbors: Indices of the neighbors of shape ``(N - offsets[0], k)``.
        offsets: Tuple of length ``B`` representing the end index of each batch.
        covariance: A callable which takes a distance and returns a covariance. Distance zero should be inflated by a small amount to ensure positive definiteness.
        initial_values: Initial values of shape ``(offsets[0],).``
        xi: Unit normal distributed parameters of shape ``(N - offsets[0],).``
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.

    Returns:
        The refined values of shape ``(N,).``

    """
    n0 = len(points) - len(neighbors)  # should equal offsets[0]
    values = initial_values
    offsets = (0,) + offsets
    for i in range(2, len(offsets)):
        start = offsets[i - 1]
        end = offsets[i]
        coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
        coarse_values = jnp.take(values, neighbors[start - n0 : end - n0], axis=0)
        fine_point = points[start:end]
        fine_xi = xi[start - n0 : end - n0]
        mean, std = jax.vmap(Partial(_conditional_mean_std, covariance))(coarse_points, coarse_values, fine_point)
        values = jnp.concatenate([values, mean + std * fine_xi], axis=0)
    return values


def refine_inv(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Callable,
    values: Array,
    *,
    cuda: bool = False,
) -> Tuple[Array, Array]:
    """
    Inverse of ``refine``.
    """
    n0 = offsets[0]
    initial_values = values[:n0]
    xi = jnp.array([], dtype=values.dtype)
    offsets = (0,) + offsets
    for i in range(len(offsets) - 1, 1, -1):
        start = offsets[i - 1]
        end = offsets[i]
        coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
        coarse_values = jnp.take(values, neighbors[start - n0 : end - n0], axis=0)
        fine_point = points[start:end]
        fine_value = values[start:end]
        mean, std = jax.vmap(Partial(_conditional_mean_std, covariance))(coarse_points, coarse_values, fine_point)
        xi = jnp.concatenate([(fine_value - mean) / std, xi], axis=0)
    return initial_values, xi


def refine_logdet(
    points: Array, neighbors: Array, offsets: Tuple[int, ...], covariance: Callable, *, cuda: bool = False
) -> Array:
    """
    Log determinant of ``refine``.
    """
    logdet = 0.0
    n0 = offsets[0]
    offsets = (0,) + offsets
    for i in range(2, len(offsets)):
        start = offsets[i - 1]
        end = offsets[i]
        coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
        fine_point = points[start:end]
        std = jax.vmap(Partial(_conditional_std, covariance))(coarse_points, fine_point)
        logdet += jnp.sum(jnp.log(std))
    return logdet


def _conditional_mean_std(covariance, coarse_points, coarse_values, fine_point):
    k = len(coarse_points)
    joint_points = jnp.concatenate([coarse_points, fine_point[jnp.newaxis]], axis=0)
    K = compute_cov_matrix(covariance, joint_points, joint_points)
    L = jnp.linalg.cholesky(K)
    mean = L[k, :k] @ jnp.linalg.solve(L[:k, :k], coarse_values)
    std = L[k, k]
    return mean, std


def _conditional_std(covariance, coarse_points, fine_point):
    k = len(coarse_points)
    joint_points = jnp.concatenate([coarse_points, fine_point[jnp.newaxis]], axis=0)
    K = compute_cov_matrix(covariance, joint_points, joint_points)
    L = jnp.linalg.cholesky(K)
    return L[k, k]
