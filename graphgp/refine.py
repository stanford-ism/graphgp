from typing import Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import Array
from jax import lax

import numpy as np

from .graph import Graph

try:
    import graphgp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False


def generate(
    graph: Graph,
    covariance: Tuple[Array, Array],
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
) -> Array:
    """
    Generate a GP with dense Cholesky for the first layer followed by conditional refinement.
    It is recommended to JIT compile before use.

    Args:
        graph: An instance of ``Graph``, can be checked for validity with ``check_graph``.
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        xi: Unit normal distributed parameters of shape ``(N,).``
        reorder: Whether to reorder parameters and values according to the original order of the points. Default is ``True``.
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer slightly.

    Returns:
        The generated values of shape ``(N,).``
    """
    if len(xi) != len(graph.points):
        raise ValueError("Length of xi must match number of points in graph.")
    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        xi = xi[graph.indices]
    initial_values = generate_dense(graph.points[:n0], covariance, xi[:n0])
    values = refine(
        graph.points, graph.neighbors, graph.offsets, covariance, initial_values, xi[n0:], cuda=cuda, fast_jit=fast_jit
    )
    if graph.indices is not None:
        values = jnp.empty_like(values).at[graph.indices].set(values, unique_indices=True)
    values = jnp.where(jnp.any(jnp.isnan(values)), jnp.nan * values, values)
    return values


def generate_dense(points: Array, covariance: Tuple[Array, Array], xi: Array) -> Array:
    """
    Generate a GP with a dense Cholesky decomposition. Note that to compare with the GraphGP values,
    the points must be provided in tree order.

    Args:
        points: Locations of points to model of shape ``(N, d)``
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        xi: Unit normal distributed parameters of shape ``(N,).``
    Returns:
        The generated values of shape ``(N,).``
    """
    if len(xi) != len(points):
        raise ValueError("Length of xi must match number of points.")
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    values = L @ xi
    return values


def refine(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Tuple[Array, Array],
    initial_values: Array,
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
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
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        initial_values: Initial values of shape ``(offsets[0],).``
        xi: Unit normal distributed parameters of shape ``(N - offsets[0],).``
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer.

    Returns:
        The refined values of shape ``(N,).``

    """
    n0 = len(points) - len(neighbors)  # should equal offsets[0]
    if len(initial_values) != n0:
        raise ValueError("Length of initial_values must match number of initial points.")
    if len(xi) != len(points) - n0:
        raise ValueError("Length of xi must match number of refined points.")
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        values = graphgp_cuda.refine(
            points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance, initial_values, xi
        )

    elif fast_jit:
        k = neighbors.shape[1]
        max_batch = np.max(np.diff(np.array(offsets)))
        values = jnp.zeros(len(points))
        values = values.at[:n0].set(initial_values)

        # Precompute matrix factorizations for all points
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        L = jnp.linalg.cholesky(K)
        mean_vec = jnp.linalg.solve(L[:, :k, :k].transpose(0, 2, 1), L[:, k, :k][..., None]).squeeze(-1)
        std = L[:, k, k]

        # For each batch defined by offsets, dot neighbor values with mean_vec and add noise
        def step(values, start):
            neighbor_values = values[lax.dynamic_slice(neighbors, (start - n0, 0), (max_batch, k))]
            mean_slice = jnp.sum(lax.dynamic_slice(mean_vec, (start - n0, 0), (max_batch, k)) * neighbor_values, axis=1)
            noise_slice = lax.dynamic_slice(std * xi, (start - n0,), (max_batch,))
            values = lax.dynamic_update_slice(values, mean_slice + noise_slice, (start,))
            return values, None

        values, _ = lax.scan(step, values, jnp.array(offsets[:-1]))

    else:
        values = initial_values
        for i in range(1, len(offsets)):
            start = offsets[i - 1]
            end = offsets[i]
            coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
            coarse_values = jnp.take(values, neighbors[start - n0 : end - n0], axis=0)
            fine_point = points[start:end]
            fine_xi = xi[start - n0 : end - n0]
            mean, std = jax.vmap(Partial(_conditional_mean_std, covariance))(coarse_points, coarse_values, fine_point)
            values = jnp.concatenate([values, mean + std * fine_xi], axis=0)

    return values


def generate_inv(
    graph: Graph,
    covariance: Tuple[Array, Array],
    values: Array,
    *,
    cuda: bool = False,
) -> Array:
    """
    Inverse of ``generate``. Ensure that the choice for ``reorder`` is the same. Recommended to JIT compile.
    """
    if len(values) != len(graph.points):
        raise ValueError("Length of values must match number of points in graph.")
    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        values = values[graph.indices]
    initial_values, xi = refine_inv(graph.points, graph.neighbors, graph.offsets, covariance, values, cuda=cuda)
    initial_xi = generate_dense_inv(graph.points[:n0], covariance, initial_values)
    xi = jnp.concatenate([initial_xi, xi], axis=0)
    if graph.indices is not None:
        xi = jnp.empty_like(xi).at[graph.indices].set(xi, unique_indices=True)
    return xi


def generate_dense_inv(points: Array, covariance: Tuple[Array, Array], values: Array) -> Array:
    """
    Inverse of ``generate_dense``.
    """
    if len(values) != len(points):
        raise ValueError("Length of values must match number of points.")
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    xi = jnp.linalg.solve(L, values)
    return xi


def refine_inv(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Tuple[Array, Array],
    values: Array,
    *,
    cuda: bool = False,
) -> Tuple[Array, Array]:
    """
    Inverse of ``refine``.
    """
    n0 = len(points) - len(neighbors)  # should equal offsets[0]
    if len(values) != len(points):
        raise ValueError("Length of values must match number of points.")
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        initial_values, xi = graphgp_cuda.refine_inv(
            points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance, values
        )
    else:
        k = neighbors.shape[1]
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        L = jnp.linalg.cholesky(K)
        mean_vec = jnp.linalg.solve(L[:, :k, :k].transpose(0, 2, 1), L[:, k, :k][..., None]).squeeze(-1)
        mean = jnp.sum(mean_vec * values[neighbors], axis=1)
        std = L[:, k, k]

        xi = (values[n0:] - mean) / std
        initial_values = values[:n0]
    return initial_values, xi


def generate_logdet(graph: Graph, covariance: Tuple[Array, Array], *, cuda: bool = False) -> Array:
    """
    Log determinant of ``generate``.
    """
    n0 = len(graph.points) - len(graph.neighbors)
    dense_logdet = generate_dense_logdet(graph.points[:n0], covariance)
    return dense_logdet + refine_logdet(graph.points, graph.neighbors, graph.offsets, covariance, cuda=cuda)


def generate_dense_logdet(points: Array, covariance: Tuple[Array, Array]) -> Array:
    """
    Log determinant of ``generate_dense``.
    """
    K = compute_cov_matrix(covariance, points, points)
    return jnp.linalg.slogdet(K)[1] / 2


def refine_logdet(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Tuple[Array, Array],
    *,
    cuda: bool = False,
) -> Array:
    """
    Log determinant of ``refine``.
    """
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        logdet = graphgp_cuda.refine_logdet(points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance)
    else:
        n0 = len(points) - len(neighbors)
        k = neighbors.shape[1]
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        L = jnp.linalg.cholesky(K)
        std = L[:, k, k]
        logdet = jnp.sum(jnp.log(std))
    return logdet


def _conditional_mean_std(covariance, coarse_points, coarse_values, fine_point):
    k = len(coarse_points)
    joint_points = jnp.concatenate([coarse_points, fine_point[jnp.newaxis]], axis=0)
    K = compute_cov_matrix(covariance, joint_points, joint_points)
    L = jnp.linalg.cholesky(K)
    mean = L[k, :k] @ jnp.linalg.solve(L[:k, :k], coarse_values)
    std = L[k, k]
    return mean, std


def compute_cov_matrix(covariance: Tuple[Array, Array], points_a: Array, points_b: Array) -> Array:
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    if isinstance(covariance, Tuple) and isinstance(covariance[0], Array) and isinstance(covariance[1], Array):
        cov_bins, cov_vals = covariance
        return cov_lookup(distances, cov_bins, cov_vals)
    else:
        raise ValueError("Invalid covariance specification.")


def cov_lookup(r, cov_bins, cov_vals):
    """
    Look up covariance in array of sampled `cov_vals` at radii `cov_bins` (equal-sized arrays).
    If `r` is inside of bounds, a linearly interpolated value is returned.
    If `r` is below the first bin, the first value is returned. But really the first bin should always be 0.0.
    If `r` is above the last bin, the last value is returned. Maybe the last value should be zero.
    """
    return jnp.interp(r, cov_bins, cov_vals)
