from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import Array
from jax import lax

import numpy as np

from .covariance import compute_cov_matrix
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
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer.

    Returns:
        The generated values of shape ``(N,).``
    """
    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        xi = xi[graph.indices]
    initial_values = generate_dense(graph.points[:n0], covariance, xi[:n0])
    values = refine(
        graph.points, graph.neighbors, graph.offsets, covariance, initial_values, xi[n0:], cuda=cuda, fast_jit=fast_jit
    )
    if graph.indices is not None:
        values = jnp.empty_like(values).at[graph.indices].set(values, unique_indices=True)
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
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        values = graphgp_cuda.refine(points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance, initial_values, xi)
    else:
        if fast_jit:
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
            noise = L[:, k, k] * xi

            # For each batch defined by offsets, dot neighbor values with mean_vec and add noise
            def step(values, start):
                neighbor_values = values[lax.dynamic_slice(neighbors, (start - n0, 0), (max_batch, k))]
                mean_slice = jnp.sum(
                    lax.dynamic_slice(mean_vec, (start - n0, 0), (max_batch, k)) * neighbor_values, axis=1
                )
                noise_slice = lax.dynamic_slice(noise, (start - n0,), (max_batch,))
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
                mean, std = jax.vmap(Partial(_conditional_mean_std, covariance))(
                    coarse_points, coarse_values, fine_point
                )
                values = jnp.concatenate([values, mean + std * fine_xi], axis=0)

    return values


def generate_inv(
    graph: Graph,
    covariance: Tuple[Array, Array],
    values: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
) -> Array:
    """
    Inverse of ``generate``. Ensure that the choice for ``reorder`` is the same. Recommended to JIT compile.
    """
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
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        initial_values, xi = graphgp_cuda.refine_inv(points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance, values)
    else:
        n0 = len(points) - len(neighbors)
        initial_values = values[:n0]
        xi = jnp.array([], dtype=values.dtype)
        for i in range(len(offsets) - 1, 0, -1):
            start = offsets[i - 1]
            end = offsets[i]
            coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
            coarse_values = jnp.take(values, neighbors[start - n0 : end - n0], axis=0)
            fine_point = points[start:end]
            fine_value = values[start:end]
            mean, std = jax.vmap(Partial(_conditional_mean_std, covariance))(coarse_points, coarse_values, fine_point)
            xi = jnp.concatenate([(fine_value - mean) / std, xi], axis=0)
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
        logdet = jnp.array(0.0)
        n0 = len(points) - len(neighbors)
        for i in range(1, len(offsets)):
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


# def _cuda_process_covariance(covariance):
#     if isinstance(covariance, Callable):
#         raise ValueError("covariance must be (cov_bins, cov_vals) or (cov_bins, cov_func), not cov_func, if cuda=True.")
#     elif isinstance(covariance, Tuple) and isinstance(covariance[0], Array) and isinstance(covariance[1], Callable):
#         cov_bins, cov_func = covariance
#         cov_vals = cov_func(cov_bins)
#     elif isinstance(covariance, Tuple) and isinstance(covariance[0], Array) and isinstance(covariance[1], Array):
#         cov_bins, cov_vals = covariance
#     else:
#         raise ValueError("Invalid covariance specification.")
#     return cov_bins, cov_vals


generate_jit = jax.jit(generate, static_argnames=("cuda"))
generate_inv_jit = jax.jit(generate_inv)
generate_logdet_jit = jax.jit(generate_logdet)

refine_jit = jax.jit(refine)
refine_inv_jit = jax.jit(refine_inv)
refine_logdet_jit = jax.jit(refine_logdet)
