from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.tree_util import Partial

from .custom_cholesky import cholesky, permute, pivot_order, solve_lower, solve_lower_transpose
from .graph import Graph

try:
    import graphgp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False

Covariance = tuple[Array, Array] | Callable[[Array, Array], Array]


def generate(
    graph: Graph,
    covariance: Covariance,
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
    clamp: bool = True,
) -> Array:
    """
    Generate a GP with dense Cholesky for the first layer followed by conditional refinement.
    It is recommended to JIT compile before use.

    Args:
        graph: An instance of ``Graph``, can be checked for validity with ``check_graph``.
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance, or a callable ``cov(x1, x2)`` taking two points of shape ``(d,)`` and returning a scalar. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        xi: Unit normal distributed parameters of shape ``(N,).``
        reorder: Whether to reorder parameters and values according to the original order of the points. Default is ``True``.
        cuda: Whether to use optional CUDA extension, if installed. Requires a discretized covariance. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer slightly.
        clamp: Handle near-duplicate points by dropping Cholesky pivots below floating point resolution, avoiding NaN. The xi for dropped points have no effect and the values are set to their conditional mean. The set of dropped points may change as the covariance changes. Derivatives hold the set of points fixed. Default is ``True`` which incurs a performance penalty in the pure JAX version. If ``cuda=True``, clamping cannot be disabled. With clamping, each point's neighbors are also ordered by pivoting for numerical stability.

    Returns:
        The generated values of shape ``(N,).``
    """
    if len(xi) != len(graph.points):
        raise ValueError("Length of xi must match number of points in graph.")
    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        xi = xi[graph.indices]
    initial_values = generate_dense(graph.points[:n0], covariance, xi[:n0], clamp=clamp)
    values = refine(
        graph.points,
        graph.neighbors,
        graph.offsets,
        covariance,
        initial_values,
        xi[n0:],
        cuda=cuda,
        fast_jit=fast_jit,
        clamp=clamp,
    )
    if graph.indices is not None:
        values = jnp.empty_like(values).at[graph.indices].set(values, unique_indices=True)
    values = jnp.where(jnp.any(jnp.isnan(values)), jnp.full_like(values, jnp.nan), values)
    return values


def generate_dense(points: Array, covariance: Covariance, xi: Array, *, clamp: bool = True) -> Array:
    """
    Generate a GP with a dense Cholesky decomposition. Note that to compare with the GraphGP values,
    the points must be provided in tree order.

    Args:
        points: Locations of points to model of shape ``(N, d)``
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance, or a callable ``cov(x1, x2)`` taking two points of shape ``(d,)`` and returning a scalar. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        xi: Unit normal distributed parameters of shape ``(N,).``
        clamp: Handle near-duplicate points by dropping Cholesky pivots below floating point resolution, avoiding NaN. The xi for dropped points have no effect and the values are set to their conditional mean. The set of dropped points may change as the covariance changes. Derivatives hold the set of points fixed. Default is ``True`` which incurs a performance penalty in the pure JAX version. If ``cuda=True``, clamping cannot be disabled. With clamping, each point's neighbors are also ordered by pivoting for numerical stability.
    Returns:
        The generated values of shape ``(N,).``
    """
    if len(xi) != len(points):
        raise ValueError("Length of xi must match number of points.")
    K = compute_cov_matrix(covariance, points, points)
    L = _cholesky(K, clamp)
    values = L @ xi
    return values


def refine(
    points: Array,
    neighbors: Array,
    offsets: tuple[int, ...],
    covariance: Covariance,
    initial_values: Array,
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
    clamp: bool = True,
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
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance, or a callable ``cov(x1, x2)`` taking two points of shape ``(d,)`` and returning a scalar. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        initial_values: Initial values of shape ``(offsets[0],).``
        xi: Unit normal distributed parameters of shape ``(N - offsets[0],).``
        cuda: Whether to use optional CUDA extension, if installed. Requires a discretized covariance. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer.
        clamp: Handle near-duplicate points by dropping Cholesky pivots below floating point resolution, avoiding NaN. The xi for dropped points have no effect and the values are set to their conditional mean. The set of dropped points may change as the covariance changes. Derivatives hold the set of points fixed. Default is ``True`` which incurs a performance penalty in the pure JAX version. If ``cuda=True``, clamping cannot be disabled. With clamping, each point's neighbors are also ordered by pivoting for numerical stability.

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
        if not clamp:
            raise ValueError("clamp=False is not supported with cuda=True, the CUDA extension always clamps.")
        _check_discretized(covariance)
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
        K, neighbors = _pivot_neighbors(K, neighbors, clamp)
        L = _cholesky(K, clamp)
        mean_vec = _kriging_weights(L, k, clamp)
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
            mean, std = jax.vmap(Partial(_conditional_mean_std, covariance, clamp=clamp))(
                coarse_points, coarse_values, fine_point
            )
            values = jnp.concatenate([values, mean + std * fine_xi], axis=0)

    return values


def generate_inv(
    graph: Graph,
    covariance: Covariance,
    values: Array,
    *,
    cuda: bool = False,
    clamp: bool = True,
) -> Array:
    """
    Inverse of ``generate``. Ensure that the choice for ``reorder`` is the same. Recommended to JIT compile.
    With ``clamp=True``, dropped points receive ``xi = 0``.
    """
    if len(values) != len(graph.points):
        raise ValueError("Length of values must match number of points in graph.")
    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        values = values[graph.indices]
    initial_values, xi = refine_inv(
        graph.points, graph.neighbors, graph.offsets, covariance, values, cuda=cuda, clamp=clamp
    )
    initial_xi = generate_dense_inv(graph.points[:n0], covariance, initial_values, clamp=clamp)
    xi = jnp.concatenate([initial_xi, xi], axis=0)
    if graph.indices is not None:
        xi = jnp.empty_like(xi).at[graph.indices].set(xi, unique_indices=True)
    xi = jnp.where(jnp.any(jnp.isnan(xi)), jnp.full_like(xi, jnp.nan), xi)
    return xi


def generate_dense_inv(points: Array, covariance: Covariance, values: Array, *, clamp: bool = True) -> Array:
    """
    Inverse of ``generate_dense``. With ``clamp=True``, dropped points receive ``xi = 0``.
    """
    if len(values) != len(points):
        raise ValueError("Length of values must match number of points.")
    K = compute_cov_matrix(covariance, points, points)
    L = _cholesky(K, clamp)
    xi = _solve_lower(L, values, clamp)
    return xi


def refine_inv(
    points: Array,
    neighbors: Array,
    offsets: tuple[int, ...],
    covariance: Covariance,
    values: Array,
    *,
    cuda: bool = False,
    clamp: bool = True,
) -> tuple[Array, Array]:
    """
    Inverse of ``refine``. With ``clamp=True``, dropped points receive ``xi = 0``.
    """
    n0 = len(points) - len(neighbors)  # should equal offsets[0]
    if len(values) != len(points):
        raise ValueError("Length of values must match number of points.")
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        if not clamp:
            raise ValueError("clamp=False is not supported with cuda=True, the CUDA extension always clamps.")
        _check_discretized(covariance)
        initial_values, xi = graphgp_cuda.refine_inv(
            points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance, values
        )
    else:
        k = neighbors.shape[1]
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        K, neighbors = _pivot_neighbors(K, neighbors, clamp)
        L = _cholesky(K, clamp)
        mean_vec = _kriging_weights(L, k, clamp)
        mean = jnp.sum(mean_vec * values[neighbors], axis=1)
        std = L[:, k, k]

        xi = _safe_div(values[n0:] - mean, std) if clamp else (values[n0:] - mean) / std
        initial_values = values[:n0]
    return initial_values, xi


def generate_logdet(graph: Graph, covariance: Covariance, *, cuda: bool = False, clamp: bool = True) -> Array:
    """
    Log determinant of ``generate``.
    With ``clamp=True``, the result is ``-inf`` if any point is dropped, since the density is then degenerate.
    """
    n0 = len(graph.points) - len(graph.neighbors)
    dense_logdet = generate_dense_logdet(graph.points[:n0], covariance, clamp=clamp)
    return dense_logdet + refine_logdet(
        graph.points, graph.neighbors, graph.offsets, covariance, cuda=cuda, clamp=clamp
    )


def generate_dense_logdet(points: Array, covariance: Covariance, *, clamp: bool = True) -> Array:
    """
    Log determinant of ``generate_dense``.
    With ``clamp=True``, the result is ``-inf`` if any point is dropped, since the density is then degenerate.
    """
    K = compute_cov_matrix(covariance, points, points)
    if clamp:
        return jnp.sum(jnp.log(jnp.diagonal(cholesky(K), axis1=-2, axis2=-1)))
    return jnp.linalg.slogdet(K)[1] / 2


def refine_logdet(
    points: Array,
    neighbors: Array,
    offsets: tuple[int, ...],
    covariance: Covariance,
    *,
    cuda: bool = False,
    clamp: bool = True,
) -> Array:
    """
    Log determinant of ``refine``.
    With ``clamp=True``, the result is ``-inf`` if any point is dropped, since the density is then degenerate.
    """
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        if not clamp:
            raise ValueError("clamp=False is not supported with cuda=True, the CUDA extension always clamps.")
        _check_discretized(covariance)
        logdet = graphgp_cuda.refine_logdet(points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance)
    else:
        n0 = len(points) - len(neighbors)
        k = neighbors.shape[1]
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        K, neighbors = _pivot_neighbors(K, neighbors, clamp)
        L = _cholesky(K, clamp)
        std = L[:, k, k]
        logdet = jnp.sum(jnp.log(std))
    return logdet


def _conditional_mean_std(covariance, coarse_points, coarse_values, fine_point, *, clamp=True):
    k = len(coarse_points)
    joint_points = jnp.concatenate([coarse_points, fine_point[jnp.newaxis]], axis=0)
    K = compute_cov_matrix(covariance, joint_points, joint_points)
    if clamp:
        perm = pivot_order(K, k)
        K, coarse_values = permute(K, perm), coarse_values[perm[:k]]
    L = _cholesky(K, clamp)
    mean = L[k, :k] @ _solve_lower(L[:k, :k], coarse_values, clamp)
    std = L[k, k]
    return mean, std


def _pivot_neighbors(K, neighbors, clamp):
    """Reorder each point's neighbors by pivoting (target stays last), returning the permuted ``K`` and ``neighbors``."""
    if not clamp:
        return K, neighbors
    k = neighbors.shape[1]
    perm = pivot_order(K, k)
    return permute(K, perm), jnp.take_along_axis(neighbors, perm[:, :k], axis=1)


def _cholesky(K, clamp):
    return cholesky(K) if clamp else jnp.linalg.cholesky(K)


def _solve_lower(L, b, clamp):
    return solve_lower(L, b) if clamp else jnp.linalg.solve(L, b)


def _kriging_weights(L, k, clamp):
    """Weights of the neighbor values in the conditional mean, ``L_cc^-T L_fc``, for a batch of factors ``L``."""
    if clamp:
        return solve_lower_transpose(L[:, :k, :k], L[:, k, :k])
    return jnp.linalg.solve(L[:, :k, :k].transpose(0, 2, 1), L[:, k, :k][..., None]).squeeze(-1)


def _safe_div(a, b):
    """``a / b``, or zero where ``b == 0`` (dropped pivot), without NaN gradients."""
    return jnp.where(b == 0, 0, a / jnp.where(b == 0, 1, b))


def compute_cov_matrix(covariance: Covariance, points_a: Array, points_b: Array) -> Array:
    """
    Compute the covariance matrix between ``points_a`` of shape ``(..., N, d)`` and ``points_b`` of shape ``(..., M, d)``.
    ``covariance`` is either a tuple ``(cov_bins, cov_vals)`` storing a discretized stationary covariance, or a callable
    ``cov(x1, x2)`` taking two points of shape ``(d,)`` and returning a scalar.
    """
    if callable(covariance):
        cov = jnp.vectorize(covariance, signature="(d),(d)->()")
        return cov(jnp.expand_dims(points_a, -2), jnp.expand_dims(points_b, -3))
    elif isinstance(covariance, tuple) and len(covariance) == 2:
        cov_bins, cov_vals = covariance
        distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
        distances = jnp.linalg.norm(distances, axis=-1)
        return cov_lookup(distances, cov_bins, cov_vals)
    else:
        raise TypeError("covariance must be a tuple (cov_bins, cov_vals) or a callable cov(x1, x2).")


def _check_discretized(covariance):
    if callable(covariance) or not (isinstance(covariance, tuple) and len(covariance) == 2):
        raise TypeError("cuda=True requires a discretized covariance tuple (cov_bins, cov_vals).")


def cov_lookup(r, cov_bins, cov_vals):
    """
    Look up covariance in array of sampled `cov_vals` at radii `cov_bins` (equal-sized arrays).
    If `r` is inside of bounds, a linearly interpolated value is returned.
    If `r` is below the first bin, the first value is returned. But really the first bin should always be 0.0.
    If `r` is above the last bin, the last value is returned. Maybe the last value should be zero.
    """
    return jnp.interp(r, cov_bins, cov_vals)
