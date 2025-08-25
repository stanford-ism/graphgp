import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial
import numpy as np

from .covariance import cov_lookup, cov_lookup_matrix, compute_cov_matrix

try:
    import hugegp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False


def generate(points, neighbors, offsets, covariance, xi):
    """Initial layer of dense generation followed by refinement."""
    initial_values = generate_dense(points[: offsets[0]], covariance, xi[: offsets[0]])
    values = refine(points, neighbors, offsets, covariance, initial_values, xi[offsets[0] :])
    return values


def generate_inv(points, neighbors, offsets, covariance, values):
    """Inverse of `generate` with respect to `xi`."""
    initial_values, xi = refine_inv(points, neighbors, offsets, covariance, values)
    initial_xi = generate_dense_inv(points[: offsets[0]], covariance, initial_values)
    return jnp.concatenate([initial_xi, xi], axis=0)


def generate_logdet(points, neighbors, offsets, covariance):
    n0 = offsets[0]
    return generate_dense_logdet(points[:n0], covariance) + refine_logdet(points, neighbors, offsets, covariance)


def generate_dense(points, covariance, xi):
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    values = L @ xi
    return values


def generate_dense_inv(points, covariance, values):
    """Inverse of `generate_dense` with respect to `xi`."""
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    xi = jnp.linalg.solve(L, values)
    return xi


def generate_dense_logdet(points, covariance):
    K = compute_cov_matrix(covariance, points, points)
    return jnp.linalg.slogdet(K)[1] / 2


def refine(points, neighbors, offsets, covariance, initial_values, xi):
    n0 = offsets[0]
    values = initial_values
    offsets = (0,) + offsets
    for i in range(2, len(offsets)):
        start = offsets[i - 1]
        end = offsets[i]
        coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
        coarse_values = jnp.take(values, neighbors[start - n0 : end - n0], axis=0)
        fine_point = points[start:end]
        fine_xi = xi[start - n0 : end - n0]
        mean, std = jax.vmap(Partial(conditional_gaussian, covariance))(coarse_points, coarse_values, fine_point)
        values = jnp.concatenate([values, mean + std * fine_xi], axis=0)
    return values


def refine_inv(points, neighbors, offsets, covariance, values):
    """Inverse of `refine` with respect to `xi`."""
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
        mean, std = jax.vmap(Partial(conditional_gaussian, covariance))(coarse_points, coarse_values, fine_point)
        xi = jnp.concatenate([(fine_value - mean) / std, xi], axis=0)
    return initial_values, xi


def refine_logdet(points, neighbors, offsets, covariance):
    logdet = 0.0
    n0 = offsets[0]
    offsets = (0,) + offsets
    for i in range(2, len(offsets)):
        start = offsets[i - 1]
        end = offsets[i]
        coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
        fine_point = points[start:end]
        std = jax.vmap(Partial(conditional_std, covariance))(coarse_points, fine_point)
        logdet += jnp.sum(jnp.log(std))
    return logdet


def conditional_gaussian(covariance, coarse_points, coarse_values, fine_point):
    """Basic building block computes the conditional mean and standard deviation for a fine point."""
    k = len(coarse_points)
    joint_points = jnp.concatenate([coarse_points, fine_point[jnp.newaxis]], axis=0)
    K = compute_cov_matrix(covariance, joint_points, joint_points)
    L = jnp.linalg.cholesky(K)
    mean = L[k, :k] @ jnp.linalg.solve(L[:k, :k], coarse_values)
    std = L[k, k]
    return mean, std


def conditional_std(cov_func, coarse_points, fine_point):
    k = len(coarse_points)
    joint_points = jnp.concatenate([coarse_points, fine_point[jnp.newaxis]], axis=0)
    K = compute_cov_matrix(cov_func, joint_points, joint_points)
    L = jnp.linalg.cholesky(K)
    return L[k, k]

    # # Refine
    # if cuda:
    #     if not has_cuda:
    #         raise ImportError("hugegp_cuda is not available")
    #     offsets = np.array(offsets, dtype=jnp.uint32)
    #     values = hugegp_cuda.refine(
    #         points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi[indices][offsets[0]:]
    #     )
    # else:
    #     values = refine(points, neighbors, offsets, cov_func, initial_values, xi[indices])
    # return values[jnp.argsort(indices)]
