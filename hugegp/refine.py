import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

from .covariance import cov_lookup, cov_lookup_matrix

try:
    import hugegp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False


def generate_dense(points, covariance, xi):
    L = jnp.linalg.cholesky(cov_lookup_matrix(points, points, *covariance))
    values = L @ xi
    return values


def generate(
    graph,
    covariance,
    xi,
    *,
    cuda=False,
):
    points, neighbors, offsets, indices = graph
    cov_bins, cov_vals = covariance

    initial_cov = cov_lookup_matrix(points[: offsets[0]], points[: offsets[0]], cov_bins, cov_vals)
    initial_cholesky = jnp.linalg.cholesky(initial_cov)

    # Refine
    if cuda:
        if not has_cuda:
            raise ImportError("hugegp_cuda is not available")
        offsets = np.array(offsets, dtype=jnp.uint32)
        values = hugegp_cuda.refine(
            points, neighbors, offsets, cov_bins, cov_vals, initial_cholesky, xi[indices]
        )
    else:
        values = refine(points, neighbors, offsets, cov_bins, cov_vals, initial_cholesky, xi[indices])
    return values[jnp.argsort(indices)]


@Partial(jax.jit, static_argnums=(2,))
def refine(points, neighbors, offsets, cov_bins, cov_vals, initial_cholesky, xi):
    initial_values = initial_cholesky @ xi[: offsets[0]]
    values = [initial_values]
    offsets = offsets + (len(points),)

    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i + 1]

        fine_point = points[start:end][:, jnp.newaxis]
        fine_xi = xi[start:end]
        coarse_points = points[neighbors[start:end]]
        coarse_values = jnp.concatenate(values)[neighbors[start:end]]

        Kff = cov_lookup(jnp.array([0.0]), cov_bins, cov_vals)
        Kcc = cov_lookup_matrix(coarse_points, coarse_points, cov_bins, cov_vals)
        Kfc = cov_lookup_matrix(fine_point, coarse_points, cov_bins, cov_vals).squeeze(-2)

        mean = Kfc * jnp.linalg.solve(Kcc, coarse_values[..., jnp.newaxis]).squeeze(-1)
        mean = jnp.sum(mean, axis=-1)

        var = Kff - jnp.sum(Kfc * jnp.linalg.solve(Kcc, Kfc[..., jnp.newaxis]).squeeze(-1), axis=-1)
        std = jnp.sqrt(var)
        values.append(mean + std * fine_xi)

    return jnp.concatenate(values)
