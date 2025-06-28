import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

from .covariance import cov_lookup, cov_lookup_matrix, test_cov, test_cov_matrix

try:
    import hugegp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False


def generate(points, covariance, xi):
    L = jnp.linalg.cholesky(cov_lookup_matrix(points, points, *covariance))
    values = L @ xi
    return values


def generate_refine(
    graph,
    covariance,
    xi,
    *,
    cuda=False,
):
    points, neighbors, offsets = graph
    cov_bins, cov_vals = covariance

    # Generate initial values
    initial_values = generate(points[: offsets[0]], covariance, xi[: offsets[0]])

    # Refine
    if cuda:
        if not has_cuda:
            raise ImportError("hugegp_cuda is not available")
        offsets = np.array(offsets, dtype=jnp.uint32)
        return hugegp_cuda.refine(
            points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi
        )
    else:
        return refine(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi)


@Partial(jax.jit, static_argnums=(2,))
def refine(points, neighbors, offsets, cov_bins, cov_vals, initial_values, xi):
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

        # Kff = test_cov(jnp.array([0.0]))
        # Kcc = test_cov_matrix(coarse_points, coarse_points)
        # Kfc = test_cov_matrix(fine_point, coarse_points).squeeze(-2)

        mean = Kfc * jnp.linalg.solve(Kcc, coarse_values[..., jnp.newaxis]).squeeze(-1)
        mean = jnp.sum(mean, axis=-1)

        var = Kff - jnp.sum(Kfc * jnp.linalg.solve(Kcc, Kfc[..., jnp.newaxis]).squeeze(-1), axis=-1)
        # var = jnp.maximum(var, 1e-2)
        std = jnp.sqrt(var)
        values.append(mean + std * fine_xi)

    return jnp.concatenate(values)
