import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

from .covariance import test_cov, test_cov_matrix

try:
    import hugegp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False


def generate(points, xi):
    L = jnp.linalg.cholesky(test_cov_matrix(points))
    values = L @ xi
    return values


def generate_refine(
    graph,
    xi,
    cuda=False,
):
    points, neighbors, level_offsets = graph

    # Generate initial values
    L = jnp.linalg.cholesky(test_cov_matrix(points[: level_offsets[0]]))
    initial_values = L @ xi[: level_offsets[0]]

    cov_r, cov = (jnp.zeros(1), jnp.zeros(1))

    # Refine
    if cuda:
        if not has_cuda:
            raise ImportError("hugegp_cuda is not available")
        level_offsets = np.array(level_offsets, dtype=np.uint32)
        return hugegp_cuda.refine(points, xi, neighbors, level_offsets, initial_values, cov_r, cov)
    else:
        return refine(points, xi, neighbors, level_offsets, initial_values, cov_r, cov)


@Partial(jax.jit, static_argnums=(3,))
def refine(points, xi, neighbors, level_offsets, initial_values, cov_r, cov):
    values = [initial_values]
    level_offsets = level_offsets + (len(points),)

    for i in range(len(level_offsets) - 1):
        start = level_offsets[i]
        end = level_offsets[i + 1]

        fine_point = points[start:end]
        fine_xi = xi[start:end]
        coarse_points = points[neighbors[start:end]]
        coarse_values = jnp.concatenate(values)[neighbors[start:end]]

        Kff = test_cov(0.0)
        Kcc = test_cov_matrix(coarse_points, coarse_points)
        Kfc = test_cov_matrix(fine_point[:, jnp.newaxis], coarse_points).squeeze(-2)

        mean = Kfc * jnp.linalg.solve(Kcc, coarse_values[..., jnp.newaxis]).squeeze(-1)
        mean = jnp.sum(mean, axis=-1)

        var = Kff - jnp.sum(Kfc * jnp.linalg.solve(Kcc, Kfc[..., jnp.newaxis]).squeeze(-1), axis=-1)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        values.append(mean + std * fine_xi)

    return jnp.concatenate(values)
