import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from .covariance import test_cov, test_cov_matrix

try:
    import hugegp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False


def generate_static(
    graph,
    xi,
    cuda=False,
):
    points, indices, neighbors, level_offsets = graph

    # Generate initial values
    L = jnp.linalg.cholesky(test_cov_matrix(points[indices[: level_offsets[0]]]))
    initial_values = L @ xi[indices[: level_offsets[0]]]

    cov_distances = jnp.zeros(1)
    cov_values = jnp.zeros(1)

    # Refine
    if cuda:
        if not has_cuda:
            raise ImportError("hugegp_cuda is not available")
        return hugegp_cuda.refine_static(
            points, xi, indices, neighbors, level_offsets, cov_distances, cov_values, initial_values
        )
    else:
        offsets = tuple(int(li) for li in level_offsets)
        return refine_static_impl(
            points, xi, indices, neighbors, offsets, cov_distances, cov_values, initial_values
        )


@Partial(jax.jit, static_argnums=(4,))
def refine_static_impl(
    points, xi, indices, neighbors, level_offsets, cov_distances, cov_values, initial_values
):
    values = [initial_values]
    level_offsets = level_offsets + (len(points),)

    for i in range(len(level_offsets) - 1):
        start = level_offsets[i]
        end = level_offsets[i + 1]

        fine_point = points[indices[start:end]]
        fine_xi = xi[indices[start:end]]
        coarse_points = points[indices[neighbors[start:end]]]
        coarse_values = jnp.concatenate(values)[neighbors[start:end]]

        Kff = test_cov(0.0)
        Kcc = test_cov_matrix(coarse_points, coarse_points)
        Kfc = test_cov_matrix(fine_point[:, jnp.newaxis], coarse_points).squeeze(-2)

        mean = Kfc * jnp.linalg.solve(Kcc, coarse_values[..., jnp.newaxis]).squeeze(-1)
        mean = jnp.sum(mean, axis=-1)

        var = Kff - jnp.sum(Kfc * jnp.linalg.solve(Kcc, Kfc[..., jnp.newaxis]).squeeze(-1), axis=-1)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        values.append(mean + std * fine_xi)

    return jnp.concatenate(values)[jnp.argsort(indices)]
