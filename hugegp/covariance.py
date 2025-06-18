import jax.numpy as jnp


def test_cov(r):
    result = (1 + (r / 0.2) ** 2) ** (-1)
    result = jnp.where(r == 0.0, result * (1 + 1e-4), result)
    return result


def test_cov_matrix(points_a, points_b=None):
    if points_b is None:
        points_b = points_a
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    return test_cov(distances)
