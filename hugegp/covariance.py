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


def test_cov_sampled():
    cov_bins = jnp.logspace(-3, 2, 1024)
    cov_bins = cov_bins.at[0].set(0.0)
    return cov_bins, test_cov(cov_bins)


def cov_lookup(r, cov_bins, cov_vals):
    """
    Look up covariance in array of sampled `cov_vals` at radii `cov_bins` (equal-sized arrays).
    If `r` is inside of bounds, a linearly interpolated value is return.
    If `r` is below the first bin, the first value is returned. But really the first bin should always be 0.0.
    If `r` is above the last bin, 0.0 is returned.
    """
    idx = jnp.maximum(jnp.searchsorted(cov_bins, r) - 1, 0)
    r0 = cov_bins[idx]
    r1 = cov_bins[idx + 1]
    c0 = cov_vals[idx]
    c1 = cov_vals[idx + 1]
    c = c0 + (c1 - c0) * jnp.maximum(r - r0, 0) / (r1 - r0)
    return jnp.where(idx < len(cov_bins) - 1, c, 0.0)


def cov_lookup_matrix(points_a, points_b, cov_bins, cov_vals):
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    return cov_lookup(distances, cov_bins, cov_vals)
