import jax.numpy as jnp


def cov_lookup(r, cov_bins, cov_vals):
    """
    Look up covariance in array of sampled `cov_vals` at radii `cov_bins` (equal-sized arrays).
    If `r` is inside of bounds, a linearly interpolated value is returned.
    If `r` is below the first bin, the first value is returned. But really the first bin should always be 0.0.
    If `r` is above the last bin, the last value is returned. Maybe the last value sohuld be zero.
    """
    # interpolate between bins
    idx = jnp.searchsorted(cov_bins, r)
    r0 = cov_bins[idx - 1]
    r1 = cov_bins[idx]
    c0 = cov_vals[idx - 1]
    c1 = cov_vals[idx]
    c = c0 + (c1 - c0) * (r - r0) / (r1 - r0)

    # handle edge cases
    c = jnp.where(idx == 0, c1, c)
    c = jnp.where(idx == len(cov_bins), c0, c)
    c = jnp.where(r0 == r1, c0, c)
    return c


def cov_lookup_matrix(points_a, points_b, cov_bins, cov_vals):
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    return cov_lookup(distances, cov_bins, cov_vals)



def test_cov(r, *, cutoff=0.2, slope=-1.0, scale=1.0):
    result = scale * (1 + (r / cutoff) ** 2) ** (slope)
    result = jnp.where(r == 0.0, result * (1 + 1e-4), result)
    return result

def test_cov_discretized(r_min, r_max, n_bins, *, cutoff=0.2, slope=-1.0, scale=1.0):
    cov_bins = jnp.logspace(jnp.log10(r_min), jnp.log10(r_max), n_bins)
    cov_bins = cov_bins.at[0].set(0.0)
    return cov_bins, test_cov(cov_bins, cutoff=cutoff, slope=slope, scale=scale)

def test_cov_matrix(points_a, points_b=None, cutoff=0.2, slope=-1.0, scale=1.0):
    if points_b is None:
        points_b = points_a
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    return test_cov(distances, cutoff=cutoff, slope=slope, scale=scale)