from typing import Callable, Tuple, Any, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


def compute_matern_covariance(
    r: Array, *, p: int = 0, sigma: float = 1.0, cutoff: float = 1.0, eps: float = 1e-5
) -> Array:
    """
    Matern covariance function for nu = p + 1/2. Power spectrum has -(nu + n/2) slope. Not differentiable with respect to ``p``.
    """
    x = jnp.sqrt(2 * p + 1) * r / cutoff
    i = jnp.arange(p + 1)
    log_coeff = (
        _log_factorial(p) + _log_factorial(p + i) - _log_factorial(i) - _log_factorial(p - i) - _log_factorial(2 * p)
    )
    polynomial = jnp.polyval(jnp.exp(log_coeff), 2 * x)
    result = sigma**2 * jnp.exp(-x) * polynomial
    result = jnp.where(r == 0.0, result * (1 + eps), result)
    return result


def compute_matern_covariance_discrete(
    *,
    p: int = 0,
    sigma: float = 1.0,
    cutoff: float = 1.0,
    eps: float = 1e-5,
    r_min: float = 1e-3,
    r_max: float = 1e3,
    n_bins: int = 1000,
) -> Tuple[Array, Array]:
    cov_bins = make_cov_bins(r_min=r_min, r_max=r_max, n_bins=n_bins)
    cov_vals = compute_matern_covariance(cov_bins, p=p, sigma=sigma, cutoff=cutoff, eps=eps)
    return (cov_bins, cov_vals)


def compute_cov_matrix(
    covariance: Tuple[Array, Array], points_a: Array, points_b: Array
) -> Array:
    """
    Compute the covariance matrix between two sets of points given a covariance function.
    """
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    if isinstance(covariance, Tuple) and isinstance(covariance[0], Array) and isinstance(covariance[1], Array):
        cov_bins, cov_vals = covariance
        return cov_lookup(distances, cov_bins, cov_vals)
    else:
        raise ValueError("Invalid covariance specification.")


def make_cov_bins(*, r_min: float, r_max: float, n_bins: int) -> Array:
    cov_bins = jnp.logspace(jnp.log10(r_min), jnp.log10(r_max), n_bins - 1)
    cov_bins = jnp.concatenate((jnp.array([0.0]), cov_bins), axis=0)
    return cov_bins


def cov_lookup(r, cov_bins, cov_vals):
    """
    Look up covariance in array of sampled `cov_vals` at radii `cov_bins` (equal-sized arrays).
    If `r` is inside of bounds, a linearly interpolated value is returned.
    If `r` is below the first bin, the first value is returned. But really the first bin should always be 0.0.
    If `r` is above the last bin, the last value is returned. Maybe the last value should be zero.
    """
    # interpolate between bins
    idx = jnp.searchsorted(cov_bins, r)
    # return cov_vals[idx]
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


def _log_factorial(x):
    return gammaln(x + 1)
