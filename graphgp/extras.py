from typing import Tuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln
import numpy as np

try:
    from scipy.special import jv, gamma

    has_scipy = True
except ImportError:
    has_scipy = False


def covariance_from_spectrum(radial_bins, k_bins, *, d, normalize=True):
    """
    Create a function which converts a discretized power spectrum to a discretized isotropic covariance.
    Certain values must be precomputed outside of JAX, hence the factory function.
    Specifically, this computes the integrals analytically using Bessel functions assuming piecewise constant P(k).

    Args:
        radial_bins: Bins at which to evaluate the covariance.
        k_bins: Edges of the power spectrum bins defining the isotropic part of the corresponding harmonic space.
        d: Dimensionality of the space.
        normalize: Whether to return normalized kernels or not (Default True)

    Returns:
        cov_func: Callable taking the power spectrum in defined k bins, returning covariance values at cov_bins.
    """

    # Precompute Hankel matrix (uses SciPy)
    hankel_matrix = _hankel_matrix(k_bins, radial_bins, d=d)

    def cov_func(spectrum):
        spectrum = 0.5 * (spectrum[1:] + spectrum[:-1])
        cov_vals = hankel_matrix @ spectrum
        return radial_bins, cov_vals / cov_vals[0] if normalize else cov_vals

    return cov_func


def _hankel_matrix(k, r, *, d):
    """
    Compute matrix to convert P(k) to C(r) via Hankel transform in d dimensions.
    Assumes P(k) is piecewise constant between k bins, output will have shape (len(r), len(k)-1).
    Requires scipy for Bessel functions, so this cannot be differentiated through.
    """
    r = r[:, None]
    limits = (k / (2 * np.pi * r)) ** (d / 2) * jv(d / 2, k * r)
    zero = (k / 2) ** d / (np.pi ** (d / 2) * gamma(d / 2 + 1))
    weights = np.where(
        r > 0, limits[:, 1:] - limits[:, :-1], zero[None, 1:] - zero[None, :-1]
    )
    return weights


def rbf_kernel(
    *,
    variance: float,
    scale: float,
    r_min: float,
    r_max: float,
    n_bins: int,
    jitter: float = 0.0,
) -> Tuple[Array, Array]:
    """
    Radial basis function (squared exponential) covariance.

    Discretized onto `n_bins` logarithmically spaced bins between `r_min` and `r_max`, with 0.0 included as the first bin.
    """
    r = logspace_radial_bins(r_min=r_min, r_max=r_max, n_bins=n_bins)
    cov = variance * jnp.exp(-1 / 2 * (r / scale) ** 2)
    cov = jnp.where(r == 0.0, cov[0] * (1.0 + jitter), cov)
    return (r, cov)


def matern_kernel(
    *,
    p: int,
    variance: float,
    cutoff: float,
    r_min: float,
    r_max: float,
    n_bins: int,
    jitter: float = 0.0,
) -> Tuple[Array, Array]:
    """
    Matern covariance function for nu = p + 1/2. Power spectrum has -(nu + n/2) slope. Not differentiable with respect to ``p``.

    Discretized onto `n_bins` logarithmically spaced bins between `r_min` and `r_max`, with 0.0 included as the first bin.
    """
    r = logspace_radial_bins(r_min=r_min, r_max=r_max, n_bins=n_bins)
    x = jnp.sqrt(2 * p + 1) * r / cutoff
    i = jnp.arange(p + 1)
    log_coeff = (
        _log_factorial(p)
        + _log_factorial(p + i)
        - _log_factorial(i)
        - _log_factorial(p - i)
        - _log_factorial(2 * p)
    )
    polynomial = jnp.polyval(jnp.exp(log_coeff), 2 * x)
    cov = variance * jnp.exp(-x) * polynomial
    cov = jnp.where(r == 0.0, cov[0] * (1.0 + jitter), cov)
    return (r, cov)


def _log_factorial(x):
    return gammaln(x + 1)


def logspace_radial_bins(*, r_min: float, r_max: float, n_bins: int, base=10) -> Array:
    cov_bins = jnp.logspace(
        jnp.log10(r_min) / jnp.log10(base),
        jnp.log10(r_max) / jnp.log10(base),
        n_bins - 1,
        base=base,
    )
    cov_bins = jnp.concatenate((jnp.array([0.0]), cov_bins), axis=0)
    return cov_bins


def logspace_k_bins(*, r_min: float, r_max: float, n_bins: int, base=10) -> Array:
    return logspace_radial_bins(
        r_min=1.0 / r_max, r_min=1.0 / r_min, n_bins=n_bins, base=base
    )
