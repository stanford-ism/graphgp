from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial, register_dataclass
from jax import Array
from jax import lax

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.special import jv, gamma, gammaln

    has_scipy = True
except ImportError:
    has_scipy = False

def matern_kernel(
    *,
    p: int = 0,
    variance: float = 1.0,
    cutoff: float = 1.0,
    r_min: float = 1e-5,
    r_max: float = 1e1,
    n_bins: int = 1_000,
    jitter: float = 1e-5,
) -> Tuple[Array, Array]:
    """
    Matern covariance function for nu = p + 1/2. Power spectrum has -(nu + n/2) slope. Not differentiable with respect to ``p``.

    Discretized onto `n_bins` logarithmically spaced bins between `r_min` and `r_max`, with 0.0 included as the first bin.
    """
    r = make_cov_bins(r_min=r_min, r_max=r_max, n_bins=n_bins)
    x = jnp.sqrt(2 * p + 1) * r / cutoff
    i = jnp.arange(p + 1)
    log_coeff = (
        _log_factorial(p) + _log_factorial(p + i) - _log_factorial(i) - _log_factorial(p - i) - _log_factorial(2 * p)
    )
    polynomial = jnp.polyval(jnp.exp(log_coeff), 2 * x)
    cov = variance * jnp.exp(-x) * polynomial
    cov = jnp.where(r == 0.0, cov[0] * (1.0 + jitter), cov)
    return (r, cov)


def _log_factorial(x):
    return gammaln(x + 1)


def make_cov_bins(*, r_min: float, r_max: float, n_bins: int) -> Array:
    cov_bins = jnp.logspace(jnp.log10(r_min), jnp.log10(r_max), n_bins - 1)
    cov_bins = jnp.concatenate((jnp.array([0.0]), cov_bins), axis=0)
    return cov_bins


def covariance_from_spectrum(cov_bins, logk, *, d):
    """
    Create a function which converts a discretized power spectrum to a discretized covariance.
    Certain values must be precomputed outside of JAX, hence the factory function.
    Specifically, this computes the integrals analytically using Bessel functions assuming piecewise constant P(k).

    Args:
        cov_bins: Radial bins at which to evaluate the covariance. First entry should probably be zero.
        logk: Logarithmic k values defining the upper edges of the power spectrum bins. The first bin is assumed to start at k=0.
        d: Dimensionality of the space.

    Returns:
        cov_func: Callable taking logarithmic power spectrum in the defined k bins, plus overall variance kwarg, returning covariance values at cov_bins.
    """

    # Precompute Hankel matrix (uses SciPy)
    k = jnp.concatenate([jnp.array([0.0]), jnp.exp(logk)])
    hankel_matrix = compute_hankel_matrix(k, cov_bins, d=d)

    def cov_func(logp, *, variance):
        cov_vals = hankel_matrix @ jnp.exp(logp)
        cov_vals = variance * (cov_vals / cov_vals[0])

    return cov_func


def compute_hankel_matrix(k, r, *, d):
    """
    Compute matrix to convert P(k) to C(r) via Hankel transform in d dimensions.
    Assumes P(k) is piecewise constant between k bins, output will have shape (len(r), len(k)-1).
    Requires scipy for Bessel functions, so this cannot be differentiated through.
    """
    r = r[:, None]
    limits = (k / (2 * np.pi * r)) ** (d / 2) * jv(d / 2, k * r)
    zero = (k / 2) ** d / (np.pi ** (d / 2) * gamma(d / 2 + 1))
    weights = np.where(r > 0, limits[:, 1:] - limits[:, :-1], zero[None, 1:] - zero[None, :-1])
    return weights
