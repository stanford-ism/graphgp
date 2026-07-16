from typing import Tuple

import jax.numpy as jnp
from jax import Array

try:
    from jax.scipy.special import gammaln

    has_scipy = True
except ImportError:
    has_scipy = False


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
    r = make_cov_bins(r_min=r_min, r_max=r_max, n_bins=n_bins)
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


def make_div_free_rbf_covariance(
    *,
    ell: float,
    sigma2: float = 1.0,
    periodic: bool = True,
):
    """
    Divergence-free 3x3 magnetic-field covariance tensor

        M_ij(x, y) = sigma2 * (a(r) delta_ij + b(r) r_i r_j)

    with

        r = y - x,
        r <- r - round(r)   if periodic,
        a(r) = (1 - 0.5 * r^2 / ell^2) * exp(-0.5 * r^2 / ell^2),
        b(r) = (0.5 / ell^2) * exp(-0.5 * r^2 / ell^2).

    The returned callable also carries metadata understood by the CUDA vector
    extension so that ``gp.generate_vector(..., cuda=True)`` can dispatch to the
    custom kernel without changing the high-level API.
    """
    ell = float(ell)
    sigma2 = float(sigma2)
    periodic = bool(periodic)
    ell2 = ell * ell

    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)
        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))

    cov_elem._graphgp_cuda_kernel = "div_free_rbf_vector_3"
    cov_elem._graphgp_cuda_ell = ell
    cov_elem._graphgp_cuda_sigma2 = sigma2
    cov_elem._graphgp_cuda_periodic = periodic
    return cov_elem
