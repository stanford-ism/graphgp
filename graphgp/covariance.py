from dataclasses import dataclass, field
from typing import Callable, Tuple, Any, Union

import jax
import jax.numpy as jnp
from jax.tree_util import Partial, register_dataclass
from jax import Array
from jax.scipy.special import gammaln

CovarianceType = Union[Callable[[Array], Array], Tuple[Array, Callable[[Array], Array]], Tuple[Array, Array]]


@register_dataclass
@dataclass
class MaternCovariance:
    """
    Matern covariance function for nu = p + 1/2. Power spectrum has -(nu + n/2) slope. Not differentiable with respect to ``p``. Simply calls ``compute_matern_covariance``.

    This dataclass can be passed to jit-compiled functions since ``p`` is marked as static. It can be called just like a normal function.
    """

    p: int = field(default=0, metadata=dict(static=True))
    eps: float = field(default=1e-5, metadata=dict(static=True))

    def __call__(self, r: Array, *, sigma: float = 1.0, cutoff: float = 1.0):
        return compute_matern_covariance(r, p=self.p, sigma=sigma, cutoff=cutoff, eps=self.eps)


def compute_matern_covariance(
    r: Array, *, p: int = 0, sigma: float = 1.0, cutoff: float = 1.0, eps: float = 1e-5
) -> Array:
    """
    Matern covariance function for nu = p + 1/2. Power spectrum has -(nu + n/2) slope. Not differentiable with respect to ``p``.

    Cannot be passed to jit-compiled functions. Use ``MaternCovariance`` object in that case.
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


def prepare_matern_covariance_discrete(
    *, p: int = 0, sigma: float = 1.0, cutoff: float = 1.0, eps: float = 1e-5, r_min: float, r_max: float, n_bins: int
) -> Tuple[Array, Callable[[Array], Array]]:
    cov_bins = make_cov_bins(r_min=r_min, r_max=r_max, n_bins=n_bins)
    return (cov_bins, Partial(MaternCovariance(p=p, eps=eps), sigma=sigma, cutoff=cutoff))


def _log_factorial(x):
    return gammaln(x + 1)


def compute_cov_matrix(
    covariance: Tuple[Array, Array] | Tuple[Array, Callable] | Callable, points_a: Array, points_b: Array
) -> Array:
    """
    Compute the covariance matrix between two sets of points given a covariance function.
    """
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    if isinstance(covariance, Callable):
        return covariance(distances)
    elif isinstance(covariance, Tuple) and isinstance(covariance[0], Array) and isinstance(covariance[1], Callable):
        return covariance[1](distances)
    elif isinstance(covariance, Tuple) and isinstance(covariance[0], Array) and isinstance(covariance[1], Array):
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
    return jax.numpy.interp(r, cov_bins, cov_vals)
