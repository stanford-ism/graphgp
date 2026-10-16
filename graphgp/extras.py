import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import gammaln


def matern_kernel(
    *,
    nu: float,
    variance: float,
    scale: float,
    r_min: float,
    r_max: float,
    n_scales: int,
) -> tuple[Array, Array]:
    """
    Matern covariance as a Gaussian scale mixture, discretized on ``n_scales`` logarithmically spaced scales between
    ``r_min`` and ``r_max``. Approximates

        ``C(r) = variance * 2^(1-nu) / Gamma(nu) * x^nu * K_nu(x)``, with ``x = sqrt(2 nu) r / scale``,

    for any ``nu > 0``. Scales outside the range are dropped, so the total variance ``sum(amplitudes)`` falls short
    of ``variance``. Differentiable with respect to ``nu``, ``variance`` and ``scale``.
    """
    scales = logspace_scales(r_min=r_min, r_max=r_max, n_scales=n_scales)
    h = np.log(r_max / r_min) / (n_scales - 1)
    x = nu * (scales / scale) ** 2  # s^2 in units of the Gamma scale
    # Gamma density in s^2 per unit log(s): 2 x^nu exp(-x) / Gamma(nu)
    amplitudes = variance * 2 * h * jnp.exp(nu * jnp.log(x) - x - gammaln(nu))
    return scales, amplitudes


def logspace_scales(*, r_min: float, r_max: float, n_scales: int) -> Array:
    """Logarithmically spaced Gaussian mixture scales between ``r_min`` and ``r_max`` inclusive."""
    return jnp.geomspace(r_min, r_max, n_scales)
