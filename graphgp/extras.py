from typing import Tuple
from functools import partial

from graphgp import Graph, build_graph, generate, generate_inv
import jax
import jax.numpy as jnp
from jax import Array

try:
    from jax.scipy.special import gammaln

    has_scipy = True
except ImportError:
    has_scipy = False


def build_and_generate_conditional(
    points_a: Array,
    points_b: Array,
    covariance: Tuple[Array, Array],
    values_a: Array,
    xi: Array,
    n0: int,
    k: int,
    cuda: bool = False,
    fast_jit: bool = True,
):
    """
    Helper function to build graphs and generate a conditional GP realization. If graphs can be reused, use ``generate_conditional`` directly.
    """
    conditioning_graph = build_graph(points_a, n0=n0, k=k, cuda=cuda)
    joint_graph = build_graph(jnp.concatenate((points_a, points_b), axis=0), n0=n0, k=k, cuda=cuda)
    return generate_conditional(
        conditioning_graph,
        joint_graph,
        covariance,
        values_a,
        xi,
        cuda=cuda,
        fast_jit=fast_jit,
    )


def generate_conditional(
    conditioning_graph: Graph,
    joint_graph: Graph,
    covariance: Tuple[Array, Array],
    conditioning_values: Array,
    joint_xi: Array,
    cuda: bool = False,
    fast_jit: bool = True,
):
    """
    Generate a GP realization at N points conditioned on the values at M points.
    In order to reuse existing GraphGP components, a graph for the M conditioning points and
    a graph for the full set of M + N points must be provided, with the conditioning points first in the order.
    We generate an independent realization at the M + N points and then apply a correction to match the M values.
    The conditioning assumes the GraphGP approximation is correct. This is not the most efficient way to generate
    conditionals but is simple in that it only relies on the existing ``generate`` and ``generate_inv`` functions.

    Args:
        conditioning_graph: Graph for the M conditioning points.
        joint_graph: Graph for the full set of N + M points. The conditioning points must be first in the order.
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        conditioning_values: Values at the M conditioning points of shape (M,).
        joint_xi: Standard normal random variables for the N points of shape (N + M,). Note that we need extra random variables for this approach!
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer slightly.
    Returns:
        Array of shape (N,) with the generated values at the N points.
    """
    M = conditioning_graph.points.shape[0]
    N = joint_graph.points.shape[0] - M

    if joint_graph.indices is None or conditioning_graph.indices is None:
        raise ValueError("Both joint_graph and conditioning_graph must have indices defined as point order determines conditioning.")
    
    # Sample GP and measure difference at conditioning points
    random_joint_values = generate(joint_graph, covariance, joint_xi, cuda=cuda, fast_jit=fast_jit)
    value_residual = conditioning_values - random_joint_values[:M]
    
    # Correct parameters to match conditioning values
    inv_sqrt = partial(generate_inv, conditioning_graph, covariance, cuda=cuda)
    xi_residual, vjp_func = jax.vjp(inv_sqrt, value_residual)
    xi_residual = vjp_func(xi_residual)[0]

    sqrt = partial(generate, joint_graph, covariance, cuda=cuda, fast_jit=fast_jit)
    correction, vjp_func = jax.vjp(sqrt, jnp.concatenate([xi_residual, jnp.zeros(N)], axis=0))
    correction = vjp_func(correction)[0]
    joint_values = random_joint_values + correction

    # Check we got the conditioning values right and return the corrected values at the N points
    # assert jnp.allclose(joint_values[:M], conditioning_values)
    return joint_values



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
