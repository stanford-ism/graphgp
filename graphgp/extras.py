import jax
import jax.numpy as jnp
from jax.tree_util import Partial, register_dataclass
from jax import Array
from jax import lax

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.special import jv, gamma

    has_scipy = True
except ImportError:
    has_scipy = False


def solve_cg(A, b, x0, *, steps, tol=None, M=None, restart=None):
    """
    Conjugate gradient solver for Ax = b.

    Args:
        A: Callable which computes the matrix vector product Ax.
        b: Right-hand side vector.
        x0: Initial guess for the solution.
        steps: Maximum number of iterations.
        tol: Relative tolerance for convergence, |r| / |b| < tol.
        M: Preconditioner which approximates the inverse of A.
        restart: Interval for recomputing residual exactly.

    Returns:
        x: Approximate solution vector.
        info: Tuple containing number of iterations and final relative residual norm.
    """

    def cond_fun(carry):
        i, x, r, p, z = carry
        return (i < steps) & (jnp.linalg.norm(r) > tol * jnp.linalg.norm(b))

    def body_fun(carry):
        i, x, r, p, z = carry
        A_p = A(p)
        alpha = jnp.dot(r, z) / jnp.dot(p, A_p)
        x_new = x + alpha * p
        if restart is None:
            r_new = r - alpha * A_p
        else:
            should_restart = jnp.mod(i + 1, restart) == 0
            r_new = jnp.where(should_restart, b - A(x_new), r - alpha * A_p)
        z_new = r_new if M is None else M(r_new)
        beta = jnp.dot(r_new, z_new) / jnp.dot(r, z)
        p_new = z_new + beta * p
        return i + 1, x_new, r_new, p_new, z_new

    def loop_fun(i, carry):
        return body_fun((i,) + carry)[1:]

    r0 = b - A(x0)
    z0 = r0 if M is None else M(r0)
    p0 = z0
    if tol is None:  # in this case vjp is supported
        x, r, p, z = lax.fori_loop(0, steps, loop_fun, (x0, r0, p0, z0))
        return x, (steps, jnp.linalg.norm(r) / jnp.linalg.norm(b))
    else:
        i, x, r, p, z = lax.while_loop(cond_fun, body_fun, (0, x0, r0, p0, z0))
        return x, (i, jnp.linalg.norm(r) / jnp.linalg.norm(b))


def newton_cg(
    f,
    metric,
    x0,
    *,
    newton_steps,
    newton_tol=None,
    cg_steps,
    cg_tol=None,
    line_search_steps=0,
):
    """
    Minimize a nonlinear function, at each step using conjugate gradient to solve M dx = -grad(f).
    A very simply backtracking line search can be enabled by setting line_search_steps > 0.
    """
    value_and_grad = jax.value_and_grad(f)

    def cond(carry):
        i, x, tol = carry
        if newton_tol is None:
            return i < newton_steps
        else:
            return (tol > newton_tol) & (i < newton_steps)

    def step(carry):
        i, x, tol = carry
        value, grad = value_and_grad(x)
        direction, _ = solve_cg(Partial(metric, x), -grad, jnp.zeros_like(x), steps=cg_steps, tol=cg_tol)
        if line_search_steps == 0:
            dx = direction
        else:
            dx, _, n_steps = _simple_line_search(f, x, direction, f_x0=value, steps=line_search_steps)
            jax.debug.print("Line search took {}/{} steps", n_steps, line_search_steps)
        tol = jnp.linalg.norm(grad)
        return i + 1, x + dx, tol

    i, x, tol = lax.while_loop(cond, step, (0, x0, jnp.inf))
    return x


def _simple_line_search(f, x0, direction, *, steps, f_x0=None, reduction=0.5):
    if f_x0 is None:
        f_x0 = f(x0)

    def cond(carry):
        i, alpha, value = carry
        return (value >= f_x0) & (i < steps)

    def step(carry):
        i, alpha, value = carry
        new_alpha = reduction * alpha
        new_value = f(x0 + new_alpha * direction)
        return i + 1, new_alpha, new_value

    value = f(x0 + direction)
    i, alpha, value = lax.while_loop(cond, step, (0, 1.0, value))
    return alpha * direction, value, i


def draw_linear_sample(transformation, x0, prior_noise, data_noise, *, cg_steps, cg_tol=None):
    """Sample from posterior using metric at x0 and linear condition M x = p + R d"""

    def A(x):
        return x + _metric(transformation, x0, x)

    b = prior_noise + _pull(transformation, x0, data_noise)
    sample, (i, r) = solve_cg(A, b, x0=prior_noise, steps=cg_steps, tol=cg_tol)
    return sample, (i, r)


def draw_linear_samples(transformation, x0, prior_noise, data_noise, *, cg_steps, cg_tol=None):
    samples, (i, r) = jax.vmap(Partial(draw_linear_sample, transformation, x0, cg_steps=cg_steps, cg_tol=cg_tol))(
        prior_noise, data_noise
    )
    return samples, (i, r)


def mgvi(
    energy,
    transformation,
    x0,
    prior_noise,
    data_noise,
    *,
    kl_steps,
    kl_tol=None,
    newton_kwargs,
    sampling_kwargs,
):
    def kl(samples, x0):
        return jnp.mean(jax.vmap(energy)(x0 + samples))

    def cond(carry):
        i, x, tol = carry
        if kl_tol is None:
            return i < kl_steps
        else:
            return (tol > kl_tol) & (i < kl_steps)

    def step(carry):
        i, x, tol = carry
        samples = draw_linear_samples(transformation, x, prior_noise, data_noise, **sampling_kwargs)
        x_new = newton_cg(Partial(kl, samples), x, **newton_kwargs)
        tol = jnp.linalg.norm(x_new - x) / jnp.linalg.norm(x)
        return i + 1, x_new, tol

    i, x, tol = lax.while_loop(cond, step, (0, x0, jnp.inf))
    return x


@register_dataclass
@dataclass
class GaussianLikelihood:
    mean: Array
    variance: Array

    def energy(self, x):
        return 1/2 * jnp.sum((x - self.mean) ** 2 / self.variance)
    
    def transformation(self, x):
        return x / jnp.sqrt(self.variance)



def compute_hankel_matrix(k, r, *, d):
    """
    Compute matrix to convert P(k) to C(r) via Hankel transform in d dimensions.
    Assumes P(k) is piecewise constant between k bins, output will have shape (len(r), len(k)-1).
    Requires scipy for Bessel functions, so this cannot be differentiated through.
    """
    r = jnp.asarray(r[:, None])
    limits = (k / (2 * np.pi * r)) ** (d / 2) * jv(d / 2, k * r)
    limits_if_zero = (k / 2) ** d / (np.pi ** (d / 2) * gamma(d / 2 + 1))
    weights = jnp.where(r > 0, limits[:, 1:] - limits[:, :-1], limits_if_zero[None, 1:] - limits_if_zero[None, :-1])
    return weights


def flexible_spectrum(logk, xi, *, slope, fluctuations):
    """
    Extremely minimal version of NIFTY's CorrelatedField.
    A power law spectrum with integrated Wiener process fluctuations.
    The overall power is not normalized yet.
    """
    deviations = jnp.cumsum(jnp.cumsum(xi)) / len(xi)
    logp = slope * logk + fluctuations * deviations
    logp = logp - logp[0]
    return logp


def flexible_covariance(cov_bins, logk, xi, *, variance, slope, fluctuations, d):
    """
    Extremely minimal version of NIFTY's CorrelatedField, in covariance form.
    A power law spectrum with integrated Wiener process fluctuations.
    Partial evaluation with fixed cov_bins, logk, and d is need before differentiation.
    """

    # Precompute Hankel matrix (uses SciPy)
    k = jnp.concatenate([jnp.array([0.0]), jnp.exp(logk)])
    hankel_matrix = compute_hankel_matrix(k, cov_bins, d=d)

    # Compute covariance from power spectrum
    logp = flexible_spectrum(logk, xi, slope=slope, fluctuations=fluctuations)
    cov_vals = hankel_matrix @ jnp.exp(logp)
    cov_vals = variance * (cov_vals / cov_vals[0])
    return cov_vals


def _push(f, x0, x):
    """Pushforward of transformation at x0 applied to x"""
    return jax.jvp(f, (x0,), (x,))[1]


def _pull(f, x0, d):
    """Pullback of transformation at x0 applied to d"""
    return jax.vjp(f, x0)[1](d)[0]


def _metric(f, x0, x):
    """Metric tensor at x0 applied to x"""
    return _pull(f, x0, _push(f, x0, x))
