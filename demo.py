# %%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import graphgp as gp

rng = jr.key(99)

#%% Generate some random points, build graph, and define covariance
n_points = 10_000
rng, key = jr.split(rng)
points = jr.normal(key, (n_points, 2))
graph = gp.build_graph(points, n0=100, k=10)
covariance = gp.extras.rbf_kernel(variance=1.0, scale=0.3, r_min=1e-4, r_max=10.0, n_bins=1000, jitter=1e-5)

#%% Generate a random GP realization and plot!
rng, key = jr.split(rng)
xi = jr.normal(key, (n_points,))
values = gp.generate(graph, covariance, xi)
assert jnp.sum(jnp.isnan(values)) == 0, "Generated values contain NaNs"

plt.scatter(*points.T, c=values, s=1)
plt.gca().set(aspect="equal", xlim=(-4, 4), ylim=(-4, 4))
plt.colorbar()
plt.show()

#%% Test inverse for Philipp
values = gp.generate(graph, covariance, xi, cuda=False)
values_cuda = gp.generate(graph, covariance, xi, cuda=True)
xi_back = gp.generate_inv(graph, covariance, values, cuda=False)

plt.plot(xi - xi_back)
plt.show()

plt.plot(values - values_cuda)
plt.show()

#%%
from functools import partial
from graphgp import Graph, build_graph, generate, generate_inv
from jax import Array
from typing import Tuple

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
    xi_residual, inv_sqrt_T = jax.vjp(inv_sqrt, value_residual)
    xi_residual = inv_sqrt_T(xi_residual)[0]

    sqrt = partial(generate, joint_graph, covariance, cuda=cuda, fast_jit=fast_jit)
    z = jnp.concatenate([xi_residual, jnp.zeros(N)], axis=0)
    _, sqrt_T = jax.vjp(sqrt, jnp.zeros(M + N))
    correction = sqrt(sqrt_T(z)[0])
    joint_values = random_joint_values + correction

    # Check we got the conditioning values right and return the corrected values at the N points
    # assert jnp.allclose(joint_values[:M], conditioning_values)
    return joint_values


#%% Extremely small example for conditioning
n_points = 100
rng, key = jr.split(rng)
points = jr.normal(key, (n_points, 1))
# points = jnp.linspace(-4, 4, n_points)[:, None]
covariance = gp.extras.rbf_kernel(variance=1.0, scale=0.5, r_min=1e-4, r_max=10.0, n_bins=1000, jitter=1e-4)

# %%
# rng, k1, k2 = jr.split(rng, 3)
# cond_xi = jr.normal(k1, (n_points,))
# joint_xi = jr.normal(k2, (n_points,))

values = gp.generate(graph, covariance, xi)

cond_points = points[:50]
conditioning_graph = gp.build_graph(cond_points, n0=49, k=49)
joint_graph = gp.build_graph(points, n0=99, k=99)

cond_values = generate(conditioning_graph, covariance, cond_xi[:50])
joint_values = generate_conditional(
    conditioning_graph, joint_graph, covariance, cond_values, joint_xi
)

order = jnp.argsort(cond_points.squeeze())
plt.plot(cond_points[order], cond_values[order])
plt.scatter(points, joint_values, c='C1')
plt.title('dense')

#%%
plt.scatter(*points.T, c=joint_values, s=1)
plt.gca().set(aspect="equal", xlim=(-4, 4), ylim=(-4, 4))
plt.colorbar()
plt.show()

#%%
plt.scatter(*cond_points.T, c=cond_values, s=1)
plt.gca().set(aspect="equal", xlim=(-4, 4), ylim=(-4, 4))
plt.colorbar()
plt.show()
# %%
