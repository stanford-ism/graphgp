import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial

import graphgp as gp

rng = jr.key(99)

# We basically just want to check CUDA against the JAX implementation.
# Testing of the JAX implementation will be done separately.

# Typically expect to match only to the 1e-4 level, while CUDA should be self-consistent to the 1e-6 level.


# def default_setup(rng_key):
#     n_points = 1000
#     points = jr.normal(rng_key, (n_points, 2))
#     graph = gp.build_graph(points, n0=100, k=8, cuda=True)
#     covariance = (gp.make_cov_bins(r_min=1e-4, r_max=10, n_bins=1000), gp.MaternCovariance(p=0))
#     return n_points, graph, covariance


# def test_forward():
#     k1, k2 = jr.split(rng, 2)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))

#     jax_values = gp.generate_jit(graph, covariance, xi, cuda=False)
#     cuda_values = gp.generate_jit(graph, covariance, xi, cuda=True)
#     assert jnp.allclose(jax_values, cuda_values, atol=1e-4), "JAX and CUDA values do not match within tolerance"


# def test_jvp_linear():
#     k1, k2, k3 = jr.split(rng, 3)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))
#     xi_tangent = jr.normal(k3, (n_points,))
#     jax_func = Partial(gp.generate_jit, graph, covariance, cuda=False)
#     cuda_func = Partial(gp.generate_jit, graph, covariance, cuda=True)

#     jax_values, jax_tangent = jax.jvp(jax_func, (xi,), (xi_tangent,))
#     cuda_values, cuda_tangent = jax.jvp(cuda_func, (xi,), (xi_tangent,))
#     assert jnp.allclose(jax_values, cuda_values, atol=1e-4), "JAX and CUDA values do not match within tolerance"
#     assert jnp.allclose(jax_tangent, cuda_tangent, atol=1e-4), "JAX and CUDA tangents do not match within tolerance"


# def test_vjp_linear():
#     k1, k2, k3 = jr.split(rng, 3)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))
#     values_tangent = jr.normal(k3, (n_points,))
#     jax_func = Partial(gp.generate_jit, graph, covariance, cuda=False)
#     cuda_func = Partial(gp.generate_jit, graph, covariance, cuda=True)

#     jax_values, jax_vjp = jax.vjp(jax_func, xi)
#     cuda_values, cuda_vjp = jax.vjp(cuda_func, xi)
#     jax_tangent = jax_vjp(values_tangent)[0]
#     cuda_tangent = cuda_vjp(values_tangent)[0]
#     assert jnp.allclose(jax_values, cuda_values, atol=1e-4), "JAX and CUDA values do not match within tolerance"
#     assert jnp.allclose(jax_tangent, cuda_tangent, atol=1e-4), "JAX and CUDA tangents do not match within tolerance"


# def test_adjoint_condition():
#     k1, k2, k3, k4 = jr.split(rng, 4)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))
#     xi_tangent = jr.normal(k3, (n_points,))
#     values_tangent = jr.normal(k4, (n_points,))
#     func = Partial(gp.generate, graph, covariance, cuda=True)

#     val1 = jnp.dot(values_tangent, jax.jvp(func, (xi,), (xi_tangent,))[1])
#     val2 = jnp.dot(xi_tangent, jax.vjp(func, xi)[1](values_tangent)[0])
#     assert jnp.isclose(val1, val2, rtol=1e-6), f"Adjoint test failed: {val1:.5e} != {val2:.5e} within rtol=1e-6"


# def test_jit_success():
#     k1, k2, k3, k4 = jr.split(rng, 4)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (3, n_points))  # Batch size of 3
#     xi_tangent = jr.normal(k3, (3, n_points))
#     values_tangent = jr.normal(k4, (3, n_points))

#     cuda_func = jax.vmap(Partial(gp.generate, graph, covariance, cuda=True))
#     _ = jax.jit(cuda_func)(xi).block_until_ready()
#     _ = jax.jit(Partial(jax.jvp, cuda_func))((xi,), (xi_tangent,))[1].block_until_ready()
#     _ = jax.jit(jax.vjp(cuda_func, xi)[1])(values_tangent)[0].block_until_ready()

# def test_forward_vmap():
#     k1, k2 = jr.split(rng, 2)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (3, n_points))  # Batch size of 3
#     jax_func = jax.jit(jax.vmap(Partial(gp.generate, graph, covariance, cuda=False)))
#     cuda_func = jax.jit(jax.vmap(Partial(gp.generate, graph, covariance, cuda=True)))

#     jax_values = jax_func(xi)
#     cuda_values = cuda_func(xi)
#     assert jnp.allclose(jax_values, cuda_values, atol=1e-4), "JAX and CUDA values do not match within tolerance"


# def test_triple_vmap():
#     k1, k2 = jr.split(rng, 2)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (1, 2, 3, n_points))
#     jax_func = jax.jit(jax.vmap(
#         jax.vmap(jax.vmap(Partial(gp.generate, graph, covariance, cuda=False)))
#     ))
#     cuda_func = jax.jit(jax.vmap(
#         jax.vmap(jax.vmap(Partial(gp.generate, graph, covariance, cuda=True)))
#     ))

#     jax_values = jax_func(xi)
#     cuda_values = cuda_func(xi)
#     assert jnp.allclose(jax_values, cuda_values, atol=1e-4), "JAX and CUDA values do not match within tolerance"


# def test_jvp_linear_batched(rtol=1e-2, frac_outliers_allowed=0.01):
#     k1, k2, k3 = jr.split(key, 3)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (3, n_points))  # Batch size of 3
#     xi_tangent = jr.normal(k3, (3, n_points))
#     jax_func = jax.vmap(Partial(hg.generate, graph, covariance, cuda=False))
#     cuda_func = jax.vmap(Partial(hg.generate, graph, covariance, cuda=True))

#     jax_values, jax_tangent = jax.jvp(jax_func, (xi,), (xi_tangent,))
#     cuda_values, cuda_tangent = jax.jvp(cuda_func, (xi,), (xi_tangent,))
#     outlier_check(
#         "JVP linear batched primals", jax_values, cuda_values, rtol, frac_outliers_allowed
#     )
#     outlier_check(
#         "JVP linear batched tangents", jax_tangent, cuda_tangent, rtol, frac_outliers_allowed
#     )


# def test_vjp_linear_batched(rtol=1e-2, frac_outliers_allowed=0.01):
#     k1, k2, k3 = jr.split(key, 3)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (3, n_points))  # Batch size of 3
#     values_tangent = jr.normal(k3, (3, n_points))
#     jax_func = jax.vmap(Partial(hg.generate, graph, covariance, cuda=False))
#     cuda_func = jax.vmap(Partial(hg.generate, graph, covariance, cuda=True))

#     jax_values, jax_vjp = jax.vjp(jax_func, xi)
#     cuda_values, cuda_vjp = jax.vjp(cuda_func, xi)
#     jax_tangent = jax_vjp(values_tangent)[0]
#     cuda_tangent = cuda_vjp(values_tangent)[0]
#     outlier_check(
#         "VJP linear batched primals", jax_values, cuda_values, rtol, frac_outliers_allowed
#     )
#     outlier_check(
#         "VJP linear batched tangents", jax_tangent, cuda_tangent, rtol, frac_outliers_allowed
#     )


# def test_loss_gradient(rtol=1e-2, frac_outliers_allowed=0.001):
#     k1, k2 = jr.split(key, 2)
#     n_points, graph, covariance = default_setup(k1)

#     def loss_func(xi, cuda=False):
#         return jnp.sum(jnp.square(hg.generate(graph, covariance, xi, cuda=cuda)))

#     xi = jr.normal(k2, (n_points,))
#     jax_grad = jax.grad(loss_func, argnums=0)(xi, cuda=False)
#     cuda_grad = jax.grad(loss_func, argnums=0)(xi, cuda=True)
#     outlier_check("Loss gradient", jax_grad, cuda_grad, rtol, frac_outliers_allowed)


# def test_fisher_metric(rtol=1e-2, frac_outliers_allowed=0.01):
#     k1, k2, k3 = jr.split(key, 3)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))
#     xi_tangent = jr.normal(k3, (n_points,))
#     jax_func = Partial(hg.generate, graph, covariance, cuda=False)
#     cuda_func = Partial(hg.generate, graph, covariance, cuda=True)

#     jax_mvp = jax.vjp(jax_func, xi)[1](jax.jvp(jax_func, (xi,), (xi_tangent,))[1])[0]
#     cuda_mvp = jax.vjp(cuda_func, xi)[1](jax.jvp(cuda_func, (xi,), (xi_tangent,))[1])[0]
#     outlier_check("Fisher metric", jax_mvp, cuda_mvp, rtol, frac_outliers_allowed)


# def test_hessian():
#     k1, k2 = jr.split(key, 2)
#     n_points = 100
#     points = jr.normal(k1, (n_points, 2))
#     graph, indices = hg.build_strict_graph(points, n_initial=7, k=4)
#     covariance = hg.test_cov_discretized(0.001, 20, 1000, cutoff=0.2, slope=-1.0, scale=1.0)
#     xi = jr.normal(k2, (n_points,))

#     def loss_func(xi, cuda=False):
#         return jnp.sum(jnp.square(hg.generate(graph, covariance, xi, cuda=cuda)))

#     jax_hess = jax.hessian(Partial(loss_func, cuda=False))(xi)
#     cuda_hess = jax.hessian(Partial(loss_func, cuda=True))(xi)
#     outlier_check("Hessian", jax_hess, cuda_hess, rtol=1e-2, frac_outliers_allowed=0.001)
