# import jax
# import jax.numpy as jnp
# import jax.random as jr
# from jax.tree_util import Partial

# import hugegp as hg

# key = jr.key(99)

# # We basically just want to check CUDA against the JAX implementation.
# # Testing of the JAX implementation will be done separately.


# def default_setup(rng_key):
#     n_points = 10_000
#     n_initial = 100
#     k = 4
#     points = jr.normal(rng_key, (n_points, 2))
#     graph, indices = hg.build_strict_graph(points, n_initial=n_initial, k=k)
#     covariance = hg.test_cov_discretized(0.001, 20, 1000, cutoff=0.2, slope=-1.0, scale=1.0)
#     return n_points, graph, covariance


# def outlier_check(name, jax_values, cuda_values, rtol, frac_outliers_allowed):
#     is_outlier = ~jnp.isclose(cuda_values, jax_values, rtol=rtol)
#     frac_outliers = jnp.sum(is_outlier) / jax_values.size
#     assert frac_outliers <= frac_outliers_allowed, (
#         f"{name} test failed: {frac_outliers:.5%} outliers beyond rtol={rtol}"
#     )


# def test_jit_success():
#     k1, k2, k3, k4 = jr.split(key, 4)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (3, n_points))  # Batch size of 3
#     xi_tangent = jr.normal(k3, (3, n_points))
#     values_tangent = jr.normal(k4, (3, n_points))

#     cuda_func = jax.vmap(Partial(hg.generate, graph, covariance, cuda=True))
#     _ = jax.jit(cuda_func)(xi).block_until_ready()
#     _ = jax.jit(Partial(jax.jvp, cuda_func))((xi,), (xi_tangent,))[1].block_until_ready()
#     _ = jax.jit(jax.vjp(cuda_func, xi)[1])(values_tangent)[0].block_until_ready()


# def test_forward(rtol=1e-2, frac_outliers_allowed=0.001):
#     k1, k2 = jr.split(key, 2)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))

#     jax_values = hg.generate(graph, covariance, xi, cuda=False)
#     cuda_values = hg.generate(graph, covariance, xi, cuda=True)
#     outlier_check("Forward", jax_values, cuda_values, rtol, frac_outliers_allowed)


# def test_forward_batched(rtol=1e-2, frac_outliers_allowed=0.01):
#     k1, k2 = jr.split(key, 2)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (3, n_points))  # Batch size of 10
#     jax_func = jax.vmap(Partial(hg.generate, graph, covariance, cuda=False))
#     cuda_func = jax.vmap(Partial(hg.generate, graph, covariance, cuda=True))

#     jax_values = jax_func(xi)
#     cuda_values = cuda_func(xi)
#     outlier_check("Forward batched", jax_values, cuda_values, rtol, frac_outliers_allowed)


# def test_triple_vmap(rtol=1e-2, frac_outliers_allowed=0.01):
#     k1, k2 = jr.split(key, 2)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (1, 2, 3, n_points))
#     jax_func = jax.vmap(
#         jax.vmap(jax.vmap(Partial(hg.generate, graph, covariance, cuda=False)))
#     )
#     cuda_func = jax.vmap(
#         jax.vmap(jax.vmap(Partial(hg.generate, graph, covariance, cuda=True)))
#     )

#     jax_values = jax_func(xi)
#     cuda_values = cuda_func(xi)
#     outlier_check("Triple vmap", jax_values, cuda_values, rtol, frac_outliers_allowed)


# def test_jvp_linear(rtol=1e-2, frac_outliers_allowed=0.001):
#     k1, k2, k3 = jr.split(key, 3)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))
#     xi_tangent = jr.normal(k3, (n_points,))
#     jax_func = Partial(hg.generate, graph, covariance, cuda=False)
#     cuda_func = Partial(hg.generate, graph, covariance, cuda=True)

#     jax_values, jax_tangent = jax.jvp(jax_func, (xi,), (xi_tangent,))
#     cuda_values, cuda_tangent = jax.jvp(cuda_func, (xi,), (xi_tangent,))
#     outlier_check("JVP linear primals", jax_values, cuda_values, rtol, frac_outliers_allowed)
#     outlier_check("JVP linear tangents", jax_tangent, cuda_tangent, rtol, frac_outliers_allowed)


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


# def test_vjp_linear(rtol=1e-2, frac_outliers_allowed=0.01):
#     k1, k2, k3 = jr.split(key, 3)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))
#     values_tangent = jr.normal(k3, (n_points,))
#     jax_func = Partial(hg.generate, graph, covariance, cuda=False)
#     cuda_func = Partial(hg.generate, graph, covariance, cuda=True)

#     jax_values, jax_vjp = jax.vjp(jax_func, xi)
#     cuda_values, cuda_vjp = jax.vjp(cuda_func, xi)
#     jax_tangent = jax_vjp(values_tangent)[0]
#     cuda_tangent = cuda_vjp(values_tangent)[0]
#     outlier_check("VJP linear primals", jax_values, cuda_values, rtol, frac_outliers_allowed)
#     outlier_check("VJP linear tangents", jax_tangent, cuda_tangent, rtol, frac_outliers_allowed)


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


# def test_linear_adjoint(rtol=1e-6):
#     k1, k2, k3, k4 = jr.split(key, 4)
#     n_points, graph, covariance = default_setup(k1)
#     xi = jr.normal(k2, (n_points,))
#     xi_tangent = jr.normal(k3, (n_points,))
#     values_tangent = jr.normal(k4, (n_points,))
#     func = Partial(hg.generate, graph, covariance, cuda=True)

#     val1 = jnp.dot(values_tangent, jax.jvp(func, (xi,), (xi_tangent,))[1])
#     val2 = jnp.dot(xi_tangent, jax.vjp(func, xi)[1](values_tangent)[0])
#     assert jnp.isclose(val1, val2, rtol=rtol), (
#         f"Adjoint test failed: {val1:.5e} != {val2:.5e} within rtol={rtol}"
#     )
