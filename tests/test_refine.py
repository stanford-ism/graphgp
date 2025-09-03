import jax
import jax.numpy as jnp
import jax.random as jr

import hugegp as gp


def test_inverse():
    n_points = 1000
    n_dim = 3
    n0 = 100
    k = 10

    k1, k2 = jr.split(jr.key(2))
    points = jr.normal(k1, (n_points, n_dim))
    xi = jr.normal(k2, (n_points,))
    graph = gp.build_graph(points, n0=n0, k=k)
    covariance = gp.MaternCovariance(p=0)
    values = gp.generate_jit(graph, covariance, xi)
    xi_back = gp.generate_inv_jit(graph, covariance, values)
    values_back = gp.generate_jit(graph, covariance, xi_back)

    assert jnp.allclose(values, values_back, atol=1e-6), (
        "Values from xi and from inverted xi do not match within atol=1e-6."
    )