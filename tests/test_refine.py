import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial

import graphgp as gp

import pytest

from test_tree import check_equal

rng = jr.key(137)


@pytest.fixture
def setup_graph():
    n_points = 1000
    n_dim = 3
    n0 = 100
    k = 10

    points = jr.normal(rng, (n_points, n_dim))
    graph = gp.build_graph(points, n0=n0, k=k)
    covariance = gp.extras.matern_kernel(p=0, r_min=1e-4, r_max=10, n_bins=1000)

    yield graph, covariance, points


def test_logdet_random(setup_graph):
    graph, covariance, points = setup_graph
    check_equal(graph.points[0, 0], -1.95624711, rtol=1e-8, text="RNG or setup changed, cannot run test")
    check_equal(
        jax.jit(gp.generate_logdet)(graph, covariance),
        -600.36165088,
        rtol=1e-8,
        text="Logdet does not match reference within rtol=1e-12.",
    )
    check_equal(
        jax.jit(gp.generate_dense_logdet)(graph.points, covariance),
        -610.90538067,
        rtol=1e-8,
        text="Dense logdet does not match reference within rtol=1e-12.",
    )


def test_inverse(setup_graph):
    graph, covariance, points = setup_graph
    xi = jr.normal(rng, (graph.points.shape[0],))
    values = jax.jit(gp.generate)(graph, covariance, xi)
    xi_back = jax.jit(gp.generate_inv)(graph, covariance, values)
    values_back = jax.jit(gp.generate)(graph, covariance, xi_back)
    check_equal(
        values, values_back, rtol=1e-12, text="Values from xi and from inverted xi do not match within rtol=1e-12."
    )


def test_fast_jit(setup_graph):
    graph, covariance, points = setup_graph
    xi = jr.normal(rng, (graph.points.shape[0],))

    v1 = jax.jit(gp.generate)(graph, covariance, xi)
    v2 = jax.jit(Partial(gp.generate, fast_jit=False))(graph, covariance, xi)
    check_equal(v1, v2, rtol=1e-12, text="Fast JIT does not match simple implementation.")


def test_approaches_dense():
    points = jr.normal(rng, (1000, 3))
    graph = gp.build_graph(points, n0=200, k=200)
    graph = gp.Graph(graph.points, graph.neighbors, graph.offsets)
    covariance = gp.extras.matern_kernel(p=0, r_min=1e-4, r_max=10, n_bins=1000)

    xi = jr.normal(rng, (graph.points.shape[0],))
    true_values = jax.jit(gp.generate_dense)(graph.points, covariance, xi)
    values = jax.jit(gp.generate)(graph, covariance, xi)
    assert jnp.allclose(true_values, values, atol=0.02), "Values do not match dense within atol=0.02."

    J = jax.jacfwd(Partial(gp.generate, graph, covariance))(jnp.zeros(graph.points.shape[0]))
    K = J @ J.T
    dense_K = gp.compute_cov_matrix(covariance, graph.points, graph.points)
    assert jnp.allclose(K, dense_K, atol=0.02), "Covariance does not match dense within atol=0.02."
