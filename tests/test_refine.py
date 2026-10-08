import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.tree_util import Partial
from test_tree import check_equal

import graphgp as gp

rng = jr.key(137)


@pytest.fixture
def setup_graph():
    n_points = 1000
    n_dim = 3
    n0 = 100
    k = 10

    points = jr.normal(rng, (n_points, n_dim))
    graph = gp.build_graph(points, n0=n0, k=k)
    covariance = gp.extras.matern_kernel(p=0, variance=1.0, cutoff=1.0, r_min=1e-4, r_max=10, n_bins=1000, jitter=1e-5)

    yield graph, covariance, points


def test_logdet_random(setup_graph):
    graph, covariance, _points = setup_graph
    check_equal(graph.points[0, 0], -1.95624711, rtol=1e-8, text="RNG or setup changed, cannot run test")
    check_equal(
        jax.jit(gp.generate_logdet)(graph, covariance),
        -600.36165088,
        rtol=1e-8,
        text="Logdet does not match reference within rtol=1e-8.",
    )
    check_equal(
        jax.jit(gp.generate_dense_logdet)(graph.points, covariance),
        -610.90538067,
        rtol=1e-8,
        text="Dense logdet does not match reference within rtol=1e-8.",
    )


def test_inverse(setup_graph):
    graph, covariance, _points = setup_graph
    xi = jr.normal(rng, (graph.points.shape[0],))
    values = jax.jit(gp.generate)(graph, covariance, xi)
    xi_back = jax.jit(gp.generate_inv)(graph, covariance, values)
    values_back = jax.jit(gp.generate)(graph, covariance, xi_back)
    check_equal(
        values, values_back, rtol=1e-12, text="Values from xi and from inverted xi do not match within rtol=1e-12."
    )


def test_fast_jit(setup_graph):
    graph, covariance, _points = setup_graph
    xi = jr.normal(rng, (graph.points.shape[0],))

    v1 = jax.jit(gp.generate)(graph, covariance, xi)
    v2 = jax.jit(Partial(gp.generate, fast_jit=False))(graph, covariance, xi)
    check_equal(v1, v2, rtol=1e-12, text="Fast JIT does not match simple implementation.")


def test_approaches_dense():
    points = jr.normal(rng, (1000, 3))
    graph = gp.build_graph(points, n0=200, k=200)
    graph = gp.Graph(graph.points, graph.neighbors, graph.offsets)
    covariance = gp.extras.matern_kernel(p=0, variance=1.0, cutoff=1.0, r_min=1e-4, r_max=10, n_bins=1000, jitter=1e-5)

    xi = jr.normal(rng, (graph.points.shape[0],))
    true_values = jax.jit(gp.generate_dense)(graph.points, covariance, xi)
    values = jax.jit(gp.generate)(graph, covariance, xi)
    assert jnp.allclose(true_values, values, atol=0.02), "Values do not match dense within atol=0.02."

    J = jax.jacfwd(Partial(gp.generate, graph, covariance))(jnp.zeros(graph.points.shape[0]))
    K = J @ J.T
    dense_K = gp.compute_cov_matrix(covariance, graph.points, graph.points)
    assert jnp.allclose(K, dense_K, atol=0.02), "Covariance does not match dense within atol=0.02."


def test_callable_covariance(setup_graph):
    graph, covariance, _points = setup_graph

    # Same Matern-1/2 kernel as the fixture, as a function instead of a lookup table
    def matern12(x1, x2):
        r = jnp.linalg.norm(x1 - x2)
        return jnp.exp(-r) * jnp.where(r == 0.0, 1.0 + 1e-5, 1.0)

    xi = jr.normal(rng, (graph.points.shape[0],))
    v_table = jax.jit(gp.generate)(graph, covariance, xi)
    v_func = jax.jit(lambda g, xi: gp.generate(g, matern12, xi))(graph, xi)
    assert jnp.allclose(v_table, v_func, atol=1e-3), "Callable covariance does not match discretized within atol=1e-3."

    logdet_table = jax.jit(gp.generate_logdet)(graph, covariance)
    logdet_func = jax.jit(lambda g: gp.generate_logdet(g, matern12))(graph)
    check_equal(logdet_table, logdet_func, rtol=1e-3, text="Callable covariance logdet does not match discretized.")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_no_jitter_duplicates(dtype):
    # smooth kernel, no jitter, and many duplicated points: stock Cholesky gives NaN here
    with jax.enable_x64(dtype == jnp.float64):
        points = jr.normal(rng, (500, 2))
        points = jnp.concatenate([points, points[:100], points[:20] + 1e-6]).astype(dtype)
        graph = gp.build_graph(points, n0=50, k=8)
        covariance = gp.extras.rbf_kernel(variance=1.0, scale=0.2, r_min=1e-5, r_max=10, n_bins=1000)
        covariance = (covariance[0].astype(dtype), covariance[1].astype(dtype))
        xi = jr.normal(rng, (len(points),)).astype(dtype)

        values = jax.jit(gp.generate)(graph, covariance, xi)
        assert jnp.all(jnp.isfinite(values))
        # a smooth kernel is ill-conditioned, so the two orders of operations differ by more than rounding
        v_slow = jax.jit(Partial(gp.generate, fast_jit=False))(graph, covariance, xi)
        assert jnp.allclose(values, v_slow, atol=2e-3 if dtype == jnp.float32 else 1e-6)

        # duplicated points get exactly the value of their twin
        assert jnp.all(values[500:600] == values[:100])

        # inverse reproduces the values (dropped targets get xi = 0)
        xi_back = jax.jit(gp.generate_inv)(graph, covariance, values)
        assert jnp.all(jnp.isfinite(xi_back))
        values_back = jax.jit(gp.generate)(graph, covariance, xi_back)
        assert jnp.allclose(values, values_back, atol=1e-3 if dtype == jnp.float32 else 1e-8)

        assert jax.jit(gp.generate_logdet)(graph, covariance) == -jnp.inf  # degenerate density
        assert jnp.all(jnp.isnan(jax.jit(Partial(gp.generate, clamp=False))(graph, covariance, xi)))
        grad = jax.jit(jax.grad(lambda c: jnp.sum(gp.generate(graph, (covariance[0], c), xi) ** 2)))(covariance[1])
        assert jnp.all(jnp.isfinite(grad))


def test_clamped_matches_unclamped(setup_graph):
    graph, covariance, _points = setup_graph
    xi = jr.normal(rng, (graph.points.shape[0],))
    for fast_jit in [True, False]:
        v1 = jax.jit(Partial(gp.generate, fast_jit=fast_jit, clamp=True))(graph, covariance, xi)
        v2 = jax.jit(Partial(gp.generate, fast_jit=fast_jit, clamp=False))(graph, covariance, xi)
        check_equal(v1, v2, rtol=1e-12, text="Clamped and unclamped generate do not match.")
    x1 = jax.jit(Partial(gp.generate_inv, clamp=True))(graph, covariance, v1)
    x2 = jax.jit(Partial(gp.generate_inv, clamp=False))(graph, covariance, v1)
    check_equal(x1, x2, rtol=1e-10, text="Clamped and unclamped inverse do not match.")
    l1 = jax.jit(Partial(gp.generate_logdet, clamp=True))(graph, covariance)
    l2 = jax.jit(Partial(gp.generate_logdet, clamp=False))(graph, covariance)
    check_equal(l1, l2, rtol=1e-10, text="Clamped and unclamped logdet do not match.")
    g1 = jax.grad(lambda c: jnp.sum(gp.generate(graph, (covariance[0], c), xi, clamp=True) ** 2))(covariance[1])
    g2 = jax.grad(lambda c: jnp.sum(gp.generate(graph, (covariance[0], c), xi, clamp=False) ** 2))(covariance[1])
    check_equal(g1, g2, rtol=1e-8, text="Clamped and unclamped gradients do not match.")
