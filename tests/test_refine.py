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


def test_fast_jit_vmap_vjp(setup_graph):
    """The vjp of the batched (vmap) fast_jit generation is the transpose of its jvp,
    with respect to the excitations and to the covariance values (regression: the
    clamped dynamic slices of the level scan gave a wrong vjp under vmap)."""
    graph, covariance, _points = setup_graph
    k1, k2, k3, k4 = jr.split(rng, 4)
    xi = jr.normal(k1, (3, graph.points.shape[0]))
    v = jr.normal(k2, xi.shape)
    w = jr.normal(k3, xi.shape)
    cov_bins, cov_vals = covariance

    def adjoint(f, x, tangent):
        _, jv = jax.jvp(f, (x,), (tangent,))
        _, vjp = jax.vjp(f, x)
        (jt,) = vjp(w)
        lhs, rhs = jnp.vdot(jv, w), jnp.vdot(tangent, jt)
        assert abs(lhs - rhs) <= 1e-4 * abs(lhs), f"<Jv,w> = {lhs}, <v,J^T w> = {rhs}"

    gen_xi = jax.vmap(lambda x: gp.generate(graph, covariance, x, fast_jit=True))
    adjoint(gen_xi, xi, v)
    check_equal(gen_xi(xi)[0], gp.generate(graph, covariance, xi[0], fast_jit=False), rtol=1e-5, text="vmap fast_jit vs simple")
    gen_cov = lambda c: jax.vmap(lambda x: gp.generate(graph, (cov_bins, c), x, fast_jit=True))(xi)  # noqa: E731
    adjoint(gen_cov, cov_vals, jr.normal(k4, cov_vals.shape) * 1e-2)


def _generate_by_steps(graph, covariance, xi_tree, pad=0):
    """Generation in the graph's order as one ``refine_step`` per level, optionally with every level
    padded by ``pad`` invalid targets (arbitrary neighbors, a point that coincides with them)."""
    n0, k = int(graph.offsets[0]), graph.neighbors.shape[1]
    values = jnp.zeros(graph.points.shape[0]).at[:n0].set(gp.generate_dense(graph.points[:n0], covariance, xi_tree[:n0]))
    for start, end in zip(graph.offsets[:-1], graph.offsets[1:]):
        start, end = int(start), int(end)
        targets, neighbors, xi = graph.points[start:end], graph.neighbors[start - n0 : end - n0], xi_tree[start:end]
        valid = None
        if pad:
            targets = jnp.concatenate([targets, jnp.zeros((pad, targets.shape[1]))])
            neighbors = jnp.concatenate([neighbors, jnp.full((pad, k), -1, neighbors.dtype)])
            xi = jnp.concatenate([xi, jnp.ones(pad)])
            valid = jnp.arange(end - start + pad) < end - start
        new = gp.refine_step(graph.points, values, targets, neighbors, xi, covariance, valid=valid)
        if pad:
            assert jnp.all(new[end - start :] == 0), "Padded targets must return 0."
        values = values.at[start:end].set(new[: end - start])
    return values


def test_refine_step_matches_generate(setup_graph):
    """One ``refine_step`` per level, with the values held in the graph's order, reproduces ``generate``."""
    graph, covariance, _points = setup_graph
    xi = jr.normal(rng, (graph.points.shape[0],))
    expected = gp.generate(graph, covariance, xi, fast_jit=False)[graph.indices]  # graph order
    values = _generate_by_steps(graph, covariance, xi[graph.indices])
    check_equal(values, expected, rtol=1e-10, text="refine_step level by level does not match generate.")


def test_refine_step_padding(setup_graph):
    """Padded targets do not change the values of the others nor the gradients, which stay finite although
    the padded entries are degenerate (identical points)."""
    graph, covariance, _points = setup_graph
    xi_tree = jr.normal(rng, (graph.points.shape[0],))[graph.indices]
    cov_bins, cov_vals = covariance

    def loss(xi_tree, cov_vals, pad):
        return jnp.sum(jnp.sin(_generate_by_steps(graph, (cov_bins, cov_vals), xi_tree, pad=pad)))

    v0, g0 = jax.value_and_grad(loss, argnums=(0, 1))(xi_tree, cov_vals, 0)
    v1, g1 = jax.value_and_grad(loss, argnums=(0, 1))(xi_tree, cov_vals, 7)
    check_equal(v1, v0, rtol=1e-10, text="Padding changes the generated values.")
    for a, b in zip(g1, g0):
        assert jnp.all(jnp.isfinite(a)), "Gradient with padded targets is not finite."
        assert jnp.allclose(a, b, rtol=1e-8, atol=1e-10), "Padding changes the gradient."
