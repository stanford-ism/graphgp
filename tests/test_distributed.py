import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.sharding import Mesh

import graphgp as gp

jax.config.update("jax_enable_x64", True)


def _setup(dtype=jnp.float64):
    n = 48
    points = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)[:, None]
    graph = gp.build_graph(points, n0=8, k=4)
    covariance = gp.extras.matern_kernel(
        p=0,
        variance=1.0,
        cutoff=0.3,
        r_min=1e-4,
        r_max=2.0,
        n_bins=64,
        jitter=1e-5,
    )
    if dtype != jnp.float64:
        graph = gp.Graph(
            graph.points.astype(dtype), graph.neighbors, graph.offsets, graph.indices
        )
        covariance = tuple(value.astype(dtype) for value in covariance)
    return graph, covariance


@pytest.mark.parametrize("n_partitions", [1, 2, 4])
def test_recompute_matches_generate(n_partitions):
    graph, covariance = _setup()
    owners = np.arange(len(graph.points)) % n_partitions
    plan = gp.distributed.partition_graph(graph, owners)
    xi = jr.normal(jr.key(1), (2, len(graph.points)), dtype=jnp.float64)

    expected = jax.vmap(lambda z: gp.generate(graph, covariance, z))(xi)
    actual = gp.distributed.generate_recompute(plan, covariance, xi)

    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_exchange_matches_generate_and_is_sharded():
    n_partitions = min(jax.device_count(), 4)
    graph, covariance = _setup()
    owners = np.arange(len(graph.points)) % n_partitions
    plan = gp.distributed.partition_graph(graph, owners)
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    xi = jr.normal(jr.key(2), (len(graph.points),), dtype=jnp.float64)

    expected = gp.generate(graph, covariance, xi)
    actual = gp.distributed.generate(plan, covariance, xi, mesh=mesh)

    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual.sharding.mesh.shape["space"] == n_partitions


def test_exchange_nonleading_output_axis_and_arbitrary_owners():
    graph, covariance = _setup()
    n_partitions = min(jax.device_count(), 4)
    owners = np.asarray(
        [((i * 7) + (i // 5)) % n_partitions for i in range(len(graph.points))]
    )
    plan = gp.distributed.partition_graph(graph, owners, output_shape=(6, 8))
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    xi = jr.normal(jr.key(22), (6, 8), dtype=jnp.float64)

    expected = gp.generate(graph, covariance, xi.reshape(-1)).reshape(6, 8)
    actual = gp.distributed.generate(plan, covariance, xi, mesh=mesh)
    oracle = gp.distributed.generate_recompute(plan, covariance, xi)

    assert plan.output_axis == 1
    assert actual.sharding.spec == jax.sharding.PartitionSpec(None, "space")
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(oracle, expected, rtol=1e-10, atol=1e-10)


def test_exchange_batched_adjoint_and_covariance_gradient():
    n_partitions = min(jax.device_count(), 4)
    graph, covariance = _setup()
    owners = np.repeat(np.arange(n_partitions), len(graph.points) // n_partitions)
    plan = gp.distributed.partition_graph(graph, owners)
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    cov_bins, cov_vals = covariance

    k1, k2, k3 = jr.split(jr.key(3), 3)
    xi = jr.normal(k1, (2, len(graph.points)), dtype=jnp.float64)
    xi_tangent = jr.normal(k2, xi.shape, dtype=xi.dtype)
    value_tangent = jr.normal(k3, xi.shape, dtype=xi.dtype)

    def generate(x, values=cov_vals):
        return gp.distributed.generate(plan, (cov_bins, values), x, mesh=mesh)

    _, jvp = jax.jvp(generate, (xi,), (xi_tangent,))
    _, pullback = jax.vjp(generate, xi)
    vjp = pullback(value_tangent)[0]
    lhs = jnp.vdot(value_tangent, jvp)
    rhs = jnp.vdot(vjp, xi_tangent)
    assert jnp.allclose(lhs, rhs, rtol=1e-8, atol=1e-8)

    cov_grad = jax.grad(lambda values: jnp.sum(generate(xi, values)))(cov_vals)
    assert jnp.all(jnp.isfinite(cov_grad))


@pytest.mark.parametrize(
    ("dtype", "tolerance"), [(jnp.float32, 1e-4), (jnp.float64, 1e-8)]
)
def test_exchange_vmap_combined_with_jvp_and_vjp(dtype, tolerance):
    n_partitions = min(jax.device_count(), 4)
    graph, covariance = _setup(dtype)
    plan = gp.distributed.partition_graph(
        graph, np.arange(len(graph.points)) % n_partitions
    )
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    key_xi, key_tangent, key_cotangent = jr.split(jr.key(31), 3)
    xi = jr.normal(key_xi, (2, len(graph.points)), dtype=dtype)
    tangent = jr.normal(key_tangent, xi.shape, dtype=dtype)
    cotangent = jr.normal(key_cotangent, xi.shape, dtype=dtype)

    mapped = jax.vmap(
        lambda z: gp.distributed.generate(plan, covariance, z, mesh=mesh)
    )
    _, jvp = jax.jvp(mapped, (xi,), (tangent,))
    _, pullback = jax.vjp(mapped, xi)
    vjp = pullback(cotangent)[0]

    assert jnp.allclose(
        jnp.vdot(cotangent, jvp),
        jnp.vdot(vjp, tangent),
        rtol=tolerance,
        atol=tolerance,
    )


def test_lowered_exchange_does_not_all_gather_the_field():
    graph, covariance = _setup()
    n_partitions = min(jax.device_count(), 4)
    plan = gp.distributed.partition_graph(
        graph, np.repeat(np.arange(n_partitions), len(graph.points) // n_partitions)
    )
    mesh = Mesh(np.asarray(jax.devices()[:n_partitions]), ("space",))
    xi = jnp.zeros((len(graph.points),), dtype=jnp.float64)
    lowered = jax.jit(
        lambda z: gp.distributed.generate(plan, covariance, z, mesh=mesh)
    ).lower(xi)
    hlo = lowered.as_text().lower()

    assert "all_gather" in hlo or "all-gather" in hlo
    assert "all_to_all" in hlo or "all-to-all" in hlo
    for line in hlo.splitlines():
        if "all_gather" in line or "all-gather" in line:
            assert f"x{len(graph.points)}" not in line


def test_partition_stats_and_validation():
    graph, _ = _setup()
    owners = np.arange(len(graph.points)) % 2
    plan = gp.distributed.partition_graph(graph, owners, output_shape=(6, 8))
    stats = gp.distributed.partition_stats(plan)

    assert stats.owned_nodes == (24, 24)
    assert stats.cross_partition_parents > 0
    assert stats.ancestor_replication_factor >= 1.0
    with pytest.raises(ValueError, match="dense range"):
        gp.distributed.partition_graph(graph, owners + 1)
    with pytest.raises(ValueError, match="product"):
        gp.distributed.partition_graph(graph, owners, output_shape=(7, 7))
