import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import graphgp as gp

from test_tree import check_equal

rng = jr.key(137)


def test_graph_random():
    n_points = 100000
    n_dim = 3
    n0 = 1000
    k = 10

    points = jr.normal(rng, (n_points, n_dim))
    graph = gp.build_graph(points, n0=n0, k=k)

    check_equal(points[0, 0], -0.61777326, rtol=1e-8, text="RNG changed, cannot run test")
    assert len(graph.offsets) == 61
    assert graph.offsets[1] == 1115
    assert graph.neighbors[0, 0] == 488
    check_equal(graph.points[0, 0], -1.38417124, rtol=1e-8, text="Point does not match reference")


def test_graph_shapes():
    points = jr.normal(rng, (5000, 6))
    graph = gp.build_graph(points, n0=500, k=15)
    assert graph.points.shape == (5000, 6), "Points shape incorrect"
    assert graph.neighbors.shape == (4500, 15), "Neighbors shape incorrect"
    assert graph.indices.shape == (5000,), "Indices shape incorrect"
    assert graph.offsets[0] == 500, "Offsets[0] incorrect"
    assert graph.offsets[-1] == 5000, "Offsets[-1] incorrect"


def test_compute_depths():
    neighbors = jnp.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [1, 3, 4],
            [2, 4, 5],
        ]
    )
    n0 = 3
    depths = gp.compute_depths(neighbors, n0=n0)
    expected_depths = jnp.array([0, 0, 0, 1, 1, 2, 3])
    check_equal(depths, expected_depths, text="Depths do not match expected values")


def test_order_by_depth():
    points = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])[:, None]
    indices = jnp.array([0, 1, 2, 3, 4, 5])
    neighbors = jnp.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 1],
        ]
    )
    depths = jnp.array([0, 0, 1, 2, 3, 1])
    new_points, new_indices, new_neighbors, new_depths = gp.order_by_depth(points, indices, neighbors, depths)
    check_equal(new_points, jnp.array([0.0, 1.0, 2.0, 5.0, 3.0, 4.0])[:, None], text="Points not ordered correctly")
    check_equal(new_indices, jnp.array([0, 1, 2, 5, 3, 4]), text="Indices not ordered correctly")
    check_equal(
        new_neighbors,
        jnp.array(
            [
                [0, 1],
                [0, 1],
                [1, 2],
                [2, 4],
            ]
        ),
        text="Neighbors not ordered correctly",
    )
    check_equal(new_depths, jnp.array([0, 0, 1, 1, 2, 3]), text="Depths not ordered correctly")
