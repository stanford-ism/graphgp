import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

import graphgp as gp

rng = jr.key(137)

def check_equal(a, b, *, text, rtol=None):
    if rtol is None:
        assert jnp.all(a == b), text
    else:
        assert jnp.allclose(a, b, rtol=rtol), text

def test_build_tree_simple():
    points = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])[:,None]
    tree = gp.build_tree(points)
    check_equal(tree[0], jnp.array([6,3,9,2,8,5,10,1,7,4])[:,None], text="Points do not match")
    check_equal(tree[1], jnp.array([0,0,0,0,0,0,0,0,0,0]), text="Split dims do not match")
    check_equal(tree[2], jnp.array([5,2,8,1,7,4,9,0,6,3]), text="Indices do not match")

def test_build_tree_random():
    points = jr.normal(rng, (100000, 3))
    check_equal(points[0][0], -0.61777326, rtol=1e-8, text="RNG changed, cannot run test")
    tree = gp.build_tree(points)
    check_equal(tree[1][1000:1005], jnp.array([2, 0, 0, 1, 1]), text="Split dims do not match reference")
    check_equal(tree[2][-1], 59735, text="Indices do not match reference")

def test_build_tree_shapes():
    for n_dim in [2, 6]:
        points = jr.normal(rng, (1000, n_dim))
        tree = gp.build_tree(points)
        assert tree[0].shape == (1000, n_dim), "Points shape incorrect"
        assert tree[1].shape == (1000,), "Split dims shape incorrect"
        assert tree[2].shape == (1000,), "Indices shape incorrect"

def test_query_preceding():
    n_points = 1000
    n_dim = 3
    k = 30
    n0 = 30

    original_points = jr.normal(rng, (n_points, n_dim))
    points, split_dims, indices = gp.build_tree(original_points)
    neighbors = gp.query_preceding_neighbors(points, split_dims, n0=n0, k=k)

    pairwise_distance = jnp.linalg.norm(points[:,None,:] - points[None,:, :], axis=-1)
    i, j = jnp.indices(pairwise_distance.shape)
    pairwise_distance = pairwise_distance.at[i <= j].set(jnp.inf)
    true_neighbors = jnp.argsort(pairwise_distance, axis=-1)[n0:,:k]

    check_equal(neighbors, true_neighbors, text="Incorrect preceding neighbors")