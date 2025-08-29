import jax
import jax.numpy as jnp
import jax.random as jr

import hugegp as gp

def test_query_preceding():
    n_points = 1000
    k = 30
    n0 = 30
    k1 = jr.key(137)
    original_points = jr.normal(k1, (n_points, 3))

    points, split_dims, indices = gp.build_tree(original_points)
    neighbors, distances = gp.query_preceding_neighbors(points, split_dims, n0=n0, k=k)

    pairwise_distance = jnp.linalg.norm(points[:,None,:] - points[None,:, :], axis=-1)
    i, j = jnp.indices(pairwise_distance.shape)
    pairwise_distance = pairwise_distance.at[i <= j].set(jnp.inf)
    true_neighbors = jnp.argsort(pairwise_distance, axis=-1)[n0:,:k]
    true_distances = jnp.linalg.norm(points[n0:,None,:] - points[true_neighbors], axis=-1)

    assert jnp.all(true_neighbors == neighbors)
    assert jnp.allclose(true_distances, distances)
