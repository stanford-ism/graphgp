import jax
import jax.numpy as jnp
import jax.random as jr

import time

import graphgp as gp

rng = jr.key(1234)

n_points = 1_000_000_000
print(f"Generating {n_points} random points...", flush=True)
points = jr.normal(rng, (n_points, 3))

print("Building graph...", end=" ", flush=True)
start = time.perf_counter()
graph = gp.build_graph(points, n0=10000, k=5, cuda=True)
graph.points.block_until_ready()
end = time.perf_counter()
print(f"{1000*(end - start):.1f} ms", flush=True)

# print("Building tree...", end=" ", flush=True)
# start = time.perf_counter()
# points_reordered, split_dims, indices = gp.build_tree(points, cuda=True)
# indices.block_until_ready()
# end = time.perf_counter()
# print(f"{1000*(end - start):.1f} ms", flush=True)

# print("Querying neighbors...", end=" ", flush=True)
# start = time.perf_counter()
# neighbors = gp.query_preceding_neighbors(points_reordered, split_dims, n0=1000, k=4, cuda=True)
# neighbors.block_until_ready()
# end = time.perf_counter()
# print(f"{1000*(end - start):.1f} ms", flush=True)

# print("Computing depths...", end=" ", flush=True)
# start = time.perf_counter()
# depths = gp.compute_depths_parallel(neighbors, n0=1000, cuda=True)
# depths.block_until_ready()
# end = time.perf_counter()
# print(f"{1000*(end - start):.1f} ms", flush=True)

# print("Ordering by depth...", end=" ", flush=True)
# start = time.perf_counter()
# points_ordered, indices_ordered, neighbors_ordered, depths_ordered = gp.order_by_depth(
#     points_reordered, indices, neighbors, depths, cuda=True
# )
# points_ordered.block_until_ready()
# end = time.perf_counter()
# print(f"{1000*(end - start):.1f} ms", flush=True)

print("Graph construction done.", flush=True)
