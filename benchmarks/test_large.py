import jax
import jax.numpy as jnp
import jax.random as jr

import time

import graphgp as gp
import graphgp_cuda as gp_cuda

rng = jr.key(1234)


n_points = 300_000_000
print(f"Generating {n_points} points...", flush=True)
rng, k1, k2, k3 = jr.split(rng, 4)
points = jr.normal(k1, (n_points, 3))
xi = jr.normal(k2, (n_points,))
fake_neighbors = jr.permutation(k3, xi.shape[0])
print()


print('Timing FFT...', flush=True)
for i in range(5):
    start = time.perf_counter()
    ffted = jnp.fft.fft(xi)
    ffted.block_until_ready()
    end = time.perf_counter()
    print(f"{1000*(end - start):.1f} ms", flush=True)
print()


print('Timing fake refine...', flush=True)
for i in range(5):
    start = time.perf_counter()
    values = gp_cuda.fake_refine(points, fake_neighbors, xi)
    values.block_until_ready()
    end = time.perf_counter()
    print(f"{1000*(end - start):.1f} ms", flush=True)
print()


print("Building graph...", flush=True)
start = time.perf_counter()
graph = gp.build_graph(points, n0=1000, k=1, cuda=True)
graph.points.block_until_ready()
end = time.perf_counter()
print(f"{1000*(end - start):.1f} ms", flush=True)
print()

print("Forward pass...", flush=True)
covariance = gp.prepare_matern_covariance_discrete(p=0, r_min=1e-3, r_max=10, n_bins=1000)
graph.indices = None
xi = jr.normal(rng, (n_points,))
for i in range(3):
    start = time.perf_counter()
    values = gp.generate(graph, covariance, xi, cuda=True)
    values.block_until_ready()
    end = time.perf_counter()
    print(f"{1000*(end - start):.1f} ms", flush=True)
print()


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