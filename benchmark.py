#!/usr/bin/env python3
"""
Benchmark script for hugegp algorithm performance testing.

Sample configuration file format:
{
  "defaults": {
    "covariance": {"matern_p": 0, "discrete_cov": true, "r_min": 1e-5, "r_max": 10.0, "n_bins": 1000},
    "distribution": {"type": "gaussian"},
    "graph": {"strict": true},
    "timing_runs": 5,
    "cuda": false,
    "seed": 137
  },
  "runs": [
    {"n": 10000, "n0": 500, "d": 2, "k": 5, "cuda": false, "discrete_cov": false},
    {"n": 10000, "n0": 500, "d": 2, "k": 5, "cuda": true},
    {"n": 100000, "n0": 1000, "d": 3, "k": 10, "cuda": false, "distribution": {"type": "radial", "r_min": 0.1, "r_max": 10.0}},
    {"n": 100000, "n0": 1000, "d": 3, "k": 10, "cuda": true,  "distribution": {"type": "uniform"}},
    {"n": 50000, "n0": 1000, "d": 2, "k": 8, "graph": {"strict": false, "factor": 1.3}}
  ]
}

Distribution types:
- "gaussian": Standard normal distribution (default)
- "uniform": Uniform distribution from -1 to 1
- "radial": Points at random directions and log-uniform distances from origin
            Requires "r_min" and "r_max" parameters

Graph types:
- "strict": true (default) - Use build_graph for highest accuracy but slower to build
- "strict": false - Use build_lazy_graph for faster building but less accuracy
            Requires "factor" parameter (multiplicative factor for batch growth)
- "serial_depth": true/false - Use compute_depths_serial if true, compute_depths if false (default)
"""

import json
import time
import argparse
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import hugegp as gp
from hugegp.tree import build_tree, query_preceding_neighbors, query_offset_neighbors
from hugegp.graph import compute_depths_parallel, compute_depths_serial, order_by_depth


def generate_points(rng, n_points, n_dim, distribution_params):
    """Generate points according to the specified distribution."""
    dist_type = distribution_params["type"]

    if dist_type == "gaussian":
        return jr.normal(rng, (n_points, n_dim))
    elif dist_type == "uniform":
        return jr.uniform(rng, (n_points, n_dim), minval=-1.0, maxval=1.0)
    elif dist_type == "radial":
        # Random directions on unit sphere
        directions = jr.normal(rng, (n_points, n_dim))
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

        # Random radii in log space
        r_min = distribution_params["r_min"]
        r_max = distribution_params["r_max"]
        log_r = jr.uniform(rng, (n_points, 1), minval=jnp.log(r_min), maxval=jnp.log(r_max))
        radii = jnp.exp(log_r)

        return directions * radii
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def run_single_benchmark(test_params):
    """Run benchmark for a single parameter combination."""
    # Build covariance
    cov_func = gp.MaternCovariance(p=test_params["covariance"]["matern_p"])
    if test_params["covariance"]["discrete_cov"]:
        cov_bins = gp.make_cov_bins(
            r_min=test_params["covariance"]["r_min"],
            r_max=test_params["covariance"]["r_max"],
            n_bins=test_params["covariance"]["n_bins"],
        )
        cov_vals = cov_func(cov_bins)
        covariance = (cov_bins, cov_vals)
    else:
        covariance = cov_func

    # Use test-specific random seed
    test_rng = jr.key(test_params["seed"])
    test_rng, k1, k2 = jr.split(test_rng, 3)

    # Generate points according to distribution
    points = generate_points(k1, test_params["n"], test_params["d"], test_params["distribution"])
    xi = jr.normal(k2, (test_params["n"],))

    # Build graph with timing
    graph_timings = {}

    if test_params["graph"]["strict"]:
        # # Time all-cuda build graph
        # start = time.perf_counter()
        # graph = gp.build_graph(points, n0=test_params["n0"], k=test_params["k"], cuda=True)
        # graph.points.block_until_ready()
        # graph_timings["build_graph"] = time.perf_counter() - start

        # Time build_tree
        for i in range(2):
            start = time.perf_counter()
            points_reordered, split_dims, indices = build_tree(points, cuda=test_params["cuda"])
            indices.block_until_ready()
            graph_timings["build_tree"] = time.perf_counter() - start

        # Time query_preceding_neighbors
        for i in range(2):
            start = time.perf_counter()
            neighbors = query_preceding_neighbors(
                points_reordered, split_dims, k=test_params["k"], n0=test_params["n0"], cuda=test_params["cuda"]
            )
            neighbors.block_until_ready()
            graph_timings["query_neighbors"] = time.perf_counter() - start

        # Time compute_depths
        for i in range(2):
            if test_params["graph"]["serial_depth"]:
                start = time.perf_counter()
                depths = compute_depths_serial(neighbors, n0=test_params["n0"], cuda=test_params["cuda"])
                depths.block_until_ready()
                graph_timings["compute_depths_serial"] = time.perf_counter() - start
            else:
                start = time.perf_counter()
                depths = compute_depths_parallel(neighbors, n0=test_params["n0"], cuda=test_params["cuda"])
                depths.block_until_ready()
                graph_timings["compute_depths_parallel"] = time.perf_counter() - start
        
        # Time order_by_depth
        for i in range(2):
            start = time.perf_counter()
            points_final, indices_final, neighbors_final, depths_final = order_by_depth(
                points_reordered, indices, neighbors, depths, cuda=test_params["cuda"]
            )
            offsets = jnp.searchsorted(depths_final, jnp.arange(1, jnp.max(depths_final) + 2))
            offsets = tuple(int(o) for o in offsets)
            depths_final.block_until_ready()
            graph_timings["order_by_depth"] = time.perf_counter() - start

        graph = gp.Graph(points_final, neighbors_final[:, ::-1], offsets, indices_final)
    else:
        # Time build_tree for lazy graph
        start = time.perf_counter()
        points_reordered, split_dims, indices = build_tree(points)
        graph_timings["build_tree"] = time.perf_counter() - start

        # Calculate offsets for lazy graph
        offsets = [test_params["n0"]]
        while offsets[-1] < len(points):
            next_offset = offsets[-1] * test_params["graph"]["factor"]
            offsets.append(int(min(next_offset, len(points))))
        offsets = tuple(offsets)

        # Time query_offset_neighbors
        start = time.perf_counter()
        neighbors = query_offset_neighbors(points_reordered, split_dims, offsets=offsets, k=test_params["k"], cuda=True)
        graph_timings["query_neighbors"] = time.perf_counter() - start

        graph = gp.Graph(points_reordered, neighbors[:, ::-1], offsets, indices)

    level_sizes = np.diff(graph.offsets).tolist()
    graph_depth = len(graph.offsets)

    # Warmup with timing (JIT compilation) - always just one run
    start = time.perf_counter()
    func = jax.jit(gp.generate, static_argnames=("cuda",))
    output = func(graph, covariance, xi, cuda=test_params["cuda"])
    output.block_until_ready()
    warmup_time = time.perf_counter() - start

    # Memory stats
    stats = func.lower(graph, covariance, xi, cuda=test_params["cuda"]).compile().memory_analysis()
    total_mem = stats.temp_size_in_bytes + stats.argument_size_in_bytes + stats.output_size_in_bytes

    # Time multiple runs
    times = []
    for _ in range(test_params["timing_runs"]):
        start = time.perf_counter()
        output = func(graph, covariance, xi, cuda=test_params["cuda"])
        output.block_until_ready()
        times.append(time.perf_counter() - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    # warmup_time = 0
    # total_mem = 0
    # mean_time = 0
    # std_time = 0
    # times = [0, 0]

    result = {
        "parameters": test_params.copy(),
        # "level_sizes": level_sizes,
        "graph_depth": graph_depth,
        "graph_timings": {k: float(v) for k, v in graph_timings.items()},
        "warmup_time": float(warmup_time),
        "compiled_memory_mb": float(total_mem / (1024**2)),
        "timing": {"mean": float(mean_time), "std": float(std_time), "times": [float(t) for t in times]},
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark hugegp algorithm performance")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--output", type=str, help="Output file for results (if not provided, only prints results)")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Merge defaults with each test combination
    run_configs = []
    for test_params in config["runs"]:
        # Start with defaults and override with test-specific params
        merged_params = config["defaults"].copy()
        merged_params.update(test_params)
        run_configs.append(merged_params)

    # Run benchmarks
    results = []
    print(f"Running {len(run_configs)} benchmark combinations...")
    print("-" * 50)

    for i, test_params in enumerate(run_configs):
        print(f"Benchmark {i + 1}/{len(run_configs)}")
        param_str = ", ".join([f"{k}={v}" for k, v in config["runs"][i].items()])
        print(f"{param_str}", flush=True)

        result = run_single_benchmark(test_params)
        results.append(result)

        # Print graph depth and all graph timings on first line
        print(", ".join([f"{k}: {1000 * v:.2f} ms" for k, v in result["graph_timings"].items()]))
        print(f"Graph depth: {result['graph_depth']}")

        # Print compiled memory usage on third line
        compiled_mem_mb = result["compiled_memory_mb"]
        print(f"Compiled memory: {compiled_mem_mb:.1f} MB")

        # Print JIT compilation time on second line
        warmup_ms = 1000 * result["warmup_time"]
        print(f"Warmup: {warmup_ms:.2f} ms")

        # Print forward pass time with std on fourth line
        mean_ms = 1000 * result["timing"]["mean"]
        std_ms = 1000 * result["timing"]["std"]
        print(f"Forward pass: {mean_ms:.2f} ± {std_ms:.2f} ms")

        # Save results after each run if output file is specified
        if args.output:
            output_data = {"timestamp": datetime.now().isoformat(), "config": config, "results": results}
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

        print()

    # Final summary
    if args.output:
        print(f"\nAll results saved to: {args.output}")
    else:
        print(f"\nCompleted {len(results)} benchmark combinations.")


if __name__ == "__main__":
    main()
