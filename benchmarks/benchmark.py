#!/usr/bin/env python3
"""
Benchmark script for graphgp algorithm performance testing.

Sample configuration file format:
{
  "defaults": {
    "covariance": {"matern_p": 0, "r_min": 1e-5, "r_max": 10.0, "n_bins": 1000},
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
from jax.tree_util import Partial

import numpy as np

import graphgp as gp
from graphgp.tree import build_tree, query_preceding_neighbors, query_offset_neighbors
from graphgp.graph import compute_depths_parallel, compute_depths_serial, order_by_depth


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
    covariance = gp.compute_matern_covariance_discrete(
        p=test_params["covariance"]["matern_p"],
        r_min=test_params["covariance"]["r_min"],
        r_max=test_params["covariance"]["r_max"],
        n_bins=test_params["covariance"]["n_bins"],
    )

    # Use test-specific random seed
    test_rng = jr.key(test_params["seed"])
    test_rng, k1 = jr.split(test_rng, 2)

    # Generate points according to distribution
    points = generate_points(k1, test_params["n"], test_params["d"], test_params["distribution"])

    # Build graph with timing
    graph_timings = {}

    if test_params["graph"]["strict"]:
        if test_params["graph"]["fuse"]:
            for i in range(2):
                # Time all-cuda build graph
                if i > 0:
                    del graph
                start = time.perf_counter()
                graph = gp.build_graph(points, n0=test_params["n0"], k=test_params["k"], cuda=test_params["cuda"])
                graph.points.block_until_ready()
                graph_timings["build_graph"] = time.perf_counter() - start

        else:
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
                    points_reordered, split_dims, n0=test_params["n0"], k=test_params["k"], cuda=test_params["cuda"]
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

            graph = gp.Graph(points_final, neighbors_final, offsets, indices_final)
    else:
        raise NotImplementedError("Lazy graph is currently broken.")
        # # Time build_tree for lazy graph
        # start = time.perf_counter()
        # points_reordered, split_dims, indices = build_tree(points)
        # graph_timings["build_tree"] = time.perf_counter() - start

        # # Calculate offsets for lazy graph
        # offsets = [test_params["n0"]]
        # while offsets[-1] < len(points):
        #     next_offset = offsets[-1] * test_params["graph"]["factor"]
        #     offsets.append(int(min(next_offset, len(points))))
        # offsets = tuple(offsets)

        # # Time query_offset_neighbors
        # start = time.perf_counter()
        # neighbors = query_offset_neighbors(points_reordered, split_dims, offsets=offsets, k=test_params["k"], cuda=True)
        # graph_timings["query_neighbors"] = time.perf_counter() - start

        # graph = gp.Graph(points_reordered, neighbors[:, ::-1], offsets, indices)

    

    # Prepare function to benchmark
    if test_params["function"] == "forward":
        func = jax.jit(lambda g, c, xi: gp.generate(g, c, xi, cuda=test_params["cuda"]))
    elif test_params["function"] == "jvp":
        func = jax.jit(lambda g, c, xi: jax.jvp(lambda x, cutoff: gp.generate(g, (c[0], Partial(c[1], cutoff=cutoff)), x, cuda=test_params["cuda"]), (xi, 1.0), (jnp.ones_like(xi), 1.0))[1])
    elif test_params["function"] == "vjp":
        func = jax.jit(lambda g, c, xi: jax.vjp(lambda x, cutoff: gp.generate(g, (c[0], Partial(c[1], cutoff=cutoff)), x, cuda=test_params["cuda"]), xi, 1.0)[1](jnp.ones_like(xi))[0])
    elif test_params["function"] == "grad":
        func = jax.jit(lambda g, c, xi: jax.grad(lambda x: jnp.linalg.norm(gp.generate(g, c, x, cuda=test_params["cuda"])))(xi))
    elif test_params["function"] == "inverse":
        func = jax.jit(lambda g, c, xi: gp.generate_inv(g, c, xi, cuda=test_params["cuda"]))
    elif test_params["function"] == "logdet":
        func = jax.jit(lambda g, c, xi: gp.generate_logdet(g, c, cuda=test_params["cuda"]))
    elif test_params["function"] == "fft":
        graph.points = None
        graph.neighbors = None
        @jax.jit
        def func(g, c, xi):
            for d in range(test_params["d"]):
                xi = jnp.fft.fft(xi)
            return xi

    # Don't re-order points, this significantly impacts runtime
    graph.indices = None

    # Time multiple runs with 1 warmup
    times = []
    for i in range(1 + test_params["timing_runs"]):
        if i > 0:
            del xi, output
        test_rng, k2 = jr.split(test_rng)
        xi = jr.normal(k2, (test_params["n"],))
        start = time.perf_counter()
        output = func(graph, covariance, xi)
        output.block_until_ready()
        times.append(time.perf_counter() - start)
    warmup_time = times[0]
    mean_time = np.mean(times[1:])
    std_time = np.std(times[1:])

    # Memory stats
    stats = func.lower(graph, covariance, xi).compile().memory_analysis()
    total_mem = stats.temp_size_in_bytes + stats.argument_size_in_bytes + stats.output_size_in_bytes

    result = {
        "parameters": test_params.copy(),
        "graph_depth": len(graph.offsets),
        "graph_timings": {k: float(v) for k, v in graph_timings.items()},
        "warmup_time": float(warmup_time),
        "compiled_memory_mb": float(total_mem / (1024**2)),
        "timing": {"mean": float(mean_time), "std": float(std_time), "times": [float(t) for t in times]},
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark graphgp algorithm performance")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--output", type=str, help="Output file for results (if not provided, only prints results)")
    parser.add_argument("--continue", dest="continue_run", action="store_true", 
                        help="Continue an existing benchmark run from output file")

    args = parser.parse_args()

    # Load configuration and existing results
    if args.continue_run:
        if args.config:
            raise ValueError("Cannot specify --config when using --continue to avoid conflicts.")
        if not args.output:
            raise ValueError("Must specify --output when using --continue to load existing results.")
        with open(args.output, "r") as f:
            output_data = json.load(f)
        config = output_data["config"]
        results = output_data["results"]
        print(f"Continuing from {args.output}: {len(results)}/{len(config['runs'])} completed")
    else:
        if not args.config:
            raise ValueError("Must specify --config when not continuing from an existing run.")
        with open(args.config, "r") as f:
            config = json.load(f)
        results = []
        print(f"Running {len(config["runs"])} benchmark combinations...")
    print("-" * 50)

    # Merge defaults with each test combination
    run_configs = []
    for test_params in config["runs"]:
        merged_params = config["defaults"].copy()
        merged_params.update(test_params)
        run_configs.append(merged_params)

    for i in range(len(results), len(run_configs)):
        test_params = run_configs[i]
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
        print(f"Run: {mean_ms:.2f} ± {std_ms:.2f} ms")

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
