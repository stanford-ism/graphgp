#!/usr/bin/env python3
"""
Simple graphgp benchmark: sweeps n, k, and backend (JAX/CUDA).

Fixed: gaussian points in 3D, matern-0 covariance, n0=1000.
Variable: n (1e6–1e9), k (4, 16, 64), cuda (True/False).

Measures per run:
  - Total graph build time (second of two builds, to exclude JIT warmup)
  - Forward pass time (mean ± std over TIMING_RUNS after one warmup)
  - Forward pass compiled memory (from XLA memory_analysis)
"""

import gc
import json
import time
import argparse
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import graphgp as gp

# --- Fixed configuration ---
N0 = 1000
D = 3
SEED = 137
TIMING_RUNS = 3

COVARIANCE_PARAMS = dict(p=0, variance=1.0, cutoff=1.0, r_min=1e-5, r_max=10.0, n_bins=1000)

N_VALUES = [1_000_000, 3_000_000]#, 10_000_000, 30_000_000, 100_000_000, 300_000_000, 1_000_000_000]
K_VALUES = [2, 8]
CUDA_VALUES = [True, False]


def clear_memory():
    """Flush JAX caches and trigger Python garbage collection."""
    gc.collect()
    jax.clear_caches()
    try:
        jax.lib.xla_bridge.get_backend("gpu").defragment()
    except Exception:
        pass


def is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in ["resource_exhausted", "out of memory", "oom", "allocation failed"])


def run_benchmark(n: int, k: int, cuda: bool) -> dict:
    result = {"n": n, "k": k, "cuda": cuda, "status": "ok"}

    covariance = gp.extras.matern_kernel(**COVARIANCE_PARAMS)
    rng = jr.key(SEED)

    # --- Graph build ---
    # Run twice; keep the second time to exclude JIT compilation overhead.
    graph = None
    try:
        rng, k1 = jr.split(rng)
        points = jr.normal(k1, (n, D))

        for i in range(2):
            if graph is not None:
                del graph
            start = time.perf_counter()
            graph = gp.build_graph(points, n0=N0, k=k, cuda=cuda)
            graph.points.block_until_ready()
            build_time = time.perf_counter() - start

        del points
        result["graph_build_time_s"] = float(build_time)
        result["graph_depth"] = len(graph.offsets)

    except Exception as e:
        result["status"] = "graph_oom" if is_oom(e) else "graph_error"
        result["error"] = str(e)[:300]
        del graph
        clear_memory()
        return result

    # --- Forward pass ---
    xi = None
    output = None
    func = None
    try:
        graph.indices = None  # not needed for generate()

        func = jax.jit(lambda g, c, xi: gp.generate(g, c, xi, cuda=cuda))

        # Warmup run (triggers JIT compilation)
        rng, k2 = jr.split(rng)
        xi = jr.normal(k2, (n,))
        output = func(graph, covariance, xi)
        output.block_until_ready()

        # Timed runs
        times = []
        for _ in range(TIMING_RUNS):
            del xi, output
            rng, k2 = jr.split(rng)
            xi = jr.normal(k2, (n,))
            t0 = time.perf_counter()
            output = func(graph, covariance, xi)
            output.block_until_ready()
            times.append(time.perf_counter() - t0)

        result["forward_mean_s"] = float(np.mean(times))
        result["forward_std_s"] = float(np.std(times))

        # XLA compiled memory analysis
        try:
            mem = func.lower(graph, covariance, xi).compile().memory_analysis()
            total_bytes = mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes
            result["forward_memory_mb"] = float(total_bytes / 1024**2)
        except Exception as mem_e:
            result["forward_memory_mb"] = None
            result["memory_note"] = str(mem_e)[:150]

    except Exception as e:
        result["status"] = "forward_oom" if is_oom(e) else "forward_error"
        result["error"] = str(e)[:300]

    # --- Cleanup ---
    del graph
    if xi is not None:
        del xi
    if output is not None:
        del output
    if func is not None:
        del func
    clear_memory()

    return result


def main():
    parser = argparse.ArgumentParser(description="Simple graphgp benchmark sweeping n, k, and backend")
    parser.add_argument(
        "--output",
        type=str,
        default="simple_results.json",
        help="Output JSON file (default: simple_results.json)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Resume an existing run from the output file",
    )
    args = parser.parse_args()

    runs = [
        {"n": n, "k": k, "cuda": cuda}
        for cuda in CUDA_VALUES
        for n in N_VALUES
        for k in K_VALUES
    ]

    if args.continue_run:
        with open(args.output) as f:
            data = json.load(f)
        results = data["results"]
        completed = {(r["n"], r["k"], r["cuda"]) for r in results}
        print(f"Continuing: {len(results)}/{len(runs)} done")
    else:
        results = []
        completed = set()
        print(f"Running {len(runs)} benchmark combinations")

    print("-" * 60)

    for i, run in enumerate(runs):
        n, k, cuda = run["n"], run["k"], run["cuda"]
        if (n, k, cuda) in completed:
            continue

        backend = "CUDA" if cuda else "JAX "
        print(f"[{i + 1:2d}/{len(runs)}] n={n:.0e}  k={k:2d}  {backend}", end="  ", flush=True)

        result = run_benchmark(n, k, cuda)
        results.append(result)

        if result["status"] == "ok":
            build_ms = 1000 * result["graph_build_time_s"]
            fwd_ms = 1000 * result["forward_mean_s"]
            fwd_std_ms = 1000 * result["forward_std_s"]
            mem_mb = result.get("forward_memory_mb")
            mem_str = f"{mem_mb:.0f} MB" if mem_mb is not None else "N/A"
            print(f"build={build_ms:.0f} ms  fwd={fwd_ms:.0f}±{fwd_std_ms:.0f} ms  mem={mem_str}")
        else:
            print(f"SKIPPED ({result['status']})")

        with open(args.output, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "n_values": N_VALUES,
                        "k_values": K_VALUES,
                        "cuda_values": CUDA_VALUES,
                        "n0": N0,
                        "d": D,
                        "seed": SEED,
                        "timing_runs": TIMING_RUNS,
                        "covariance": COVARIANCE_PARAMS,
                    },
                    "results": results,
                },
                f,
                indent=2,
            )

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
