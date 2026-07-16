#!/usr/bin/env python3
"""
Benchmark GraphGP vector-field generation: JAX path vs CUDA path.

Examples
--------
Small local laptop run:

    python benchmark_vector_cuda.py --sizes 8 10 12 14 16 18 20 --ell 0.075 --k 8 --n0 200

More expensive local run:

    python benchmark_vector_cuda.py --sizes 20 22 24 26 28 30 --ell 0.075 --k 8 --n0 200 --repeats 3

CUDA-only run, useful when the JAX reference path becomes too memory-heavy:

    python benchmark_vector_cuda.py --sizes 20 24 28 32 36 --ell 0.075 --k 8 --n0 200 --cuda-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import pathlib
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import graphgp as gp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8, 10, 12, 14, 16, 18, 20],
        help="Grid side lengths. Total points are n^3.",
    )
    parser.add_argument("--ell", type=float, default=0.075, help="RBF correlation length.")
    parser.add_argument("--sigma2", type=float, default=1.0, help="Kernel amplitude.")
    parser.add_argument("--k", type=int, default=8, help="Number of GraphGP neighbours.")
    parser.add_argument("--n0", type=int, default=200, help="Number of initial dense points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed repeats after warm-up.")
    parser.add_argument("--cuda-only", action="store_true", help="Only benchmark CUDA path.")
    parser.add_argument("--no-periodic", action="store_true", help="Disable periodic covariance.")
    parser.add_argument(
        "--out",
        type=str,
        default="temp-figures/vector_cuda_benchmark.csv",
        help="CSV output path.",
    )

    return parser.parse_args()


def make_points(n: int):
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    return jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))


def time_call(fn, repeats: int):
    # Warm-up / compile
    y = fn()
    y.block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = fn()
        y.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "min": float(np.min(times)),
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
    }


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.8g}"
    return x


def main() -> None:
    args = parse_args()

    outpath = pathlib.Path(args.out)
    outpath.parent.mkdir(exist_ok=True)

    periodic = not args.no_periodic

    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())
    print(f"ell={args.ell}, sigma2={args.sigma2}, k={args.k}, n0={args.n0}")
    print(f"repeats={args.repeats}, periodic={periodic}, cuda_only={args.cuda_only}")
    print(f"output CSV: {outpath}")

    rows = []

    for n in args.sizes:
        N = n**3

        if args.n0 >= N:
            print(f"\nSkipping n={n}: n0={args.n0} >= N={N}")
            continue

        print("\n" + "=" * 72)
        print(f"n={n}, N={N}")

        rng = jr.key(args.seed)
        points = make_points(n)

        print("Building graph...")
        t0 = time.perf_counter()
        graph = gp.build_graph(points, n0=args.n0, k=args.k, cuda=True)
        graph.points.block_until_ready()
        graph_build_time = time.perf_counter() - t0
        print(f"graph build time: {graph_build_time:.4f} s")

        cov_elem = gp.extras.make_div_free_rbf_covariance(
            ell=args.ell,
            sigma2=args.sigma2,
            periodic=periodic,
        )

        xi = jr.normal(rng, (N, 3))
        xi.block_until_ready()

        tj = None
        tc = None
        max_abs = None
        rel_rms = None

        if not args.cuda_only:
            print("Timing JAX reference path...")
            try:
                tj = time_call(
                    lambda: gp.generate_vector(graph, cov_elem, xi, cuda=False),
                    repeats=args.repeats,
                )
                print(
                    "JAX median:",
                    f"{tj['median']:.6f} s",
                    "min:",
                    f"{tj['min']:.6f} s",
                )
            except Exception as exc:
                print("JAX path failed:", repr(exc))
                tj = None
                gc.collect()

        print("Timing CUDA path...")
        try:
            tc = time_call(
                lambda: gp.generate_vector(graph, cov_elem, xi, cuda=True),
                repeats=args.repeats,
            )
            print(
                "CUDA median:",
                f"{tc['median']:.6f} s",
                "min:",
                f"{tc['min']:.6f} s",
            )
        except Exception as exc:
            print("CUDA path failed:", repr(exc))
            tc = None
            gc.collect()

        speedup_median = None
        speedup_min = None

        if tj is not None and tc is not None:
            speedup_median = tj["median"] / tc["median"]
            speedup_min = tj["min"] / tc["min"]

            print("Checking numerical agreement...")
            B_jax = gp.generate_vector(graph, cov_elem, xi, cuda=False)
            B_cuda = gp.generate_vector(graph, cov_elem, xi, cuda=True)
            B_jax.block_until_ready()
            B_cuda.block_until_ready()

            diff = B_cuda - B_jax
            max_abs = float(jnp.max(jnp.abs(diff)))
            rel_rms = float(
                jnp.sqrt(jnp.mean(diff**2)) /
                jnp.sqrt(jnp.mean(B_jax**2))
            )

            print("max abs diff:", f"{max_abs:.6e}")
            print("relative RMS diff:", f"{rel_rms:.6e}")
            print("speedup median JAX/CUDA:", f"{speedup_median:.3f}x")

            del B_jax, B_cuda, diff

        row = {
            "n": n,
            "N": N,
            "ell": args.ell,
            "sigma2": args.sigma2,
            "k": args.k,
            "n0": args.n0,
            "periodic": periodic,
            "graph_build_s": graph_build_time,

            "jax_min_s": None if tj is None else tj["min"],
            "jax_median_s": None if tj is None else tj["median"],
            "jax_mean_s": None if tj is None else tj["mean"],
            "jax_std_s": None if tj is None else tj["std"],

            "cuda_min_s": None if tc is None else tc["min"],
            "cuda_median_s": None if tc is None else tc["median"],
            "cuda_mean_s": None if tc is None else tc["mean"],
            "cuda_std_s": None if tc is None else tc["std"],

            "speedup_median": speedup_median,
            "speedup_min": speedup_min,
            "max_abs_diff": max_abs,
            "rel_rms_diff": rel_rms,
        }

        rows.append(row)

        del points, graph, xi
        gc.collect()

    if rows:
        fieldnames = list(rows[0].keys())
        with outpath.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: fmt(v) for k, v in row.items()})

        print("\nSaved benchmark CSV:", outpath)

    print("\nDone.")


if __name__ == "__main__":
    main()
