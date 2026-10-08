#!/usr/bin/env python3
"""Small JSON-producing strong-scaling benchmark for distributed GraphGP."""

import argparse
import json
import time

import jax
import numpy as np
from jax.sharding import Mesh

import graphgp as gp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16_384)
    parser.add_argument("--n0", type=int, default=128)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    n_devices = jax.device_count()
    if args.n % n_devices:
        raise ValueError("--n must be divisible by the number of devices")
    key_points, key_xi = jax.random.split(jax.random.key(42))
    points = jax.random.uniform(key_points, (args.n, 3))
    graph = gp.build_graph(points, n0=args.n0, k=args.k)
    owners = np.repeat(np.arange(n_devices), args.n // n_devices)
    plan = gp.distributed.partition_graph(graph, owners)
    covariance = gp.extras.matern_kernel(
        p=0,
        variance=1.0,
        cutoff=0.2,
        r_min=1e-5,
        r_max=2.0,
        n_bins=512,
        jitter=1e-5,
    )
    xi = jax.random.normal(key_xi, (args.n,))
    mesh = Mesh(np.asarray(jax.devices()), ("space",))
    generate = jax.jit(lambda x: gp.distributed.generate(plan, covariance, x, mesh=mesh))

    start = time.perf_counter()
    lowered = generate.lower(xi)
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start
    memory = compiled.memory_analysis()
    compiled(xi).block_until_ready()
    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        compiled(xi).block_until_ready()
        timings.append(time.perf_counter() - start)

    stats = gp.distributed.partition_stats(plan)
    result = {
        "n": args.n,
        "n0": args.n0,
        "k": args.k,
        "devices": n_devices,
        "compile_seconds": compile_seconds,
        "runtime_seconds": timings,
        "runtime_mean_seconds": float(np.mean(timings)),
        "runtime_std_seconds": float(np.std(timings)),
        "temporary_memory_bytes": memory.temp_size_in_bytes,
        "argument_memory_bytes": memory.argument_size_in_bytes,
        "output_memory_bytes": memory.output_size_in_bytes,
        "partition": stats.__dict__,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
