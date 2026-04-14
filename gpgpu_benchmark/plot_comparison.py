#!/usr/bin/env python3
"""
Plot forward-pass timing comparison between graphgp and GpGpU.

Usage:
    python gpgpu_benchmark/plot_comparison.py

Expects:
    gpgpu_benchmark/graphgp_results.json  (from benchmarks/benchmark.py)
    gpgpu_benchmark/gpgpu_results.json    (from benchmark_gpgpu.R)
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def load_graphgp_results(path):
    """Extract (n, time_mean, time_std) grouped by cuda flag."""
    with open(path) as f:
        data = json.load(f)

    jax_series = []  # cuda=False
    cuda_series = []  # cuda=True

    for result in data["results"]:
        p = result["parameters"]
        entry = (p["n"], result["timing"]["mean"], result["timing"]["std"])
        if p.get("cuda", False):
            cuda_series.append(entry)
        else:
            jax_series.append(entry)

    jax_series.sort()
    cuda_series.sort()
    return jax_series, cuda_series


def load_gpgpu_results(path):
    """Extract (n, time_mean, time_std) from GpGpU results."""
    with open(path) as f:
        data = json.load(f)

    series = []
    for result in data["results"]:
        p = result["parameters"]
        series.append((p["n"], result["timing"]["mean"], result["timing"]["std"]))

    series.sort()
    return series


def plot(jax, cuda, gpgpu, output_path):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    for series, label, marker, color in [
        (jax, "graphgp (JAX)", "s", "C0"),
        (cuda, "graphgp (CUDA)", "o", "C1"),
        (gpgpu, "GpGpU (GPU)", "^", "C2"),
    ]:
        if not series:
            continue
        n_vals = [s[0] for s in series]
        means = [s[1] for s in series]
        stds = [s[2] for s in series]
        ax.errorbar(n_vals, means, yerr=stds, label=label,
                     marker=marker, color=color, capsize=3, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of points (n)")
    ax.set_ylabel("Time [s]")
    ax.set_title("Forward pass, k=30, d=3")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    graphgp_path = "gpgpu_benchmark/graphgp_results.json"
    gpgpu_path = "gpgpu_benchmark/gpgpu_results.json"
    output_path = "gpgpu_benchmark/gpgpu_comparison.pdf"

    jax_series, cuda_series = load_graphgp_results(graphgp_path)
    gpgpu_series = load_gpgpu_results(gpgpu_path)

    plot(jax_series, cuda_series, gpgpu_series, output_path)
