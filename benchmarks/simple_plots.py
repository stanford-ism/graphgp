#!/usr/bin/env python3
"""
Plot results from benchmark_simple.py.

Produces a 2×3 panel figure:
  rows  — CUDA / JAX backends
  cols  — graph build time / forward pass time / forward pass memory
Lines are colored by k value.
"""

import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def collect_groups(results, cuda, y_key, std_key=None):
    """Return {k: (ns, vals, stds)} for successful runs matching cuda flag."""
    raw = {}
    for r in results:
        if r.get("cuda") != cuda or r.get("status") != "ok":
            continue
        val = r.get(y_key)
        if val is None:
            continue
        k = r["k"]
        if k not in raw:
            raw[k] = []
        std = r.get(std_key) if std_key else None
        raw[k].append((r["n"], val, std))

    groups = {}
    for k, triples in raw.items():
        triples.sort()
        ns   = np.array([t[0] for t in triples])
        vals = np.array([t[1] for t in triples])
        stds = np.array([t[2] if t[2] is not None else 0.0 for t in triples])
        groups[k] = (ns, vals, stds)
    return groups


def add_linear_ref(ax, n_vals, y_ref_at_min, label="O(N)"):
    """Plot a dashed linear-scaling reference line anchored at y_ref_at_min for n=n_vals[0]."""
    ns = np.array([n_vals[0], n_vals[-1]])
    ax.plot(ns, y_ref_at_min * ns / ns[0], ls=":", c="k", alpha=0.3, label=label)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark_simple.py results")
    parser.add_argument("--input",  default="simple_results.json", help="Input JSON file")
    parser.add_argument("--output", default=None, help="Save figure to this path instead of displaying")
    args = parser.parse_args()

    data    = load_results(args.input)
    results = data["results"]

    k_values = sorted({r["k"] for r in results if r.get("status") == "ok"})
    n_values = sorted({r["n"] for r in results if r.get("status") == "ok"})

    cmap   = plt.get_cmap("inferno")
    colors = {k: cmap((i + 1) / (len(k_values) + 1)) for i, k in enumerate(k_values)}

    # columns: (title, y_key, std_key, y_label, y_unit_scale, y_unit_label)
    # y_unit_scale converts stored units to displayed units
    columns = [
        ("Graph build time",     "graph_build_time_s", None,              "Time [s]",   1,          None),
        ("Forward pass time",    "forward_mean_s",     "forward_std_s",   "Time [s]",   1,          None),
        ("Forward pass memory",  "forward_memory_mb",  None,              "Memory [GB]", 1/1024,    None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150)
    fig.suptitle("GraphGP simple benchmark", fontsize=13, y=1.01)

    for row, (cuda, backend) in enumerate([(True, "CUDA"), (False, "JAX")]):
        for col, (title, y_key, std_key, ylabel, scale, _) in enumerate(columns):
            ax = axes[row, col]
            groups = collect_groups(results, cuda, y_key, std_key)

            plotted_any = False
            for k in k_values:
                if k not in groups:
                    continue
                ns, vals, stds = groups[k]
                scaled_vals = vals * scale
                scaled_stds = stds * scale

                ax.plot(ns, scaled_vals, "o-", color=colors[k], label=f"k={k}", markeredgecolor="none")

                if std_key and np.any(stds > 0):
                    lo = np.maximum(scaled_vals - scaled_stds, scaled_vals * 1e-3)
                    hi = scaled_vals + scaled_stds
                    ax.fill_between(ns, lo, hi, color=colors[k], alpha=0.15)

                plotted_any = True

            if plotted_any and len(n_values) >= 2:
                # Anchor reference line at a visually sensible position
                first_group = groups[k_values[0]]
                anchor = float(first_group[1][0]) * scale
                add_linear_ref(ax, n_values, anchor)

            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Points",
                ylabel=ylabel,
                title=f"{title} ({backend})",
            )
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))

            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                # Put k legend entries first, then the O(N) reference
                k_handles = [h for h, l in zip(handles, labels) if l.startswith("k=")]
                k_labels  = [l for l in labels if l.startswith("k=")]
                ref_h     = [h for h, l in zip(handles, labels) if not l.startswith("k=")]
                ref_l     = [l for l in labels if not l.startswith("k=")]
                ax.legend(k_handles + ref_h, k_labels + ref_l, loc="upper left", fontsize=8)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
