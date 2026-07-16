#!/usr/bin/env python3
"""
demo_fft_n0_k_sweep.py

Sweep GraphGP periodic-vector-field FFT diagnostics over n0 and k.

This script generates one figure with four panels, all plotted for several
(n0, k) settings:

    1. shell-summed longitudinal power E_L(|k|)
    2. shell-summed total vector power E_tot(|k|)
    3. shell fraction f_L(|k|) = E_L(|k|) / E_tot(|k|)
    4. distribution of modewise f_L(k)

The intended diagnostic is:
    - increasing n0 should improve large-scale / low-wavenumber structure;
    - increasing k should usually reduce longitudinal leakage overall;
    - the total longitudinal fraction sum(E_L)/sum(E_tot) is printed and saved.

Outputs are written to temp-figures/ by default.

Run from the repository root, e.g.
    python demo_fft_n0_k_sweep.py

For a quicker/slower run, edit the parameters in main().
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTDIR = Path("temp-figures")
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "figure.dpi": 150,
    "savefig.dpi": 240,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def import_local_graphgp():
    """Import graphgp when this file is run from either repo root or package dir."""
    spec = importlib.util.find_spec("graphgp")
    if spec is not None:
        import graphgp as gp  # type: ignore
        return gp

    here = Path(__file__).resolve().parent
    candidates = [here / "graphgp" / "__init__.py", here / "__init__.py"]
    for init_path in candidates:
        if init_path.exists():
            spec = importlib.util.spec_from_file_location(
                "graphgp",
                init_path,
                submodule_search_locations=[str(init_path.parent)],
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["graphgp"] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError("Could not find/import local graphgp package.")


gp = import_local_graphgp()


# ------------------------------
# Grid, graph, and FFT diagnostics
# ------------------------------


def make_points_3d(n: int):
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    return jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3)), np.asarray(xs)


def min_image_delta(a, b, box=1.0):
    d = np.asarray(a) - np.asarray(b)
    return d - box * np.round(d / box)


def predecessor_neighbors(points_tree, n0: int, k: int, periodic: bool = True, boxsize: float = 1.0):
    """k nearest predecessors in tree order, optionally using minimum-image distance."""
    pts = np.asarray(points_tree)
    neigh = np.empty((len(pts) - n0, k), dtype=int)
    for i in range(n0, len(pts)):
        prev = pts[:i]
        if periodic:
            d = prev - pts[i]
            d = d - boxsize * np.round(d / boxsize)
            dist = np.linalg.norm(d, axis=1)
        else:
            dist = np.linalg.norm(prev - pts[i], axis=1)
        idx = np.argpartition(dist, k - 1)[:k]
        idx = idx[np.argsort(dist[idx], kind="stable")]
        neigh[i - n0] = idx
    return jnp.asarray(neigh)


def build_graph_local(points, *, n0: int, k: int, periodic: bool = True, boxsize: float = 1.0):
    """Build GraphGP graph with periodic predecessor search but normal topo
    print('Savedlogical ordering."""
    points_tree, split_dims, indices = gp.build_tree(points)
    neighbors = predecessor_neighbors(points_tree, n0=n0, k=k, periodic=periodic, boxsize=boxsize)
    depths = gp.compute_depths(neighbors, n0=n0)
    pts_ord, idx_ord, neigh_ord, depths = gp.order_by_depth(points_tree, indices, neighbors, depths)
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
    graph = gp.Graph(pts_ord, neigh_ord, offsets, idx_ord)
    return graph


def divergence_fft(Bgrid: np.ndarray, spacing: Tuple[float, float, float]):
    nx, ny, nz = Bgrid.shape[:3]
    dx, dy, dz = spacing
    Bhat = np.fft.fftn(Bgrid, axes=(0, 1, 2))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    divhat = 1j * (KX * Bhat[..., 0] + KY * Bhat[..., 1] + KZ * Bhat[..., 2])
    div = np.fft.ifftn(divhat, axes=(0, 1, 2)).real
    return div, (kx, ky, kz, KX, KY, KZ, Bhat)


def fft_power_diagnostics(Bgrid: np.ndarray, spacing: Tuple[float, float, float], nbins: int = 16):
    """
    Compute shell powers and modewise longitudinal fraction.

    E_L(k) = |k . Bhat(k)|^2 / |k|^2
    E_tot(k) = |Bxhat|^2 + |Byhat|^2 + |Bzhat|^2
    f_L(k) = E_L(k) / E_tot(k)
    """
    _, (kx, ky, kz, KX, KY, KZ, Bhat) = divergence_fft(Bgrid, spacing)
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    kmag = np.sqrt(k2)
    etot = np.abs(Bhat[..., 0]) ** 2 + np.abs(Bhat[..., 1]) ** 2 + np.abs(Bhat[..., 2]) ** 2
    kdot = KX * Bhat[..., 0] + KY * Bhat[..., 1] + KZ * Bhat[..., 2]
    mask = k2 > 0
    elong = np.abs(kdot) ** 2 / np.where(mask, k2, 1.0)

    kbins = np.linspace(0.0, kmag[mask].max() * 1.0001, nbins + 1)
    shell_mid = 0.5 * (kbins[:-1] + kbins[1:])
    shell_elong = np.full(nbins, np.nan)
    shell_etot = np.full(nbins, np.nan)
    shell_frac = np.full(nbins, np.nan)
    shell_counts = np.zeros(nbins, dtype=int)
    for ib in range(nbins):
        m = mask & (kmag >= kbins[ib]) & (kmag < kbins[ib + 1])
        shell_counts[ib] = int(m.sum())
        if shell_counts[ib] > 0:
            shell_elong[ib] = float(np.sum(elong[m]).real)
            shell_etot[ib] = float(np.sum(etot[m]).real)
            shell_frac[ib] = shell_elong[ib] / shell_etot[ib]

    frac_mode = np.zeros_like(etot.real)
    good = (k2 > 0) & (etot > 0)
    frac_mode[good] = (np.abs(kdot[good]) ** 2 / k2[good]) / etot[good]
    total_elong = float(np.sum(elong[mask]).real)
    total_etot = float(np.sum(etot[mask]).real)
    total_frac = total_elong / total_etot
    return {
        "shell_k": shell_mid,
        "shell_elong": shell_elong,
        "shell_etot": shell_etot,
        "shell_frac": shell_frac,
        "shell_counts": shell_counts,
        "frac_mode": frac_mode[good],
        "total_elong": total_elong,
        "total_etot": total_etot,
        "total_frac": total_frac,
    }


def seam_ratio(Bgrid: np.ndarray):
    ratios = []
    for ax in range(3):
        seam = Bgrid.take(indices=0, axis=ax) - Bgrid.take(indices=Bgrid.shape[ax] - 1, axis=ax)
        seam_rms = np.sqrt(np.mean(np.sum(seam ** 2, axis=-1)))
        diffs = np.diff(Bgrid, axis=ax)
        interior_rms = np.sqrt(np.mean(np.sum(diffs ** 2, axis=-1)))
        ratios.append(float(seam_rms / interior_rms))
    return tuple(ratios)


# ------------------------------
# Sweep and plotting
# ------------------------------


def run_sweep(
    *,
    n: int,
    ell: float,
    sigma2: float,
    n0_values: Iterable[int],
    k_values: Iterable[int],
    seeds: Iterable[int],
    nbins: int = 16,
):
    points, coords_1d = make_points_3d(n)
    spacing = (1.0 / n, 1.0 / n, 1.0 / n)
    cov = gp.extras.make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=True)

    records = []
    curves: Dict[Tuple[int, int], Dict[str, list]] = {}

    N = int(points.shape[0])
    for n0 in n0_values:
        for k in k_values:
            if n0 < k:
                print(f"Skipping n0={n0}, k={k}: n0 must be >= k")
                continue
            if n0 >= N:
                print(f"Skipping n0={n0}, k={k}: n0 must be < N={N}")
                continue

            print(f"Building periodic graph for n0={n0}, k={k} ...")
            graph = build_graph_local(points, n0=n0, k=k, periodic=True, boxsize=1.0)
            cfg = (n0, k)
            curves[cfg] = {"elong": [], "etot": [], "frac": [], "frac_mode": []}

            for seed in seeds:
                print(f"  seed={seed}")
                xi = jr.normal(jr.key(seed), (N, 3))
                B = np.asarray(gp.generate_vector(graph, cov, xi, fast_jit=True)).reshape((n, n, n, 3))
                if not np.isfinite(B).all():
                    raise RuntimeError(f"NaNs encountered for n0={n0}, k={k}, seed={seed}")

                diag = fft_power_diagnostics(B, spacing, nbins=nbins)
                curves[cfg]["elong"].append(diag["shell_elong"])
                curves[cfg]["etot"].append(diag["shell_etot"])
                curves[cfg]["frac"].append(diag["shell_frac"])
                curves[cfg]["frac_mode"].append(diag["frac_mode"])
                sx, sy, sz = seam_ratio(B)
                records.append({
                    "n": n,
                    "ell": ell,
                    "sigma2": sigma2,
                    "n0": n0,
                    "k": k,
                    "seed": seed,
                    "total_longitudinal_power": diag["total_elong"],
                    "total_vector_power": diag["total_etot"],
                    "total_longitudinal_fraction": diag["total_frac"],
                    "low_k_fraction_first_shell": diag["shell_frac"][0],
                    "seam_ratio_x": sx,
                    "seam_ratio_y": sy,
                    "seam_ratio_z": sz,
                })
                del B, xi
                gc.collect()

    df = pd.DataFrame(records)
    shell_k = diag["shell_k"] if records else np.array([])
    return df, shell_k, curves, coords_1d, spacing


def mean_sem(arrs):
    arr = np.asarray(arrs, dtype=float)
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] > 1:
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def plot_n0_k_sweep(df: pd.DataFrame, shell_k: np.ndarray, curves: dict, *, outpath: Path):
    if df.empty:
        raise ValueError("No sweep results to plot.")

    k_values = sorted(df["k"].unique())
    n0_values = sorted(df["n0"].unique())
    colors = {k: plt.get_cmap("viridis")((i + 0.5) / max(len(k_values), 1)) for i, k in enumerate(k_values)}

    # Make the n0 coding genuinely easy to distinguish.
    # Example with the default values:
    #   n0 = 128  -> solid
    #   n0 = 512  -> long dashed
    #   n0 = 1024 -> dotted
    distinct_linestyles = [
        "-",
        (0, (8.0, 3.0)),
        (0, (1.2, 1.8)),
        (0, (10.0, 2.5, 2.0, 2.5)),
        (0, (6.0, 2.0, 1.2, 2.0)),
    ]
    styles = {n0: distinct_linestyles[i % len(distinct_linestyles)] for i, n0 in enumerate(n0_values)}

    fig, axs = plt.subplots(2, 2, figsize=(15.8, 10.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16, wspace=0.10, hspace=0.18)
    axL, axT, axF, axH = axs.ravel()

    for (n0, k), data in sorted(curves.items(), key=lambda item: (item[0][1], item[0][0])):
        color = colors[k]
        ls = styles[n0]

        mL, sL = mean_sem(data["elong"])
        mT, sT = mean_sem(data["etot"])
        mF, sF = mean_sem(data["frac"])
        axL.plot(shell_k, mL, marker="o", ms=3.2, lw=1.9, color=color, ls=ls)
        axL.fill_between(shell_k, np.maximum(mL - sL, 1e-300), mL + sL, color=color, alpha=0.08)
        axT.plot(shell_k, mT, marker="o", ms=3.2, lw=1.9, color=color, ls=ls)
        axT.fill_between(shell_k, np.maximum(mT - sT, 1e-300), mT + sT, color=color, alpha=0.08)
        axF.plot(shell_k, mF, marker="o", ms=3.2, lw=1.9, color=color, ls=ls)
        axF.fill_between(shell_k, np.maximum(mF - sF, 0.0), mF + sF, color=color, alpha=0.08)

        frac_mode = np.concatenate([np.asarray(x) for x in data["frac_mode"]])
        hi = np.nanpercentile(frac_mode, 99.5)
        bins = np.linspace(0.0, hi if hi > 0 else 1.0, 70)
        axH.hist(frac_mode, bins=bins, density=True, histtype="step", lw=1.9, color=color, ls=ls)
        total_mean = df[(df["n0"] == n0) & (df["k"] == k)]["total_longitudinal_fraction"].mean()
        axH.axvline(total_mean, color=color, ls=ls, lw=1.0, alpha=0.35)

    axL.set_title(r"Shell longitudinal power $E_L(|k|)$")
    axL.set_ylabel(r"$E_L$ (shell sum)")
    axL.set_yscale("log")

    axT.set_title(r"Shell total vector power $E_{\rm tot}(|k|)$")
    axT.set_ylabel(r"$E_{\rm tot}$ (shell sum)")
    axT.set_yscale("log")

    axF.set_title(r"Shell longitudinal fraction $E_L/E_{\rm tot}$")
    axF.set_xlabel(r"$|k|$")
    axF.set_ylabel(r"$f_L(|k|)$")
    axF.set_ylim(bottom=0.0)

    axH.set_title(r"Modewise distribution of longitudinal fraction")
    axH.set_xlabel(r"$f_L(\mathbf{k})$")
    axH.set_ylabel("normalised density")

    color_handles = [plt.Line2D([0], [0], color=colors[k], lw=2.6, label=fr"$k={k}$") for k in k_values]
    style_handles = [plt.Line2D([0], [0], color="0.15", ls=styles[n0], lw=2.8, label=fr"$n_0={n0}$") for n0 in n0_values]

    fig.legend(
        handles=color_handles,
        labels=[h.get_label() for h in color_handles],
        loc="lower center",
        bbox_to_anchor=(0.28, 0.045),
        ncol=len(color_handles),
        frameon=True,
        title="Colour = k",
    )
    fig.legend(
        handles=style_handles,
        labels=[h.get_label() for h in style_handles],
        loc="lower center",
        bbox_to_anchor=(0.74, 0.045),
        ncol=len(style_handles),
        frameon=True,
        title=r"Linestyle = $n_0$",
    )

    fig.suptitle(
        "FFT longitudinal leakage sweep over GraphGP initial layer n0 and neighbour count k\n"
        "Colour encodes k; linestyle encodes n0. Lower curves in the fraction panels indicate less longitudinal leakage.",
        y=0.965,
        fontsize=14.5,
        fontweight="bold",
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")

def write_summary(df: pd.DataFrame, *, outpath: Path):
    if df.empty:
        outpath.write_text("No results.\n")
        return
    grouped = df.groupby(["n0", "k"], as_index=False).agg(
        total_longitudinal_fraction_mean=("total_longitudinal_fraction", "mean"),
        total_longitudinal_fraction_std=("total_longitudinal_fraction", "std"),
        low_k_fraction_first_shell_mean=("low_k_fraction_first_shell", "mean"),
        low_k_fraction_first_shell_std=("low_k_fraction_first_shell", "std"),
        seam_ratio_x_mean=("seam_ratio_x", "mean"),
        seam_ratio_y_mean=("seam_ratio_y", "mean"),
        seam_ratio_z_mean=("seam_ratio_z", "mean"),
    )
    lines = [
        "FFT periodic diagnostics: n0/k sweep",
        "====================================",
        "",
        "Interpretation guide:",
        "  total_longitudinal_fraction = sum_k E_L(k) / sum_k E_tot(k)",
        "  low_k_fraction_first_shell probes the largest resolved Fourier scales.",
        "  Increasing n0 should mostly help the low-k / large-scale shells.",
        "  Increasing k should usually improve the local conditional approximation and reduce total leakage.",
        "",
        grouped.to_string(index=False),
        "",
    ]
    outpath.write_text("\n".join(lines))
    print(f"Saved {outpath}")


def main():
    # Defaults are intentionally laptop-friendly. Increase n and n0_values for a production run.
    n = 20
    ell = 0.08
    sigma2 = 1.0
    n0_values = (128, 512, 2048)
    k_values = (6, 9, 12, 18)
    seeds = (0, 1, 2)
    nbins = 14

    df, shell_k, curves, *_ = run_sweep(
        n=n,
        ell=ell,
        sigma2=sigma2,
        n0_values=n0_values,
        k_values=k_values,
        seeds=seeds,
        nbins=nbins,
    )
    df.to_csv(OUTDIR / "fft_n0_k_sweep_metrics.csv", index=False)
    write_summary(df, outpath=OUTDIR / "fft_n0_k_sweep_summary.txt")
    plot_n0_k_sweep(df, shell_k, curves, outpath=OUTDIR / "figure_fft_n0_k_sweep.png")
    print("Wrote outputs to", OUTDIR.resolve())


if __name__ == "__main__":
    main()
