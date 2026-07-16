#!/usr/bin/env python3
"""
DUCC NUFFT divergence probe for a GraphGP vector field on a HEALPix x log-r grid.

This script computes two different spectral quantities on the same k-grid:

1. Raw/windowed longitudinal fraction from the vector-field NUFFT

       f_window(k) = |k . Bhat_window(k)|^2 / (k^2 |Bhat_window(k)|^2)

   This is the quantity that tends to ~1/3 for a finite spherical window, even
   for a constant divergence-free vector field.

2. Divergence-NUFFT fraction from a real-space divergence estimate

       f_div(k) = |widehat{div B}(k)|^2 / (k^2 |Bhat_window(k)|^2)

   Here div B is computed on the HEALPix x log-r point cloud first, then NUFFT'd.
   A constant divergence-free field has div B = 0 and therefore f_div = 0 up to
   numerical derivative error. The default derivative is a local Cartesian linear
   fit because it gives exactly zero for a constant vector field; a spherical
   HEALPix/log-r derivative is also available. This is the recommended NUFFT-style
   probe for physical non-solenoidal contamination on a finite HEALPix/log-r domain.

The output bottom-row figure follows the same style as figure_divergence_fd_fft:
modewise fraction map, shell fraction/powers, and modewise-fraction histogram.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

OUTDIR = Path("temp-figures")
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "figure.dpi": 150,
    "savefig.dpi": 240,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def import_base_module():
    here = Path(__file__).resolve().parent
    base_path = here / "demo_healpix_logr_divergence_periodic.py"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Expected {base_path}. Keep this script next to demo_healpix_logr_divergence_periodic.py."
        )
    spec = importlib.util.spec_from_file_location("healpix_logr_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["healpix_logr_base"] = module
    spec.loader.exec_module(module)
    return module


base = import_base_module()


def positive_int_tuple3(text: str) -> tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n, n)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected N or Nx,Ny,Nz")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def default_nthreads() -> int:
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return os.cpu_count() or 1


# -----------------------------------------------------------------------------
# DUCC wrappers
# -----------------------------------------------------------------------------


def ducc_type1_scalar_nufft(points_phys, scalar, weights, *, n_modes, box_min, box_width, epsilon, nthreads):
    """NUFFT of a scalar field sampled at nonuniform points."""
    if not base.HAS_DUCC:
        raise ImportError("ducc0 is not installed in this environment.")
    points_phys = np.asarray(points_phys, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    coord = base.physical_points_to_ducc_coords(points_phys, box_min, box_width)
    strengths = (weights[None, :] * scalar[None, :]).astype(np.complex128, copy=False)
    out = np.zeros((1,) + tuple(n_modes), dtype=np.complex128)
    res = base.ducc_nufft.nu2u(
        points=strengths,
        coord=coord,
        forward=True,
        epsilon=float(epsilon),
        nthreads=int(nthreads),
        out=out,
        fft_order=True,
        periodicity=2.0,
    )
    return np.asarray(res[0])


# -----------------------------------------------------------------------------
# Spectral diagnostics
# -----------------------------------------------------------------------------


def k_grid_from_bhat(Bhat: np.ndarray, box_width: np.ndarray):
    kx, ky, kz = base.k_axes_from_box(Bhat.shape[1:], box_width)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    kmag = np.sqrt(k2)
    return kx, ky, kz, KX, KY, KZ, k2, kmag


def shell_bin_sums(kmag, nonzero, nbins, *arrays):
    kbins = np.linspace(0.0, kmag[nonzero].max() * 1.0001, nbins + 1)
    shell_k = 0.5 * (kbins[:-1] + kbins[1:])
    which = np.digitize(kmag.ravel(), kbins) - 1
    shell_counts = np.zeros(nbins, dtype=int)
    sums = [np.full(nbins, np.nan, dtype=np.float64) for _ in arrays]
    flat_nonzero = nonzero.ravel()
    for i in range(nbins):
        mask = flat_nonzero & (which == i)
        shell_counts[i] = int(mask.sum())
        if mask.any():
            for out, arr in zip(sums, arrays):
                out[i] = float(np.sum(np.asarray(arr).ravel()[mask]).real)
    return kbins, shell_k, shell_counts, sums


def raw_windowed_longitudinal_diag(Bhat: np.ndarray, box_width: np.ndarray, *, nbins: int):
    """The problematic raw k.Bhat diagnostic, included for comparison."""
    kx, ky, kz, KX, KY, KZ, k2, kmag = k_grid_from_bhat(Bhat, box_width)
    Bxhat, Byhat, Bzhat = Bhat[0], Bhat[1], Bhat[2]
    etot = (np.abs(Bxhat) ** 2 + np.abs(Byhat) ** 2 + np.abs(Bzhat) ** 2).real
    kdot = KX * Bxhat + KY * Byhat + KZ * Bzhat
    nonzero = k2 > 0.0
    elong = np.zeros_like(etot)
    elong[nonzero] = (np.abs(kdot[nonzero]) ** 2 / k2[nonzero]).real
    valid = nonzero & (etot > 0.0)
    frac = np.zeros_like(etot)
    frac[valid] = elong[valid] / etot[valid]
    kbins, shell_k, shell_counts, (shell_elong, shell_etot) = shell_bin_sums(
        kmag, nonzero, nbins, elong, etot
    )
    shell_frac = np.divide(shell_elong, shell_etot, out=np.full(nbins, np.nan), where=shell_etot > 0.0)
    total_elong = float(np.nansum(shell_elong))
    total_etot = float(np.nansum(shell_etot))
    total_frac = total_elong / total_etot if total_etot > 0 else np.nan
    kz_idx = int(np.argmin(np.abs(kz)))
    return {
        "name": "raw_windowed_kdotB",
        "kx_shift": np.fft.fftshift(kx),
        "ky_shift": np.fft.fftshift(ky),
        "frac2d": np.fft.fftshift(frac[:, :, kz_idx]),
        "frac_flat": frac[valid],
        "shell_k": shell_k,
        "shell_frac": shell_frac,
        "shell_elong": shell_elong,
        "shell_etot": shell_etot,
        "shell_counts": shell_counts,
        "total_frac": total_frac,
        "total_elong": total_elong,
        "total_etot": total_etot,
    }


def divergence_nufft_fraction_diag(
    Bhat: np.ndarray,
    divhat: np.ndarray,
    box_width: np.ndarray,
    *,
    nbins: int,
    clip_modewise_for_plot: bool = True,
):
    """
    Estimate longitudinal/divergent power from NUFFT(div B), not k.Bhat_window.

    E_div(k) = |divhat(k)|^2 / k^2.
    f_div(k) = E_div(k) / E_tot(k).
    """
    kx, ky, kz, KX, KY, KZ, k2, kmag = k_grid_from_bhat(Bhat, box_width)
    etot = (np.abs(Bhat[0]) ** 2 + np.abs(Bhat[1]) ** 2 + np.abs(Bhat[2]) ** 2).real
    nonzero = k2 > 0.0
    ediv = np.zeros_like(etot)
    ediv[nonzero] = (np.abs(divhat[nonzero]) ** 2 / k2[nonzero]).real
    valid = nonzero & (etot > 0.0)
    frac = np.zeros_like(etot)
    frac[valid] = ediv[valid] / etot[valid]
    frac_for_plot = np.clip(frac, 0.0, 1.0) if clip_modewise_for_plot else frac

    kbins, shell_k, shell_counts, (shell_ediv, shell_etot) = shell_bin_sums(
        kmag, nonzero, nbins, ediv, etot
    )
    shell_frac = np.divide(shell_ediv, shell_etot, out=np.full(nbins, np.nan), where=shell_etot > 0.0)
    total_ediv = float(np.nansum(shell_ediv))
    total_etot = float(np.nansum(shell_etot))
    total_frac = total_ediv / total_etot if total_etot > 0.0 else np.nan
    kz_idx = int(np.argmin(np.abs(kz)))
    return {
        "name": "nufft_of_realspace_divergence",
        "kx_shift": np.fft.fftshift(kx),
        "ky_shift": np.fft.fftshift(ky),
        "frac2d": np.fft.fftshift(frac_for_plot[:, :, kz_idx]),
        "frac2d_unclipped": np.fft.fftshift(frac[:, :, kz_idx]),
        "frac_flat": frac[valid],
        "frac_flat_clipped": np.clip(frac[valid], 0.0, 1.0),
        "shell_k": shell_k,
        "shell_frac": shell_frac,
        "shell_ediv": shell_ediv,
        "shell_etot": shell_etot,
        "shell_counts": shell_counts,
        "total_frac": total_frac,
        "total_ediv": total_ediv,
        "total_etot": total_etot,
        "kbins": kbins,
    }


def subtract_constant_mode_divergence_artifact(B: np.ndarray, div: np.ndarray, grid: dict[str, Any], method: str, *, angular_k: int, local_k: int):
    """
    Remove numerical divergence produced by applying the same derivative estimator
    to the volume-mean constant vector field. This is mostly relevant for the
    spherical-fit derivative, where the radial and angular terms cancel only
    approximately on a finite HEALPix grid.
    """
    B = np.asarray(B, dtype=np.float64)
    mean_vec = np.mean(B, axis=0)
    if np.allclose(mean_vec, 0.0):
        return div, np.zeros_like(div), mean_vec
    Bc = np.zeros_like(B)
    Bc[:] = mean_vec[None, :]
    if method == "spherical":
        c = base.spherical_divergence_contribution(Bc, grid, angular_k=angular_k)["div"].reshape(-1)
    elif method == "local":
        c = base.local_linear_divergence(grid["points"], Bc, k_neighbors=local_k)["div"]
    else:
        raise ValueError(f"unknown divergence method {method!r}")
    return div - c, c, mean_vec


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def figure_title(fig, title, subtitle):
    fig.text(0.06, 0.985, title, ha="left", va="top", fontsize=15.5, fontweight="bold")
    fig.text(0.06, 0.940, subtitle, ha="left", va="top", fontsize=11.2)


def plot_bottom_row(diag: dict[str, Any], *, prefix: str, label: str, raw_diag: dict[str, Any] | None = None):
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.8))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.78, wspace=0.28)
    figure_title(
        fig,
        f"DUCC NUFFT divergence probe without the finite-window 1/3 leakage: {label}",
        r"Plotted quantity: $f_{\nabla\cdot B}(\mathbf{k})=|\widehat{\nabla\cdot B}(\mathbf{k})|^2/[k^2 E_{\rm tot}(\mathbf{k})]$, "
        r"with $E_{\rm tot}=|\hat B_x|^2+|\hat B_y|^2+|\hat B_z|^2$.",
    )

    frac2d = diag["frac2d"]
    vmaxf = np.nanpercentile(frac2d, 99.5)
    if not np.isfinite(vmaxf) or vmaxf <= 0.0:
        vmaxf = 1.0
    im = axs[0].imshow(
        frac2d.T,
        origin="lower",
        extent=[diag["kx_shift"].min(), diag["kx_shift"].max(), diag["ky_shift"].min(), diag["ky_shift"].max()],
        aspect="equal",
        vmin=0.0,
        vmax=vmaxf,
        cmap="magma",
    )
    axs[0].set_title(r"Modewise divergence fraction $f_{\nabla\cdot B}(k_x,k_y,k_z{=}0)$")
    axs[0].set_xlabel(r"$k_x$")
    axs[0].set_ylabel(r"$k_y$")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03, label=r"$f_{\nabla\cdot B}(\mathbf{k})$")

    hfrac = axs[1].plot(
        diag["shell_k"], diag["shell_frac"], "o-", lw=1.8, ms=4.0,
        color="tab:blue", label=r"$E_{\nabla\cdot B}/E_{\rm tot}$"
    )
    htot = axs[1].axhline(
        diag["total_frac"], ls="--", lw=1.2, color="0.4", label=fr"total = {diag['total_frac']:.4g}"
    )
    if raw_diag is not None:
        axs[1].plot(
            raw_diag["shell_k"], raw_diag["shell_frac"], "o-", lw=1.0, ms=3.0,
            color="tab:gray", alpha=0.45, label=r"raw windowed $k\cdot\hat B$"
        )
        axs[1].axhline(raw_diag["total_frac"], ls=":", lw=1.2, color="tab:gray", alpha=0.8,
                       label=fr"raw windowed total = {raw_diag['total_frac']:.3f}")
    axs[1].set_title("Divergence fraction by $|k|$ shell, with powers")
    axs[1].set_xlabel(r"$|k|$")
    axs[1].set_ylabel(r"fraction")
    axpow = axs[1].twinx()
    hdiv = axpow.plot(
        diag["shell_k"], diag["shell_ediv"], "--", color="tab:red", lw=1.8,
        alpha=0.75, label=r"$E_{\nabla\cdot B}$"
    )
    het = axpow.plot(
        diag["shell_k"], diag["shell_etot"], ":", color="tab:green", lw=2.0,
        alpha=0.90, label=r"$E_{\rm tot}$"
    )
    axpow.set_ylabel("shell power")
    axpow.set_yscale("log")
    handles, labels = axs[1].get_legend_handles_labels()
    handles2, labels2 = axpow.get_legend_handles_labels()
    axs[1].legend(handles + handles2, labels + labels2, loc="upper right", fontsize=8.6)

    frac_flat = diag["frac_flat_clipped"]
    high = np.nanpercentile(frac_flat, 99.8) if frac_flat.size else 1.0
    if not np.isfinite(high) or high <= 0.0:
        high = 1.0
    binsf = np.linspace(0.0, high, 70)
    axs[2].hist(frac_flat, bins=binsf, density=True, histtype="step", linewidth=2.0, color="tab:red")
    axs[2].axvline(diag["total_frac"], ls="--", lw=1.2, color="0.45", label=fr"total = {diag['total_frac']:.4g}")
    axs[2].set_title("Distribution of modewise divergence fraction")
    axs[2].set_xlabel(r"$f_{\nabla\cdot B}(\mathbf{k})$")
    axs[2].set_ylabel("normalised density")
    axs[2].legend(loc="best")

    out = OUTDIR / f"{prefix}_{label}_nufft_divergence_probe_bottom_row.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Completed", out)
    return out


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------


def compute_divergence_field(B: np.ndarray, grid: dict[str, Any], *, method: str, angular_k: int, local_k: int):
    if method == "spherical":
        div2d = base.spherical_divergence_contribution(B, grid, angular_k=angular_k)["div"]
        return div2d.reshape(-1)
    if method == "local":
        return base.local_linear_divergence(grid["points"], B, k_neighbors=local_k)["div"]
    raise ValueError(f"Unknown divergence method {method!r}")


def run_case(args, *, periodic_graph: bool, periodic_covariance: bool, label: str):
    grid = base.build_healpix_logr_grid(
        nside=args.nside, n_shells=args.n_shells, r_min=args.r_min, r_max=args.r_max
    )
    weights_flat, shell_weights, redges = base.healpix_logr_volume_weights(
        grid["radii"], int(grid["npix"]), grid["shell_ids"]
    )
    B, boxinfo = base.generate_graphgp_healpix_field(
        grid=grid,
        redges=redges,
        n0=args.n0,
        graph_k=args.graph_k,
        ell=args.ell,
        sigma2=args.sigma2,
        seed=args.seed,
        periodic_graph=periodic_graph,
        periodic_covariance=periodic_covariance,
    )

    div = compute_divergence_field(B, grid, method=args.divergence_method, angular_k=args.angular_k, local_k=args.local_k)
    div_before_const_artifact = div.copy()
    div_const_artifact = np.zeros_like(div)
    mean_vec = np.mean(B, axis=0)
    if not args.no_subtract_constant_divergence_artifact:
        div, div_const_artifact, mean_vec = subtract_constant_mode_divergence_artifact(
            B, div, grid, args.divergence_method, angular_k=args.angular_k, local_k=args.local_k
        )

    Bhat = base.ducc_type1_vector_nufft(
        grid["points"], B, weights_flat,
        n_modes=args.n_modes,
        box_min=boxinfo["box_min"],
        box_width=boxinfo["box_width"],
        epsilon=args.epsilon,
        nthreads=args.nthreads,
    )
    divhat = ducc_type1_scalar_nufft(
        grid["points"], div, weights_flat,
        n_modes=args.n_modes,
        box_min=boxinfo["box_min"],
        box_width=boxinfo["box_width"],
        epsilon=args.epsilon,
        nthreads=args.nthreads,
    )

    raw_diag = raw_windowed_longitudinal_diag(Bhat, boxinfo["box_width"], nbins=args.nbins)
    div_diag = divergence_nufft_fraction_diag(Bhat, divhat, boxinfo["box_width"], nbins=args.nbins)
    fig = plot_bottom_row(div_diag, prefix=args.prefix, label=label, raw_diag=raw_diag)

    B_rms = np.sqrt(np.average(np.sum(B * B, axis=1), weights=weights_flat))
    div_rms = np.sqrt(np.average(div * div, weights=weights_flat))
    div_rms_before = np.sqrt(np.average(div_before_const_artifact * div_before_const_artifact, weights=weights_flat))
    artifact_rms = np.sqrt(np.average(div_const_artifact * div_const_artifact, weights=weights_flat))
    epsilon_rms = args.ell * div_rms / B_rms

    npz_path = OUTDIR / f"{args.prefix}_{label}_nufft_divergence_probe_data.npz"
    np.savez_compressed(
        npz_path,
        B=B,
        points=grid["points"],
        weights=weights_flat,
        radii=grid["radii"],
        div=div,
        div_before_constant_artifact_subtraction=div_before_const_artifact,
        div_constant_artifact=div_const_artifact,
        Bhat=Bhat,
        divhat=divhat,
        shell_k=div_diag["shell_k"],
        shell_frac_divergence_nufft=div_diag["shell_frac"],
        shell_ediv=div_diag["shell_ediv"],
        shell_etot=div_diag["shell_etot"],
        total_frac_divergence_nufft=np.asarray(div_diag["total_frac"]),
        raw_windowed_shell_frac=raw_diag["shell_frac"],
        raw_windowed_total_frac=np.asarray(raw_diag["total_frac"]),
        B_rms=np.asarray(B_rms),
        div_rms=np.asarray(div_rms),
        epsilon_rms=np.asarray(epsilon_rms),
        mean_B=mean_vec,
    )

    summary_path = OUTDIR / f"{args.prefix}_{label}_nufft_divergence_probe_summary.txt"
    lines = [
        "NUFFT divergence probe for HEALPix x log-r GraphGP vector field",
        "===============================================================",
        f"case = {label}",
        f"periodic_graph = {periodic_graph}",
        f"periodic_covariance = {periodic_covariance}",
        f"nside = {args.nside}",
        f"n_shells = {args.n_shells}",
        f"N_points = {grid['points'].shape[0]}",
        f"n_modes = {args.n_modes}",
        f"divergence_method = {args.divergence_method}",
        f"subtract_constant_divergence_artifact = {not args.no_subtract_constant_divergence_artifact}",
        "",
        "Key outputs:",
        f"  raw_windowed_kdotB_total_frac = {raw_diag['total_frac']}",
        f"  divergence_nufft_total_frac = {div_diag['total_frac']}",
        f"  B_rms_volume_weighted = {B_rms}",
        f"  div_rms_volume_weighted = {div_rms}",
        f"  div_rms_before_constant_artifact_subtraction = {div_rms_before}",
        f"  constant_divergence_artifact_rms = {artifact_rms}",
        f"  epsilon_rms = ell * div_rms / B_rms = {epsilon_rms}",
        f"  volume_mean_B = {mean_vec.tolist()}",
        "",
        "Files:",
        f"  figure = {fig}",
        f"  data = {npz_path}",
        "",
        "Interpretation:",
        "  raw_windowed_kdotB_total_frac is the old finite-window longitudinal fraction and may be ~1/3.",
        "  divergence_nufft_total_frac is computed from NUFFT(div B), so constant divergence-free fields give zero numerator.",
        "  This is the quantity to use as the NUFFT probe of physical non-solenoidal contamination on the HEALPix/log-r grid.",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print("Wrote", summary_path)
    print("Wrote", npz_path)
    return {"label": label, "summary": summary_path, "figure": fig, "data": npz_path, "raw": raw_diag, "div": div_diag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=4)
    parser.add_argument("--n-shells", type=int, default=12)
    parser.add_argument("--r-min", type=float, default=0.08)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--n0", type=int, default=256)
    parser.add_argument("--graph-k", type=int, default=8)
    parser.add_argument("--ell", type=float, default=0.09)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-modes", type=positive_int_tuple3, default=(32, 32, 32))
    parser.add_argument("--nbins", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--nthreads", type=int, default=default_nthreads())
    parser.add_argument("--divergence-method", choices=("spherical", "local"), default="local")
    parser.add_argument("--angular-k", type=int, default=8)
    parser.add_argument("--local-k", type=int, default=32)
    parser.add_argument("--no-subtract-constant-divergence-artifact", action="store_true")
    parser.add_argument("--periodic-case", action="store_true", help="also run experimental Cartesian-torus GraphGP embedding")
    parser.add_argument("--prefix", type=str, default="healpix_logr_fl_probe")
    args = parser.parse_args()

    cases = []
    cases.append(run_case(args, periodic_graph=False, periodic_covariance=False, label="nonperiodic"))
    if args.periodic_case:
        cases.append(run_case(args, periodic_graph=True, periodic_covariance=True, label="periodic_torus_embedding"))

    comp = [
        "NUFFT divergence probe comparison",
        "=================================",
        f"prefix = {args.prefix}",
        "",
    ]
    for case in cases:
        comp.append(f"[{case['label']}]")
        comp.append(f"  raw_windowed_kdotB_total_frac = {case['raw']['total_frac']}")
        comp.append(f"  divergence_nufft_total_frac = {case['div']['total_frac']}")
        comp.append(f"  figure = {case['figure']}")
        comp.append(f"  data = {case['data']}")
        comp.append(f"  summary = {case['summary']}")
        comp.append("")
    comparison_path = OUTDIR / f"{args.prefix}_comparison.txt"
    comparison_path.write_text("\n".join(comp) + "\n")
    print("Wrote", comparison_path)


if __name__ == "__main__":
    main()
