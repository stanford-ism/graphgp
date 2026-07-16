#!/usr/bin/env python3
"""
Compare GraphGP vector-field generation using the JAX and CUDA paths.

Default:
    n = 20
    ell = 0.05

Output:
    temp-figures/jax_vs_cuda_vector_quiver_n{n}_ell{ell}.png
"""

from __future__ import annotations

import argparse
import pathlib

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

import graphgp as gp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Grid side length. Total points are n^3.")
    parser.add_argument("--ell", type=float, default=0.05, help="RBF correlation length.")
    parser.add_argument("--sigma2", type=float, default=1.0, help="Kernel amplitude.")
    parser.add_argument("--n0", type=int, default=64, help="Number of initial dense points.")
    parser.add_argument("--k", type=int, default=8, help="Number of GraphGP neighbours.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--stride", type=int, default=1, help="Quiver stride on the plotted slice.")
    parser.add_argument("--no-periodic", action="store_true", help="Disable periodic covariance.")
    parser.add_argument("--outdir", type=str, default="temp-figures", help="Output directory.")
    return parser.parse_args()


def make_grid_points(n: int):
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))
    return points, xs


def slice_xy(Bgrid: np.ndarray, zslice: int, stride: int):
    Bx = Bgrid[::stride, ::stride, zslice, 0]
    By = Bgrid[::stride, ::stride, zslice, 1]
    Bz = Bgrid[::stride, ::stride, zslice, 2]
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    return Bx.T, By.T, Bmag.T


def main() -> None:
    args = parse_args()

    n = args.n
    N = n**3
    periodic = not args.no_periodic

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    ell_tag = str(args.ell).replace(".", "p")
    outpath = outdir / f"jax_vs_cuda_vector_quiver_n{n}_ell{ell_tag}.png"

    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())
    print(f"Grid: n={n}, N={N}, n0={args.n0}, k={args.k}, ell={args.ell}")

    points, xs = make_grid_points(n)

    graph = gp.build_graph(points, n0=args.n0, k=args.k, cuda=True)

    cov_elem = gp.extras.make_div_free_rbf_covariance(
        ell=args.ell,
        sigma2=args.sigma2,
        periodic=periodic,
    )

    rng = jr.key(args.seed)
    xi = jr.normal(rng, (N, 3))

    print("Generating JAX field...")
    B_jax = gp.generate_vector(graph, cov_elem, xi, cuda=False)
    B_jax.block_until_ready()

    print("Generating CUDA field...")
    B_cuda = gp.generate_vector(graph, cov_elem, xi, cuda=True)
    B_cuda.block_until_ready()

    diff = B_cuda - B_jax

    max_abs = float(jnp.max(jnp.abs(diff)))
    rms_jax = float(jnp.sqrt(jnp.mean(B_jax**2)))
    rms_cuda = float(jnp.sqrt(jnp.mean(B_cuda**2)))
    rms_diff = float(jnp.sqrt(jnp.mean(diff**2)))
    rel_rms_diff = rms_diff / rms_jax

    print("B_jax shape:", B_jax.shape)
    print("B_cuda shape:", B_cuda.shape)
    print("finite CUDA:", bool(jnp.all(jnp.isfinite(B_cuda))))
    print("max abs diff:", max_abs)
    print("rms JAX:", rms_jax)
    print("rms CUDA:", rms_cuda)
    print("rms diff:", rms_diff)
    print("relative rms diff:", rel_rms_diff)

    B_jax_grid = np.asarray(B_jax).reshape((n, n, n, 3))
    B_cuda_grid = np.asarray(B_cuda).reshape((n, n, n, 3))
    diff_grid = B_cuda_grid - B_jax_grid

    zslice = n // 2
    stride = args.stride

    xplot = np.asarray(xs)[::stride]
    yplot = np.asarray(xs)[::stride]
    Xq, Yq = np.meshgrid(xplot, yplot, indexing="xy")

    U_jax, V_jax, M_jax = slice_xy(B_jax_grid, zslice, stride)
    U_cuda, V_cuda, M_cuda = slice_xy(B_cuda_grid, zslice, stride)
    U_diff, V_diff, M_diff = slice_xy(diff_grid, zslice, stride)

    vmax = max(np.nanmax(M_jax), np.nanmax(M_cuda))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    panels = [
        (axs[0], U_jax, V_jax, M_jax, "JAX vector field"),
        (axs[1], U_cuda, V_cuda, M_cuda, "CUDA vector field"),
        (axs[2], U_diff, V_diff, M_diff, "CUDA - JAX difference"),
    ]

    for ax, U, V, M, title in panels:
        is_diff = "difference" in title.lower()

        im = ax.imshow(
            M,
            origin="lower",
            extent=(0, 1, 0, 1),
            vmin=0.0,
            vmax=None if is_diff else vmax,
            alpha=0.65,
        )

        ax.quiver(
            Xq,
            Yq,
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=2.5 if is_diff else 45,
            width=0.004,
        )

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        rf"GraphGP vector field, central slice z={zslice}, "
        rf"$n={n}^3$, $\ell={args.ell}$, $k={args.k}$, $n_0={args.n0}$"
        + "\n"
        + rf"max |CUDA-JAX| = {max_abs:.3e}, relative RMS diff = {rel_rms_diff:.3e}",
        y=1.05,
    )

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    print("saved:", outpath)


if __name__ == "__main__":
    main()
