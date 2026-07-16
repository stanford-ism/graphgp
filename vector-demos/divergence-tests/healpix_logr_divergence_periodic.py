#!/usr/bin/env python3
"""
HEALPix x log-r GraphGP vector-field diagnostics with two complementary methods:

1. Real-space HEALPix/log-r divergence contribution
   Computes div B directly on the spherical grid using
       div B = r^-2 d_r(r^2 B_r) + r^-1 div_S(B_tangent),
   with the angular surface divergence estimated by local tangent-plane fits on
   the HEALPix sphere. This avoids interpreting finite-window NUFFT leakage as
   physical divergence.

2. Experimental periodic GraphGP embedding
   Shifts the spherical point cloud into a Cartesian periodic box and builds the
   GraphGP graph using the periodic minimum-image metric. The covariance is also
   made box-size-aware. This makes the prior periodic on the enclosing Cartesian
   torus, but it does *not* remove the spherical sampling/window function unless
   the full periodic box is sampled.

The script also keeps the DUCC NUFFT bottom-row spectral plots for comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

try:
    import ducc0.nufft as ducc_nufft
    HAS_DUCC = True
except Exception:
    ducc_nufft = None
    HAS_DUCC = False

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


def import_local_graphgp():
    here = Path(__file__).resolve().parent
    init_path = here / "__init__.py"
    if not init_path.exists():
        init_path = here / "graphgp" / "__init__.py"
    if init_path.exists():
        spec = importlib.util.spec_from_file_location(
            "graphgp", init_path, submodule_search_locations=[str(init_path.parent)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load graphgp from {init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["graphgp"] = module
        spec.loader.exec_module(module)
        return module
    import graphgp as gp  # type: ignore
    return gp


gp = import_local_graphgp()


# -----------------------------------------------------------------------------
# HEALPix x log-r geometry and weights
# -----------------------------------------------------------------------------


def healpix_ring_angles(nside: int):
    """Minimal HEALPix RING pixel centres, avoiding a healpy dependency."""
    if nside < 1:
        raise ValueError("nside must be >= 1")
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    nl2 = 2 * nside
    nl4 = 4 * nside
    fact1 = 2.0 / (3.0 * nside)
    fact2 = 1.0 / (3.0 * nside * nside)
    theta = np.empty(npix, dtype=np.float64)
    phi = np.empty(npix, dtype=np.float64)
    for ipix in range(npix):
        ipix1 = ipix + 1
        if ipix1 <= ncap:
            iring = int(0.5 * (1.0 + np.sqrt(2.0 * ipix1 - 1.0)))
            iphi = ipix1 - 2 * iring * (iring - 1)
            z = 1.0 - (iring * iring) * fact2
            ang = (iphi - 0.5) * np.pi / (2.0 * iring)
        elif ipix1 <= npix - ncap:
            ip = ipix1 - ncap - 1
            iring = ip // nl4 + nside
            iphi = ip % nl4 + 1
            fodd = 0.5 * (1 + ((iring + nside) & 1))
            z = (nl2 - iring) * fact1
            ang = (iphi - fodd) * np.pi / (2.0 * nside)
        else:
            ip = npix - ipix1 + 1
            iring = int(0.5 * (1.0 + np.sqrt(2.0 * ip - 1.0)))
            iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
            z = -1.0 + (iring * iring) * fact2
            ang = (iphi - 0.5) * np.pi / (2.0 * iring)
        theta[ipix] = np.arccos(np.clip(z, -1.0, 1.0))
        phi[ipix] = np.mod(ang, 2.0 * np.pi)
    return theta, phi


def spherical_basis(theta: np.ndarray, phi: np.ndarray):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    e_r = np.stack([st * cp, st * sp, ct], axis=-1)
    e_theta = np.stack([ct * cp, ct * sp, -st], axis=-1)
    e_phi = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)
    return e_r, e_theta, e_phi


def build_healpix_logr_grid(nside: int, n_shells: int, r_min: float, r_max: float) -> dict[str, Any]:
    theta, phi = healpix_ring_angles(nside)
    radii = np.geomspace(r_min, r_max, n_shells)
    e_r, e_theta, e_phi = spherical_basis(theta, phi)
    points = []
    shell_ids = []
    pixel_ids = []
    for ishell, r in enumerate(radii):
        xyz = r * e_r
        points.append(xyz)
        shell_ids.append(np.full(e_r.shape[0], ishell, dtype=np.int32))
        pixel_ids.append(np.arange(e_r.shape[0], dtype=np.int32))
    return {
        "points": np.concatenate(points, axis=0),
        "shell_ids": np.concatenate(shell_ids),
        "pixel_ids": np.concatenate(pixel_ids),
        "radii": radii,
        "theta_pix": theta,
        "phi_pix": phi,
        "unit_xyz": e_r,
        "e_r_pix": e_r,
        "e_theta_pix": e_theta,
        "e_phi_pix": e_phi,
        "npix": e_r.shape[0],
    }


def log_radial_edges_from_centres(radii: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("Need at least two radial shell centres.")
    edges = np.empty(radii.size + 1, dtype=np.float64)
    edges[1:-1] = np.sqrt(radii[:-1] * radii[1:])
    edges[0] = radii[0] ** 2 / edges[1]
    edges[-1] = radii[-1] ** 2 / edges[-2]
    return edges


def healpix_logr_volume_weights(radii: np.ndarray, npix: int, shell_ids: np.ndarray):
    redges = log_radial_edges_from_centres(radii)
    d_omega = 4.0 * np.pi / float(npix)
    shell_weights = d_omega * (redges[1:] ** 3 - redges[:-1] ** 3) / 3.0
    return shell_weights[np.asarray(shell_ids, dtype=np.int64)], shell_weights, redges


def interior_shell_mask(n_shells: int, npix: int, n_exclude: int = 1) -> np.ndarray:
    mask = np.ones((n_shells, npix), dtype=bool)
    if n_shells > 2 * n_exclude:
        mask[:n_exclude] = False
        mask[-n_exclude:] = False
    return mask


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(weights * values) / np.sum(weights))


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(weighted_mean(np.asarray(values) ** 2, weights)))


# -----------------------------------------------------------------------------
# Periodic GraphGP covariance and field generation
# -----------------------------------------------------------------------------


def make_div_free_rbf_covariance_box(*, ell: float, sigma2: float = 1.0, periodic: bool = False, boxsize=None):
    """Boxsize-aware divergence-free RBF covariance for periodic experiments."""
    ell = float(ell)
    sigma2 = float(sigma2)
    ell2 = ell * ell
    box = None if boxsize is None else jnp.asarray(boxsize, dtype=jnp.float64)

    def cov_elem(x, y):
        r = y - x
        if periodic:
            if box is None:
                r = r - jnp.round(r)
            else:
                r = r - box * jnp.round(r / box)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)
        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))

    # Metadata is harmless on the CPU/JAX path. CUDA kernels would need to be
    # extended before using non-unit box sizes there.
    cov_elem._graphgp_cuda_kernel = "div_free_rbf_vector_3"
    cov_elem._graphgp_cuda_ell = ell
    cov_elem._graphgp_cuda_sigma2 = sigma2
    cov_elem._graphgp_cuda_periodic = periodic
    cov_elem._graphgp_cuda_boxsize = None if boxsize is None else np.asarray(boxsize, dtype=float)
    return cov_elem


def graph_points_for_periodic_box(points_phys: np.ndarray, redges: np.ndarray, *, padding: float = 1e-9):
    """Shift a centred spherical cloud into a positive Cartesian box for periodic cKDTree."""
    r_outer = float(redges[-1])
    width = 2.0 * r_outer * (1.0 + padding)
    box_min = np.array([-0.5 * width, -0.5 * width, -0.5 * width], dtype=np.float64)
    box_width = np.array([width, width, width], dtype=np.float64)
    points_graph = points_phys - box_min[None, :]
    # Guard against exact upper boundary due to roundoff.
    eps = np.finfo(np.float64).eps * width * 16.0
    points_graph = np.minimum(np.maximum(points_graph, 0.0), box_width[None, :] - eps)
    return points_graph, box_min, box_width


def generate_graphgp_healpix_field(
    *,
    grid: dict[str, Any],
    redges: np.ndarray,
    n0: int,
    graph_k: int,
    ell: float,
    sigma2: float,
    seed: int,
    periodic_graph: bool = False,
    periodic_covariance: bool = False,
):
    points_phys = np.asarray(grid["points"], dtype=np.float64)
    n_points = points_phys.shape[0]
    if not (graph_k <= n0 < n_points):
        raise ValueError(f"Need graph_k <= n0 < n_points. Got graph_k={graph_k}, n0={n0}, N={n_points}.")

    if periodic_graph or periodic_covariance:
        points_graph, box_min, box_width = graph_points_for_periodic_box(points_phys, redges)
        cov = make_div_free_rbf_covariance_box(
            ell=ell, sigma2=sigma2, periodic=periodic_covariance, boxsize=box_width
        )
    else:
        points_graph = points_phys
        box_min = np.array([-redges[-1], -redges[-1], -redges[-1]], dtype=np.float64)
        box_width = np.array([2.0 * redges[-1], 2.0 * redges[-1], 2.0 * redges[-1]], dtype=np.float64)
        cov = gp.extras.make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=False)

    print(
        "Building GraphGP graph:",
        f"N={n_points}",
        f"n0={n0}",
        f"k={graph_k}",
        f"periodic_graph={periodic_graph}",
        f"periodic_covariance={periodic_covariance}",
    )
    if periodic_graph:
        graph = gp.build_graph(jnp.asarray(points_graph), n0=n0, k=graph_k, periodic=True, boxsize=box_width)
    else:
        graph = gp.build_graph(jnp.asarray(points_graph), n0=n0, k=graph_k, periodic=False)

    xi = jr.normal(jr.key(seed), (n_points, 3))
    B = np.asarray(gp.generate_vector(graph, cov, xi, fast_jit=True), dtype=np.float64)
    if not np.isfinite(B).all():
        raise RuntimeError("GraphGP generated non-finite values.")
    return B, {"points_graph": points_graph, "box_min": box_min, "box_width": box_width}


# -----------------------------------------------------------------------------
# Real-space HEALPix/log-r divergence contribution
# -----------------------------------------------------------------------------


def xyz_to_spherical_components(B_shell: np.ndarray, e_r: np.ndarray, e_theta: np.ndarray, e_phi: np.ndarray):
    Br = np.sum(B_shell * e_r[None, :, :], axis=-1)
    Btheta = np.sum(B_shell * e_theta[None, :, :], axis=-1)
    Bphi = np.sum(B_shell * e_phi[None, :, :], axis=-1)
    return Br, Btheta, Bphi


def angular_surface_divergence_from_tangent_fit(
    B_shell: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    unit_xyz: np.ndarray,
    *,
    k_neighbors: int = 8,
    reg: float = 1e-8,
):
    """Approximate div_S of tangential vector field on the unit sphere."""
    npix = unit_xyz.shape[0]
    tree = cKDTree(unit_xyz)
    dists, idx = tree.query(unit_xyz, k=k_neighbors + 1)
    dists = np.asarray(dists[:, 1:])
    idx = np.asarray(idx[:, 1:])

    Btheta = np.sum(B_shell * e_theta, axis=-1)
    Bphi = np.sum(B_shell * e_phi, axis=-1)
    F_tan_xyz = Btheta[:, None] * e_theta + Bphi[:, None] * e_phi

    surf_div = np.empty(npix, dtype=np.float64)
    cond = np.empty(npix, dtype=np.float64)
    fit_rms = np.empty(npix, dtype=np.float64)

    for i in range(npix):
        neigh = idx[i]
        et = e_theta[i]
        ep = e_phi[i]
        delta_xyz = unit_xyz[neigh] - unit_xyz[i]
        X = np.column_stack([delta_xyz @ et, delta_xyz @ ep])
        dF_xyz = F_tan_xyz[neigh] - F_tan_xyz[i]
        Y = np.column_stack([dF_xyz @ et, dF_xyz @ ep])
        scale = max(np.median(dists[i]), 1e-12)
        w = np.exp(-0.5 * (dists[i] / scale) ** 2)
        WX = X * w[:, None]
        A = X.T @ WX + reg * np.eye(2)
        C = X.T @ (Y * w[:, None])
        J = np.linalg.solve(A, C)
        resid = X @ J - Y
        surf_div[i] = np.trace(J)
        cond[i] = np.linalg.cond(A)
        fit_rms[i] = np.sqrt(np.mean(np.sum(resid**2, axis=1)))
    return {"surface_div": surf_div, "cond": cond, "fit_rms": fit_rms, "neighbor_k": k_neighbors}


def spherical_divergence_contribution(
    B: np.ndarray,
    grid: dict[str, Any],
    *,
    angular_k: int = 8,
):
    """Compute radial, angular and total divergence on HEALPix x log-r centres."""
    radii = np.asarray(grid["radii"], dtype=np.float64)
    npix = int(grid["npix"])
    n_shells = len(radii)
    B_shell = np.asarray(B, dtype=np.float64).reshape(n_shells, npix, 3)
    e_r = np.asarray(grid["e_r_pix"], dtype=np.float64)
    e_theta = np.asarray(grid["e_theta_pix"], dtype=np.float64)
    e_phi = np.asarray(grid["e_phi_pix"], dtype=np.float64)
    unit_xyz = np.asarray(grid["unit_xyz"], dtype=np.float64)

    Br, Btheta, Bphi = xyz_to_spherical_components(B_shell, e_r, e_theta, e_phi)

    radial_flux = radii[:, None] ** 2 * Br
    edge_order = 2 if n_shells >= 3 else 1
    radial_term = np.gradient(radial_flux, radii, axis=0, edge_order=edge_order) / (radii[:, None] ** 2)

    surface_div = np.empty((n_shells, npix), dtype=np.float64)
    angular_cond = np.empty((n_shells, npix), dtype=np.float64)
    angular_fit_rms = np.empty((n_shells, npix), dtype=np.float64)
    for ishell in range(n_shells):
        res = angular_surface_divergence_from_tangent_fit(
            B_shell[ishell], e_theta, e_phi, unit_xyz, k_neighbors=angular_k
        )
        surface_div[ishell] = res["surface_div"]
        angular_cond[ishell] = res["cond"]
        angular_fit_rms[ishell] = res["fit_rms"]

    angular_term = surface_div / radii[:, None]
    div = radial_term + angular_term
    return {
        "div": div,
        "radial_term": radial_term,
        "angular_term": angular_term,
        "Br": Br,
        "Btheta": Btheta,
        "Bphi": Bphi,
        "surface_div_unit_sphere": surface_div,
        "angular_cond": angular_cond,
        "angular_fit_rms": angular_fit_rms,
        "angular_k": angular_k,
    }


def local_linear_divergence(points: np.ndarray, field: np.ndarray, *, k_neighbors: int = 32, reg: float = 1e-8):
    """Cartesian point-cloud divergence cross-check via local weighted linear fits."""
    tree = cKDTree(points)
    dists, idx = tree.query(points, k=k_neighbors + 1)
    dists = np.asarray(dists[:, 1:])
    idx = np.asarray(idx[:, 1:])
    npts = points.shape[0]
    div = np.empty(npts, dtype=np.float64)
    cond = np.empty(npts, dtype=np.float64)
    fit_rms = np.empty(npts, dtype=np.float64)
    for i in range(npts):
        neigh = idx[i]
        X = points[neigh] - points[i]
        Y = field[neigh] - field[i]
        scale = max(np.median(dists[i]), 1e-12)
        w = np.exp(-0.5 * (dists[i] / scale) ** 2)
        WX = X * w[:, None]
        A = X.T @ WX + reg * np.eye(3)
        C = X.T @ (Y * w[:, None])
        J = np.linalg.solve(A, C)
        resid = X @ J - Y
        div[i] = np.trace(J)
        cond[i] = np.linalg.cond(A)
        fit_rms[i] = np.sqrt(np.mean(np.sum(resid**2, axis=1)))
    return {"div": div, "cond": cond, "fit_rms": fit_rms, "neighbor_k": k_neighbors}


def summarize_realspace_divergence(
    B: np.ndarray,
    grid: dict[str, Any],
    weights_flat: np.ndarray,
    divres: dict[str, Any],
    *,
    ell: float,
    local_div: dict[str, Any] | None = None,
):
    n_shells = len(grid["radii"])
    npix = int(grid["npix"])
    weights2d = weights_flat.reshape(n_shells, npix)
    B2 = np.sum(np.asarray(B).reshape(n_shells, npix, 3) ** 2, axis=-1)
    B_rms = float(np.sqrt(np.sum(weights2d * B2) / np.sum(weights2d)))

    interior = interior_shell_mask(n_shells, npix)
    w_int = weights2d[interior]

    def wrms_2d(arr, mask=None):
        if mask is None:
            return weighted_rms(arr, weights2d)
        return weighted_rms(arr[mask], weights2d[mask])

    div = divres["div"]
    radial = divres["radial_term"]
    angular = divres["angular_term"]

    summary = {
        "B_rms_volume_weighted": B_rms,
        "div_rms_all": wrms_2d(div),
        "div_rms_interior": wrms_2d(div, interior),
        "epsilon_rms_all": ell * wrms_2d(div) / B_rms,
        "epsilon_rms_interior": ell * wrms_2d(div, interior) / B_rms,
        "radial_term_rms_all": wrms_2d(radial),
        "angular_term_rms_all": wrms_2d(angular),
        "radial_term_rms_interior": wrms_2d(radial, interior),
        "angular_term_rms_interior": wrms_2d(angular, interior),
        "median_angular_cond": float(np.median(divres["angular_cond"])),
        "median_angular_fit_rms": float(np.median(divres["angular_fit_rms"])),
    }
    if local_div is not None:
        local2d = local_div["div"].reshape(n_shells, npix)
        summary.update({
            "local_xyz_div_rms_all": wrms_2d(local2d),
            "local_xyz_div_rms_interior": wrms_2d(local2d, interior),
            "local_xyz_epsilon_rms_all": ell * wrms_2d(local2d) / B_rms,
            "local_xyz_epsilon_rms_interior": ell * wrms_2d(local2d, interior) / B_rms,
            "median_local_xyz_cond": float(np.median(local_div["cond"])),
            "median_local_xyz_fit_rms": float(np.median(local_div["fit_rms"])),
        })
    return summary


# -----------------------------------------------------------------------------
# DUCC NUFFT and spectral diagnostics
# -----------------------------------------------------------------------------


def physical_points_to_ducc_coords(points: np.ndarray, box_min: np.ndarray, box_width: np.ndarray) -> np.ndarray:
    return np.asarray(2.0 * np.pi * (points - box_min[None, :]) / box_width[None, :], dtype=np.float64)


def ducc_type1_vector_nufft(points_phys, B_xyz, weights, *, n_modes, box_min, box_width, epsilon, nthreads):
    if not HAS_DUCC:
        raise ImportError("ducc0 is not installed in this environment.")
    points_phys = np.asarray(points_phys, dtype=np.float64)
    B_xyz = np.asarray(B_xyz, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    coords = physical_points_to_ducc_coords(points_phys, box_min, box_width)
    strengths = (weights[None, :] * B_xyz.T).astype(np.complex128, copy=False)
    out = np.zeros((3,) + tuple(n_modes), dtype=np.complex128)
    return ducc_nufft.nu2u(
        points=strengths,
        coord=coords,
        forward=True,
        epsilon=float(epsilon),
        nthreads=int(nthreads),
        out=out,
        fft_order=True,
        periodicity=2.0 * np.pi,
    )


def k_axes_from_box(n_modes: tuple[int, int, int], box_width: np.ndarray):
    kx = 2.0 * np.pi * np.fft.fftfreq(n_modes[0], d=box_width[0] / n_modes[0])
    ky = 2.0 * np.pi * np.fft.fftfreq(n_modes[1], d=box_width[1] / n_modes[1])
    kz = 2.0 * np.pi * np.fft.fftfreq(n_modes[2], d=box_width[2] / n_modes[2])
    return kx, ky, kz


def longitudinal_fraction_from_bhat(Bhat: np.ndarray, box_width: np.ndarray, *, nbins: int = 16):
    kx, ky, kz = k_axes_from_box(Bhat.shape[1:], box_width)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    kmag = np.sqrt(k2)
    Bxhat, Byhat, Bzhat = Bhat[0], Bhat[1], Bhat[2]
    kdot = KX * Bxhat + KY * Byhat + KZ * Bzhat
    etot = (np.abs(Bxhat) ** 2 + np.abs(Byhat) ** 2 + np.abs(Bzhat) ** 2).real
    elong = np.zeros_like(etot)
    nonzero = k2 > 0.0
    elong[nonzero] = (np.abs(kdot[nonzero]) ** 2 / k2[nonzero]).real
    valid = nonzero & (etot > 0.0)
    frac = np.zeros_like(etot)
    frac[valid] = elong[valid] / etot[valid]

    kz_idx = int(np.argmin(np.abs(kz)))
    frac2d = np.fft.fftshift(frac[:, :, kz_idx])
    frac_flat = frac[valid]
    kbins = np.linspace(0.0, kmag[nonzero].max() * 1.0001, nbins + 1)
    shell_k = 0.5 * (kbins[:-1] + kbins[1:])
    shell_frac = np.full(nbins, np.nan)
    shell_elong = np.full(nbins, np.nan)
    shell_etot = np.full(nbins, np.nan)
    shell_counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        mask = nonzero & (kmag >= kbins[i]) & (kmag < kbins[i + 1])
        shell_counts[i] = int(mask.sum())
        if mask.any():
            shell_elong[i] = float(np.sum(elong[mask]).real)
            shell_etot[i] = float(np.sum(etot[mask]).real)
            if shell_etot[i] > 0.0:
                shell_frac[i] = shell_elong[i] / shell_etot[i]
    total_elong = float(np.sum(elong[nonzero]).real)
    total_etot = float(np.sum(etot[nonzero]).real)
    total_frac = total_elong / total_etot if total_etot > 0.0 else np.nan
    return {
        "kx_shift": np.fft.fftshift(kx),
        "ky_shift": np.fft.fftshift(ky),
        "frac2d": frac2d,
        "frac_flat": frac_flat,
        "shell_k": shell_k,
        "shell_frac": shell_frac,
        "shell_elong": shell_elong,
        "shell_etot": shell_etot,
        "shell_counts": shell_counts,
        "total_frac": total_frac,
        "total_elong": total_elong,
        "total_etot": total_etot,
    }


def nufft_window_baseline(points_phys, weights, *, n_modes, box_min, box_width, epsilon, nthreads, nbins):
    """Constant-field baseline. A finite spherical window gives about 1/3 even for div-free constants."""
    vals = []
    for comp in range(3):
        Bc = np.zeros((points_phys.shape[0], 3), dtype=np.float64)
        Bc[:, comp] = 1.0
        Bhat = ducc_type1_vector_nufft(
            points_phys, Bc, weights, n_modes=n_modes, box_min=box_min,
            box_width=box_width, epsilon=epsilon, nthreads=nthreads,
        )
        vals.append(longitudinal_fraction_from_bhat(Bhat, box_width, nbins=nbins)["total_frac"])
    return float(np.mean(vals)), np.asarray(vals)


# -----------------------------------------------------------------------------
# Plots and summaries
# -----------------------------------------------------------------------------


def figure_title(fig, title, subtitle):
    fig.text(0.06, 0.985, title, ha="left", va="top", fontsize=15.5, fontweight="bold")
    fig.text(0.06, 0.940, subtitle, ha="left", va="top", fontsize=11.2)


def plot_realspace_divergence_figure(grid, divres, summary, weights_flat, *, ell: float, prefix: str, label: str):
    radii = np.asarray(grid["radii"])
    npix = int(grid["npix"])
    n_shells = len(radii)
    weights2d = weights_flat.reshape(n_shells, npix)
    mid_shell = n_shells // 2
    eps = ell * divres["div"] / summary["B_rms_volume_weighted"]
    eps_rad = ell * divres["radial_term"] / summary["B_rms_volume_weighted"]
    eps_ang = ell * divres["angular_term"] / summary["B_rms_volume_weighted"]
    interior = interior_shell_mask(n_shells, npix)

    def shell_wrms(arr):
        out = np.empty(n_shells, dtype=np.float64)
        for i in range(n_shells):
            out[i] = weighted_rms(arr[i], weights2d[i])
        return out

    phi_deg = np.degrees(grid["phi_pix"])
    theta_deg = np.degrees(grid["theta_pix"])

    fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.9))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.78, wspace=0.28)
    figure_title(
        fig,
        f"HEALPix x log-r real-space divergence contribution: {label}",
        r"Using $\nabla\!\cdot B=r^{-2}\partial_r(r^2B_r)+r^{-1}\nabla_S\!\cdot B_\perp$; angular derivative from local tangent-plane fits.",
    )

    vmax = np.nanpercentile(np.abs(eps[mid_shell]), 99)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    sc = axs[0].scatter(phi_deg, theta_deg, c=eps[mid_shell], s=26, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axs[0].set_title(rf"Mid shell $\,\epsilon=\ell(\nabla\!\cdot B)/B_{{\rm rms}}$")
    axs[0].set_xlabel(r"$\phi$ [deg]")
    axs[0].set_ylabel(r"$\theta$ [deg]")
    cb = fig.colorbar(sc, ax=axs[0], fraction=0.046, pad=0.03)
    cb.set_label(r"$\epsilon$")

    axs[1].plot(radii, shell_wrms(eps), "o-", lw=1.9, ms=4.0, label="total")
    axs[1].plot(radii, shell_wrms(eps_rad), "--", lw=1.8, label="radial term")
    axs[1].plot(radii, shell_wrms(eps_ang), ":", lw=2.2, label="angular term")
    axs[1].set_xscale("log")
    axs[1].set_title("Volume-weighted shell RMS")
    axs[1].set_xlabel(r"radius $r$")
    axs[1].set_ylabel(r"RMS $\,\epsilon$")
    axs[1].legend(loc="best")

    all_vals = eps.ravel()
    int_vals = eps[interior].ravel()
    lo = np.nanpercentile(all_vals, 0.5)
    hi = np.nanpercentile(all_vals, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0
    bins = np.linspace(lo, hi, 80)
    axs[2].hist(all_vals, bins=bins, density=True, histtype="step", linewidth=2.0, label="all shells")
    axs[2].hist(int_vals, bins=bins, density=True, histtype="step", linewidth=2.0, label="radial boundary shells removed")
    axs[2].axvline(0.0, ls="--", lw=1.2, color="0.45")
    axs[2].set_title("Distribution of real-space divergence")
    axs[2].set_xlabel(r"$\epsilon$")
    axs[2].set_ylabel("normalised density")
    axs[2].legend(loc="best")

    out = OUTDIR / f"{prefix}_{label}_healpix_realspace_divergence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Completed", out)
    return out


def plot_nufft_spectral_bottom_row(diag: dict[str, Any], *, prefix: str, label: str):
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.8))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.78, wspace=0.28)
    figure_title(
        fig,
        f"DUCC NUFFT windowed spectral diagnostic: {label}",
        "This is the longitudinal fraction of the finite HEALPix/log-r windowed field, not a pure physical divergence fraction.",
    )
    frac2d = diag["frac2d"]
    vmaxf = np.nanpercentile(frac2d, 99.5)
    if not np.isfinite(vmaxf) or vmaxf <= 0:
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
    axs[0].set_title(r"Modewise longitudinal fraction $f_L(k_x,k_y,k_z{=}0)$")
    axs[0].set_xlabel(r"$k_x$")
    axs[0].set_ylabel(r"$k_y$")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03, label=r"$f_L(\mathbf{k})$")

    hfrac = axs[1].plot(diag["shell_k"], diag["shell_frac"], "o-", lw=1.8, ms=4.0, color="tab:blue", label=r"$E_L/E_{\rm tot}$")
    htot = axs[1].axhline(diag["total_frac"], ls="--", lw=1.2, color="0.4", label=fr"total = {diag['total_frac']:.3f}")
    axs[1].set_title("Longitudinal fraction by shell, with shell powers")
    axs[1].set_xlabel(r"$|k|$")
    axs[1].set_ylabel(r"$E_L/E_{\rm tot}$")
    axpow = axs[1].twinx()
    hel = axpow.plot(diag["shell_k"], diag["shell_elong"], "--", color="tab:red", lw=1.8, alpha=0.75, label=r"$E_L$")
    het = axpow.plot(diag["shell_k"], diag["shell_etot"], ":", color="tab:green", lw=2.0, alpha=0.90, label=r"$E_{\rm tot}$")
    axpow.set_ylabel("shell power")
    axpow.set_yscale("log")
    handles = hfrac + hel + het + [htot]
    axs[1].legend(handles, [h.get_label() for h in handles], loc="upper right")

    frac_flat = diag["frac_flat"]
    high = np.nanpercentile(frac_flat, 99.8) if frac_flat.size else 1.0
    if not np.isfinite(high) or high <= 0:
        high = 1.0
    binsf = np.linspace(0.0, high, 70)
    axs[2].hist(frac_flat, bins=binsf, density=True, histtype="step", linewidth=2.0, color="tab:red")
    axs[2].axvline(diag["total_frac"], ls="--", lw=1.2, color="0.45", label=fr"total = {diag['total_frac']:.3f}")
    axs[2].set_title("Distribution of modewise longitudinal fraction")
    axs[2].set_xlabel(r"$f_L(\mathbf{k})$")
    axs[2].set_ylabel("normalised density")
    axs[2].legend(loc="best")

    out = OUTDIR / f"{prefix}_{label}_nufft_spectral_bottom_row.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Completed", out)
    return out


def save_text_summary(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n")
    print("Wrote", path)


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


def run_case(args, *, periodic_graph: bool, periodic_covariance: bool, label: str, grid, weights_flat, shell_weights, redges):
    B, boxinfo = generate_graphgp_healpix_field(
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
    divres = spherical_divergence_contribution(B, grid, angular_k=args.angular_k)
    local = None
    if args.local_xyz:
        local = local_linear_divergence(grid["points"], B, k_neighbors=args.local_k)
    summary = summarize_realspace_divergence(B, grid, weights_flat, divres, ell=args.ell, local_div=local)
    realfig = plot_realspace_divergence_figure(grid, divres, summary, weights_flat, ell=args.ell, prefix=args.prefix, label=label)

    nufft_fig = None
    nufft_diag = None
    baseline_mean = np.nan
    baseline_xyz = np.full(3, np.nan)
    if not args.skip_nufft:
        Bhat = ducc_type1_vector_nufft(
            grid["points"],
            B,
            weights_flat,
            n_modes=args.n_modes,
            box_min=boxinfo["box_min"],
            box_width=boxinfo["box_width"],
            epsilon=args.epsilon,
            nthreads=args.nthreads,
        )
        nufft_diag = longitudinal_fraction_from_bhat(Bhat, boxinfo["box_width"], nbins=args.nbins)
        nufft_fig = plot_nufft_spectral_bottom_row(nufft_diag, prefix=args.prefix, label=label)
        if args.window_baseline:
            baseline_mean, baseline_xyz = nufft_window_baseline(
                grid["points"],
                weights_flat,
                n_modes=args.n_modes,
                box_min=boxinfo["box_min"],
                box_width=boxinfo["box_width"],
                epsilon=args.epsilon,
                nthreads=args.nthreads,
                nbins=args.nbins,
            )

    np.savez_compressed(
        OUTDIR / f"{args.prefix}_{label}_diagnostics.npz",
        B=B,
        points=grid["points"],
        radii=grid["radii"],
        weights=weights_flat,
        div=divres["div"],
        radial_term=divres["radial_term"],
        angular_term=divres["angular_term"],
        epsilon=args.ell * divres["div"] / summary["B_rms_volume_weighted"],
        shell_k=np.array([]) if nufft_diag is None else nufft_diag["shell_k"],
        shell_frac=np.array([]) if nufft_diag is None else nufft_diag["shell_frac"],
        total_nufft_frac=np.array(np.nan if nufft_diag is None else nufft_diag["total_frac"]),
        constant_window_baseline_mean=np.array(baseline_mean),
        constant_window_baseline_xyz=baseline_xyz,
    )

    lines = [
        f"Case: {label}",
        f"periodic_graph = {periodic_graph}",
        f"periodic_covariance = {periodic_covariance}",
        f"realspace_figure = {realfig}",
        f"nufft_figure = {nufft_fig}",
        "",
        "Real-space HEALPix/log-r divergence contribution:",
    ]
    for key, value in summary.items():
        lines.append(f"  {key}: {value}")
    if nufft_diag is not None:
        lines.extend([
            "",
            "DUCC NUFFT windowed spectral diagnostic:",
            f"  total_frac: {nufft_diag['total_frac']}",
            f"  total_elong: {nufft_diag['total_elong']}",
            f"  total_etot: {nufft_diag['total_etot']}",
            f"  constant_field_window_baseline_mean: {baseline_mean}",
            f"  constant_field_window_baseline_xyz: {baseline_xyz.tolist()}",
            f"  excess_over_constant_window_baseline: {nufft_diag['total_frac'] - baseline_mean if np.isfinite(baseline_mean) else np.nan}",
        ])
    lines.extend([
        "",
        "Interpretation notes:",
        "  - The real-space divergence diagnostic is the recommended HEALPix/log-r check.",
        "  - The NUFFT diagnostic is still windowed by the finite spherical domain.",
        "  - A periodic graph/covariance makes the prior periodic on the enclosing Cartesian torus,",
        "    but it does not remove the spherical sampling mask/window leakage by itself.",
    ])
    summary_path = OUTDIR / f"{args.prefix}_{label}_summary.txt"
    save_text_summary(summary_path, lines)
    return {"label": label, "summary": summary, "nufft_diag": nufft_diag, "baseline": baseline_mean}


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
    parser.add_argument("--angular-k", type=int, default=8)
    parser.add_argument("--local-xyz", action="store_true", help="also compute slower local Cartesian kNN divergence cross-check")
    parser.add_argument("--local-k", type=int, default=32)
    parser.add_argument("--n-modes", type=positive_int_tuple3, default=(32, 32, 32))
    parser.add_argument("--nbins", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--nthreads", type=int, default=default_nthreads())
    parser.add_argument("--skip-nufft", action="store_true")
    parser.add_argument("--window-baseline", action="store_true", help="compute constant-field NUFFT leakage baseline")
    parser.add_argument("--periodic-case", action="store_true", help="also run experimental Cartesian-torus GraphGP embedding")
    parser.add_argument("--prefix", type=str, default="healpix_logr_methods")
    args = parser.parse_args()

    grid = build_healpix_logr_grid(nside=args.nside, n_shells=args.n_shells, r_min=args.r_min, r_max=args.r_max)
    weights_flat, shell_weights, redges = healpix_logr_volume_weights(grid["radii"], int(grid["npix"]), grid["shell_ids"])

    print("Geometry:", f"nside={args.nside}", f"npix={grid['npix']}", f"n_shells={args.n_shells}", f"N={grid['points'].shape[0]}")
    print("Volume:", f"r_edges=[{redges[0]:.4g}, {redges[-1]:.4g}]", f"sum(weights)={weights_flat.sum():.6g}")

    cases = []
    cases.append(run_case(
        args,
        periodic_graph=False,
        periodic_covariance=False,
        label="nonperiodic",
        grid=grid,
        weights_flat=weights_flat,
        shell_weights=shell_weights,
        redges=redges,
    ))
    if args.periodic_case:
        cases.append(run_case(
            args,
            periodic_graph=True,
            periodic_covariance=True,
            label="periodic_torus_embedding",
            grid=grid,
            weights_flat=weights_flat,
            shell_weights=shell_weights,
            redges=redges,
        ))

    comparison_lines = [
        "HEALPix/log-r divergence and periodic-graph method comparison",
        "=============================================================",
        f"nside = {args.nside}",
        f"n_shells = {args.n_shells}",
        f"N_total = {grid['points'].shape[0]}",
        f"n0 = {args.n0}",
        f"graph_k = {args.graph_k}",
        f"ell = {args.ell}",
        f"sigma2 = {args.sigma2}",
        "",
    ]
    for case in cases:
        s = case["summary"]
        comparison_lines.append(f"[{case['label']}]")
        comparison_lines.append(f"  epsilon_rms_all = {s['epsilon_rms_all']}")
        comparison_lines.append(f"  epsilon_rms_interior = {s['epsilon_rms_interior']}")
        comparison_lines.append(f"  div_rms_all = {s['div_rms_all']}")
        comparison_lines.append(f"  div_rms_interior = {s['div_rms_interior']}")
        if case["nufft_diag"] is not None:
            comparison_lines.append(f"  nufft_total_frac_windowed = {case['nufft_diag']['total_frac']}")
            comparison_lines.append(f"  constant_window_baseline = {case['baseline']}")
        comparison_lines.append("")
    save_text_summary(OUTDIR / f"{args.prefix}_comparison_summary.txt", comparison_lines)
    print("Done. Outputs are in", OUTDIR.resolve())


if __name__ == "__main__":
    main()
