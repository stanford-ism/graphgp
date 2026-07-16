import importlib.util
import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

OUTDIR = Path('temp-figures')
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'figure.dpi': 150,
    'savefig.dpi': 240,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


# ------------------------------
# GraphGP import helper
# ------------------------------

def import_local_graphgp():
    spec = importlib.util.spec_from_file_location(
        'graphgp', 'graphgp/__init__.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['graphgp'] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gp = import_local_graphgp()


# ------------------------------
# Generic helpers
# ------------------------------

def figure_title(fig, title, eq):
    fig.text(0.06, 0.985, title, ha='left', va='top', fontsize=15.5, fontweight='bold')
    fig.text(0.06, 0.952, eq, ha='left', va='top', fontsize=11.2)


def make_points_2d(n):
    xs = (jnp.arange(n) + 0.5) / n
    X, Y = jnp.meshgrid(xs, xs, indexing='ij')
    return jnp.stack([X, Y], axis=-1).reshape((-1, 2))


def make_points_3d(n):
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing='ij')
    return jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3)), np.asarray(xs)


def min_image_delta(a, b, box=1.0):
    d = np.asarray(a) - np.asarray(b)
    return d - box * np.round(d / box)




def predecessor_neighbors(points_tree, n0, k, periodic=False, boxsize=1.0):
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
        idx = idx[np.argsort(dist[idx], kind='stable')]
        neigh[i - n0] = idx
    return jnp.asarray(neigh)

def build_graph_local(points, *, n0, k, periodic=False, boxsize=1.0):
    points_tree, split_dims, indices = gp.build_tree(points)
    neighbors = predecessor_neighbors(points_tree, n0=n0, k=k, periodic=periodic, boxsize=boxsize)
    depths = gp.compute_depths(neighbors, n0=n0)
    pts_ord, idx_ord, neigh_ord, depths = gp.order_by_depth(points_tree, indices, neighbors, depths)
    offsets = jnp.searchsorted(depths, jnp.arange(1, jnp.max(depths) + 2))
    offsets = tuple(int(x) for x in offsets)
    graph = gp.Graph(pts_ord, neigh_ord, offsets, idx_ord)
    return graph, np.asarray(points_tree), np.asarray(split_dims), np.asarray(indices), np.asarray(neighbors)


def boundary_distance_2d(pts):
    return np.min(np.stack([pts[:, 0], pts[:, 1], 1 - pts[:, 0], 1 - pts[:, 1]], axis=1), axis=1)


def boundary_distance_3d(pts):
    return np.min(np.stack([pts[:, 0], pts[:, 1], pts[:, 2], 1 - pts[:, 0], 1 - pts[:, 1], 1 - pts[:, 2]], axis=1), axis=1)


def choose_boundary_queries(pts, n0, nshow=4, threshold=0.16):
    bd = boundary_distance_2d(pts)
    cand = np.where((np.arange(len(pts)) >= n0) & (bd < threshold))[0]
    score = pts[cand, 1] + 0.37 * pts[cand, 0]
    cand = cand[np.argsort(score)]
    if len(cand) == 0:
        return np.array([], dtype=int)
    pick = np.linspace(0, len(cand) - 1, min(nshow, len(cand))).astype(int)
    return cand[pick]


def choose_boundary_queries_3d(pts, n0, nshow=3, threshold=0.22):
    bd = boundary_distance_3d(pts)
    cand = np.where((np.arange(len(pts)) >= n0) & (bd < threshold))[0]
    score = pts[cand, 2] + 0.31 * pts[cand, 1] + 0.17 * pts[cand, 0]
    cand = cand[np.argsort(score)]
    if len(cand) == 0:
        return np.array([], dtype=int)
    pick = np.linspace(0, len(cand) - 1, min(nshow, len(cand))).astype(int)
    return cand[pick]


# ------------------------------
# Field diagnostics
# ------------------------------

def divergence_fd_periodic(Bgrid, spacing):
    dx, dy, dz = spacing
    return (
        (np.roll(Bgrid[..., 0], -1, axis=0) - np.roll(Bgrid[..., 0], 1, axis=0)) / (2 * dx)
        + (np.roll(Bgrid[..., 1], -1, axis=1) - np.roll(Bgrid[..., 1], 1, axis=1)) / (2 * dy)
        + (np.roll(Bgrid[..., 2], -1, axis=2) - np.roll(Bgrid[..., 2], 1, axis=2)) / (2 * dz)
    )


def divergence_fft(Bgrid, spacing):
    nx, ny, nz = Bgrid.shape[:3]
    dx, dy, dz = spacing
    Bhat = np.fft.fftn(Bgrid, axes=(0, 1, 2))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    divhat = 1j * (KX * Bhat[..., 0] + KY * Bhat[..., 1] + KZ * Bhat[..., 2])
    div = np.fft.ifftn(divhat, axes=(0, 1, 2)).real
    return div, (kx, ky, kz, KX, KY, KZ, Bhat)


def longitudinal_fraction_spectrum(Bgrid, spacing, nbins=14):
    _, (kx, ky, kz, KX, KY, KZ, Bhat) = divergence_fft(Bgrid, spacing)
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    mask = k2 > 0
    kdot = KX * Bhat[..., 0] + KY * Bhat[..., 1] + KZ * Bhat[..., 2]
    elong = np.abs(kdot) ** 2 / np.where(mask, k2, 1.0)
    etot = np.abs(Bhat[..., 0]) ** 2 + np.abs(Bhat[..., 1]) ** 2 + np.abs(Bhat[..., 2]) ** 2
    kmag = np.sqrt(k2)
    kbins = np.linspace(0.0, kmag[mask].max() * 1.0001, nbins + 1)
    shell_mid = 0.5 * (kbins[:-1] + kbins[1:])
    shell_frac = np.full(nbins, np.nan)
    shell_counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        m = mask & (kmag >= kbins[i]) & (kmag < kbins[i + 1])
        shell_counts[i] = int(m.sum())
        if m.any():
            shell_frac[i] = elong[m].sum() / etot[m].sum()
    total_frac = elong[mask].sum() / etot[mask].sum()
    return shell_mid, shell_frac, total_frac, kmag, (kx, ky, kz), kbins, shell_counts


def modewise_longitudinal_fraction(Bgrid, spacing):
    _, (kx, ky, kz, KX, KY, KZ, Bhat) = divergence_fft(Bgrid, spacing)
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    etot = np.abs(Bhat[..., 0]) ** 2 + np.abs(Bhat[..., 1]) ** 2 + np.abs(Bhat[..., 2]) ** 2
    kdot = KX * Bhat[..., 0] + KY * Bhat[..., 1] + KZ * Bhat[..., 2]
    frac = np.zeros_like(etot.real)
    mask = (k2 > 0) & (etot > 0)
    frac[mask] = (np.abs(kdot[mask]) ** 2 / k2[mask]) / etot[mask]

    kz_idx = int(np.argmin(np.abs(kz)))
    frac2d = np.fft.fftshift(frac[:, :, kz_idx])
    kx_shift = np.fft.fftshift(kx)
    ky_shift = np.fft.fftshift(ky)
    frac_flat = frac[mask]
    return frac2d, kx_shift, ky_shift, frac_flat


def seam_ratio(Bgrid):
    ratios = []
    for ax in range(3):
        seam = Bgrid.take(indices=0, axis=ax) - Bgrid.take(indices=Bgrid.shape[ax] - 1, axis=ax)
        seam_rms = np.sqrt(np.mean(np.sum(seam ** 2, axis=-1)))
        diffs = np.diff(Bgrid, axis=ax)
        interior_rms = np.sqrt(np.mean(np.sum(diffs ** 2, axis=-1)))
        ratios.append(float(seam_rms / interior_rms))
    return tuple(ratios)


def radial_shell_profile(values, coords_1d, nbins=16):
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    rr = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    values = np.asarray(values)
    bins = np.linspace(0.0, rr.max() * 1.0001, nbins + 1)
    which = np.digitize(rr.ravel(), bins) - 1
    mean = np.full(nbins, np.nan)
    std = np.full(nbins, np.nan)
    rms = np.full(nbins, np.nan)
    stderr = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    flat = values.ravel()
    for b in range(nbins):
        mask = which == b
        counts[b] = int(mask.sum())
        if counts[b] == 0:
            continue
        vals = flat[mask]
        mean[b] = vals.mean()
        std[b] = vals.std(ddof=0)
        rms[b] = np.sqrt(np.mean(vals ** 2))
        stderr[b] = std[b] / np.sqrt(counts[b])
    return {
        'rcent': 0.5 * (bins[:-1] + bins[1:]),
        'mean': mean,
        'std': std,
        'rms': rms,
        'stderr': stderr,
        'outer_removed_mask': rr < bins[-2],
    }


def correlation_tensor_axis_lags(Bgrid, max_lag, periodic=True):
    lags = np.arange(0, max_lag + 1)
    Ct = np.zeros((3, 3, 3, len(lags)), dtype=np.float64)
    for a in range(3):
        for ih, h in enumerate(lags):
            if periodic:
                C = np.roll(Bgrid, -int(h), axis=a)
                A = Bgrid
            else:
                sl0 = [slice(None)] * 4
                sl1 = [slice(None)] * 4
                sl0[a] = slice(0, Bgrid.shape[a] - h)
                sl1[a] = slice(h, Bgrid.shape[a])
                A = Bgrid[tuple(sl0)]
                C = Bgrid[tuple(sl1)]
            for i in range(3):
                for j in range(3):
                    Ct[a, i, j, ih] = np.mean(A[..., i] * C[..., j])
    return lags, Ct


def summarize_correlations(Bgrid, ell, periodic=True):
    n = Bgrid.shape[0]
    max_lag = min(n // 2, max(6, int(np.ceil(9.0 * ell * n))))
    lags, Ct = correlation_tensor_axis_lags(Bgrid, max_lag, periodic=periodic)
    r = lags / n
    cpar_axes = np.stack([Ct[a, a, a, :] for a in range(3)], axis=0)
    cperp_axes = np.stack([
        0.5 * (Ct[a, (a + 1) % 3, (a + 1) % 3, :] + Ct[a, (a + 2) % 3, (a + 2) % 3, :])
        for a in range(3)
    ], axis=0)

    def sym_pair(i, j):
        arr = np.stack([0.5 * (Ct[a, i, j, :] + Ct[a, j, i, :]) for a in range(3)], axis=0)
        return arr.mean(axis=0), arr.std(axis=0)

    cxy_mean, cxy_std = sym_pair(0, 1)
    cxz_mean, cxz_std = sym_pair(0, 2)
    cyz_mean, cyz_std = sym_pair(1, 2)
    return {
        'r': r,
        'cpar_mean': cpar_axes.mean(axis=0),
        'cpar_std': cpar_axes.std(axis=0),
        'cperp_mean': cperp_axes.mean(axis=0),
        'cperp_std': cperp_axes.std(axis=0),
        'cxy_mean': cxy_mean,
        'cxy_std': cxy_std,
        'cxz_mean': cxz_mean,
        'cxz_std': cxz_std,
        'cyz_mean': cyz_mean,
        'cyz_std': cyz_std,
    }


def theory_parallel_perp_trace(r, ell, sigma2):
    q = (r / ell) ** 2
    cpar = sigma2 * np.exp(-0.5 * q)
    cperp = sigma2 * (1.0 - 0.5 * q) * np.exp(-0.5 * q)
    ctrace = cpar + 2.0 * cperp
    return cpar, cperp, ctrace


def plot_corr_panel(axs, corr, ell, sigma2, subtitle):
    r = corr['r']
    x = r / ell
    cpar_th, cperp_th, ctrace_th = theory_parallel_perp_trace(r, ell, sigma2)
    ctrace = corr['cpar_mean'] + 2.0 * corr['cperp_mean']
    ctrace_std = np.sqrt(corr['cpar_std'] ** 2 + 4.0 * corr['cperp_std'] ** 2)

    top = [
        (axs[0, 0], corr['cpar_mean'] / sigma2, corr['cpar_std'] / sigma2, cpar_th / sigma2, r'$C_{\parallel}(r)/\sigma^2$', None),
        (axs[0, 1], corr['cperp_mean'] / sigma2, corr['cperp_std'] / sigma2, cperp_th / sigma2, r'$C_{\perp}(r)/\sigma^2$', np.sqrt(2.0)),
        (axs[0, 2], ctrace / sigma2, ctrace_std / sigma2, ctrace_th / sigma2, r'${\rm tr}\,C(r)/\sigma^2$', None),
    ]
    legend_handles = None
    for ax, mean, std, theory, title, xline in top:
        h1 = ax.plot(x, mean, 'o', ms=4.0, color='tab:blue', label='sample mean')
        h2 = ax.fill_between(x, mean - std, mean + std, alpha=0.20, color='tab:blue', label='axis scatter')
        h3 = ax.plot(x, theory, lw=2.0, color='tab:orange', label='theory')
        ax.axhline(0.0, ls='--', lw=1.0, color='0.4')
        if xline is not None:
            ax.axvline(xline, ls=':', lw=1.2, color='0.35')
            ax.text(xline + 0.10, 0.05, r'$r/\ell=\sqrt{2}$', fontsize=9)
        ax.set_title(title)
        if legend_handles is None:
            legend_handles = [h1[0], h2, h3[0]]

    bottom = [
        (axs[1, 0], corr['cxy_mean'] / sigma2, corr['cxy_std'] / sigma2, r'$C_{xy}(r)/\sigma^2$'),
        (axs[1, 1], corr['cxz_mean'] / sigma2, corr['cxz_std'] / sigma2, r'$C_{xz}(r)/\sigma^2$'),
        (axs[1, 2], corr['cyz_mean'] / sigma2, corr['cyz_std'] / sigma2, r'$C_{yz}(r)/\sigma^2$'),
    ]
    for ax, mean, std, title in bottom:
        ax.plot(x, mean, 'o', ms=4.0, color='tab:blue')
        ax.fill_between(x, mean - std, mean + std, alpha=0.20, color='tab:blue')
        ax.axhline(0.0, color='tab:orange', lw=2.0)
        ax.set_title(title)
        ax.set_xlabel(r'$r/\ell$')

    for ax in axs[0, :]:
        ax.set_xlabel(r'$r/\ell$')
    axs[0, 0].text(-0.18, 1.18, subtitle, transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold', va='top')
    return legend_handles


# ------------------------------
# Plotting
# ------------------------------

def figure_predecessor_search_2d(n=10, n0=8, k=4):
    points = make_points_2d(n)
    _, pts_tree, _, _, neigh_std = build_graph_local(points, n0=n0, k=k, periodic=False)
    _, _, _, _, neigh_per = build_graph_local(points, n0=n0, k=k, periodic=True)
    qidx = choose_boundary_queries(pts_tree, n0=n0, nshow=4, threshold=0.16)

    fig, axs = plt.subplots(1, 2, figsize=(13.6, 6.2), constrained_layout=True)
    for ax, neigh, title, color, periodic in [
        (axs[0], neigh_std, 'Standard GraphGP predecessor search', 'tab:blue', False),
        (axs[1], neigh_per, 'Periodic GraphGP predecessor search', 'tab:green', True),
    ]:
        ax.scatter(pts_tree[:n0, 0], pts_tree[:n0, 1], c='k', s=28, label='initial points')
        ax.scatter(pts_tree[n0:, 0], pts_tree[n0:, 1], c='0.7', s=18, label='refined points')
        for idx in qidx:
            q = pts_tree[idx]
            ax.scatter([q[0]], [q[1]], s=100, marker='*', c='tab:red', zorder=5)
            for nbr in neigh[idx - n0]:
                p = pts_tree[nbr]
                if periodic:
                    pplot = q + min_image_delta(p, q, box=1.0)
                    ax.plot([q[0], pplot[0]], [q[1], pplot[1]], color=color, lw=2.0, alpha=0.9)
                    if not (0 <= pplot[0] <= 1 and 0 <= pplot[1] <= 1):
                        ax.scatter([pplot[0]], [pplot[1]], marker='x', s=54, c=color, zorder=4)
                else:
                    ax.plot([q[0], p[0]], [q[1], p[1]], color=color, lw=2.0, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        if periodic:
            ax.set_xlim(-0.16, 1.16)
            ax.set_ylim(-0.16, 1.16)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    axs[0].legend(loc='upper right')
    fig.savefig(OUTDIR / 'figure_predecessor_search_2d.png', bbox_inches='tight')
    plt.close(fig)
    print('Completed figure_predecessor_search_2d.png')


def figure_predecessor_search_3d(n=6, n0=16, k=8):
    points, _ = make_points_3d(n)
    _, pts_tree, _, _, neigh_std = build_graph_local(points, n0=n0, k=k, periodic=False)
    _, _, _, _, neigh_per = build_graph_local(points, n0=n0, k=k, periodic=True)
    qidx = choose_boundary_queries_3d(pts_tree, n0=n0, nshow=3, threshold=0.22)

    fig = plt.figure(figsize=(13.8, 6.4), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    for ax, neigh, title, color, periodic in [
        (ax0, neigh_std, 'Standard GraphGP predecessor search', 'tab:blue', False),
        (ax1, neigh_per, 'Periodic GraphGP predecessor search', 'tab:green', True),
    ]:
        ax.scatter(pts_tree[:n0, 0], pts_tree[:n0, 1], pts_tree[:n0, 2], c='k', s=20, depthshade=False)
        ax.scatter(pts_tree[n0:, 0], pts_tree[n0:, 1], pts_tree[n0:, 2], c='0.7', s=10, depthshade=False)
        for idx in qidx:
            q = pts_tree[idx]
            ax.scatter([q[0]], [q[1]], [q[2]], s=110, marker='*', c='tab:red', depthshade=False)
            for nbr in neigh[idx - n0]:
                p = pts_tree[nbr]
                if periodic:
                    pplot = q + min_image_delta(p, q, box=1.0)
                    ax.plot([q[0], pplot[0]], [q[1], pplot[1]], [q[2], pplot[2]], color=color, lw=1.8, alpha=0.9)
                    if not np.all((0 <= pplot) & (pplot <= 1)):
                        ax.scatter([pplot[0]], [pplot[1]], [pplot[2]], marker='x', s=46, c=color, depthshade=False)
                else:
                    ax.plot([q[0], p[0]], [q[1], p[1]], [q[2], p[2]], color=color, lw=1.8, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if periodic:
            ax.set_xlim(-0.16, 1.16)
            ax.set_ylim(-0.16, 1.16)
            ax.set_zlim(-0.16, 1.16)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-58)
    fig.savefig(OUTDIR / 'figure_predecessor_search_3d.png', bbox_inches='tight')
    plt.close(fig)
    print('Completed figure_predecessor_search_3d.png')


def figure_field_periodicity_checks(Bstd, Bper, df):
    n = Bstd.shape[0]
    zmid = n // 2
    fig = plt.figure(figsize=(14.8, 7.8))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    vmax = np.max(np.abs(np.concatenate([Bstd[:, :, zmid, 2].ravel(), Bper[:, :, zmid, 2].ravel()])))
    diff = Bper[:, :, zmid, 2] - Bstd[:, :, zmid, 2]
    vmaxd = np.max(np.abs(diff))
    im0 = ax0.imshow(Bstd[:, :, zmid, 2].T, origin='lower', extent=[0, 1, 0, 1], vmin=-vmax, vmax=vmax)
    ax0.set(title='Mid-plane $B_z$: standard graph search', xlabel='x', ylabel='y')
    ax1.imshow(Bper[:, :, zmid, 2].T, origin='lower', extent=[0, 1, 0, 1], vmin=-vmax, vmax=vmax)
    ax1.set(title='Mid-plane $B_z$: periodic graph search', xlabel='x', ylabel='y')
    im2 = ax2.imshow(diff.T, origin='lower', extent=[0, 1, 0, 1], vmin=-vmaxd, vmax=vmaxd)
    ax2.set(title='Difference in $B_z$ (periodic - standard)', xlabel='x', ylabel='y')
    cbar = fig.colorbar(im2, ax=[ax0, ax1, ax2], shrink=0.9, pad=0.02)
    cbar.set_label('$B_z$ / difference')

    labels = ['x', 'y', 'z']
    xpos = np.arange(len(labels))
    width = 0.35
    for off, name, color, key in [(-width / 2, 'standard', 'tab:blue', 'standard'), (width / 2, 'periodic', 'tab:orange', 'periodic')]:
        sub = df[df['graph_search'] == key]
        means = [sub[f'seam_ratio_{lab}'].mean() for lab in labels]
        stds = [sub[f'seam_ratio_{lab}'].std(ddof=1) for lab in labels]
        ax3.bar(xpos + off, means, yerr=stds, width=width, label=name, capsize=4)
    ax3.axhline(1.0, color='0.35', ls='--', lw=1.2)
    ax3.set_xticks(xpos, labels)
    ax3.set(title='Seam jump / interior jump ratio', ylabel='ratio')
    ax3.legend(loc='upper right')

    fig.savefig(OUTDIR / 'figure_field_periodicity_checks.png', bbox_inches='tight')
    plt.close(fig)
    print('Completed figure_field_periodicity_checks.png')


def figure_fft_and_k_mapping(df, shell_k, shell_curves, spacing):
    arr_std = np.vstack(shell_curves['standard'])
    arr_per = np.vstack(shell_curves['periodic'])
    shell_mean_std = np.divide(np.nansum(arr_std, axis=0), np.sum(np.isfinite(arr_std), axis=0), out=np.full(arr_std.shape[1], np.nan), where=np.sum(np.isfinite(arr_std), axis=0) > 0)
    shell_mean_per = np.divide(np.nansum(arr_per, axis=0), np.sum(np.isfinite(arr_per), axis=0), out=np.full(arr_per.shape[1], np.nan), where=np.sum(np.isfinite(arr_per), axis=0) > 0)
    valid = np.isfinite(shell_mean_per)
    idx_min = int(np.nanargmin(shell_mean_per[valid]))
    valid_indices = np.where(valid)[0]
    idx_min = valid_indices[idx_min]
    selected = [i for i in range(max(0, idx_min - 2), min(len(shell_k), idx_min + 3)) if np.isfinite(shell_k[i])]
    selected = [i for i in selected if shell_k[i] > 0]

    fig = plt.figure(figsize=(14.8, 10.4))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])

    ax0.plot(shell_k, shell_mean_std, 'o-', label='standard graph search')
    ax0.plot(shell_k, shell_mean_per, 'o-', label='periodic graph search')
    ax0.scatter([shell_k[idx_min]], [shell_mean_per[idx_min]], s=90, marker='*', color='tab:red', zorder=5, label='lowest periodic shell')
    for i in selected:
        ax0.axvline(shell_k[i], color='0.75', ls=':', lw=1.0)
    ax0.set(title='FFT longitudinal power fraction by $|k|$ shell', xlabel=r'$|k|$', ylabel=r'$E_L/E_{\rm tot}$')
    ax0.legend(loc='upper right')

    metrics = [('eps_fd', r'$\epsilon_{\rm FD}$'), ('eps_fft', r'$\epsilon_{\rm FFT}$'), ('longitudinal_fraction_total', r'$E_L/E_{\rm tot}$')]
    x = np.arange(len(metrics))
    width = 0.35
    for off, key, label in [(-width / 2, 'standard', 'standard'), (width / 2, 'periodic', 'periodic')]:
        sub = df[df['graph_search'] == key]
        means = [sub[m].mean() for m, _ in metrics]
        stds = [sub[m].std(ddof=1) for m, _ in metrics]
        ax1.bar(x + off, means, yerr=stds, width=width, label=label, capsize=4)
    ax1.set_xticks(x, [lab for _, lab in metrics])
    ax1.set(title='Field diagnostics over seeds', ylabel='value')
    ax1.legend(loc='upper right')

    nx = int(round(1.0 / spacing[0]))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=spacing[0])
    ky = 2 * np.pi * np.fft.fftfreq(nx, d=spacing[1])
    ax2.set_title(r'$k_x$-$k_y$ plane with rings around the minimum shell')
    ax2.set_xlabel(r'$k_x$')
    ax2.set_ylabel(r'$k_y$')
    ax2.set_aspect('equal')
    kmax = max(abs(kx).max(), abs(ky).max())
    ax2.set_xlim(-kmax, kmax)
    ax2.set_ylim(-kmax, kmax)
    ax2.axhline(0.0, color='0.5', lw=1.0)
    ax2.axvline(0.0, color='0.5', lw=1.0)
    for i in selected:
        color = 'tab:red' if i == idx_min else 'tab:gray'
        lw = 2.6 if i == idx_min else 1.5
        alpha = 0.95 if i == idx_min else 0.75
        circ = Circle((0.0, 0.0), shell_k[i], fill=False, ec=color, lw=lw, alpha=alpha)
        ax2.add_patch(circ)
        ax2.text(shell_k[i] / np.sqrt(2), shell_k[i] / np.sqrt(2), fr'$k={shell_k[i]:.1f}$', color=color, fontsize=9)

    fig.savefig(OUTDIR / 'figure_fft_and_k_mapping.png', bbox_inches='tight')
    plt.close(fig)
    print('Completed figure_fft_and_k_mapping.png')


def figure_realspace_correlations(Bstd, Bper, ell, sigma2=1.0):
    corr_std = summarize_correlations(Bstd, ell, periodic=True)
    corr_per = summarize_correlations(Bper, ell, periodic=True)

    fig = plt.figure(figsize=(16.6, 8.8))
    figure_title(
        fig,
        'Real-space correlation comparison for sampled magnetic fields',
        r'$C_{\parallel}(r)=\sigma^2 e^{-r^2/(2\ell^2)}$, '
        r'$C_{\perp}(r)=\sigma^2(1-r^2/(2\ell^2))e^{-r^2/(2\ell^2)}$, '
        r'${\rm tr}\,C=C_{\parallel}+2C_{\perp}$, and $C_{xy}=C_{xz}=C_{yz}=0$.'
    )
    subfigs = fig.subfigures(1, 2, wspace=0.04)
    subfigs[0].subplots_adjust(left=0.08, right=0.96, bottom=0.11, top=0.84, wspace=0.28, hspace=0.30)
    subfigs[1].subplots_adjust(left=0.08, right=0.96, bottom=0.11, top=0.84, wspace=0.28, hspace=0.30)
    axs0 = subfigs[0].subplots(2, 3)
    axs1 = subfigs[1].subplots(2, 3)
    handles = plot_corr_panel(axs0, corr_std, ell, sigma2, 'standard graph search')
    plot_corr_panel(axs1, corr_per, ell, sigma2, 'periodic graph search')
    fig.legend(handles, ['sample mean', 'axis scatter', 'theory'], loc='upper right', bbox_to_anchor=(0.98, 0.91), frameon=True)
    fig.savefig(OUTDIR / 'figure_realspace_correlations.png', bbox_inches='tight')
    plt.close(fig)
    print('Completed figure_realspace_correlations.png')


def figure_divergence_fd_fft(Bper, coords_1d, ell, spacing):
    n = Bper.shape[0]
    zmid = n // 2
    Bmag = np.sqrt(np.sum(Bper ** 2, axis=-1))
    Brms = float(np.sqrt(np.mean(Bmag ** 2)))

    div_fd = divergence_fd_periodic(Bper, spacing)
    eps_fd_field = ell * div_fd / Brms
    shell_fd = radial_shell_profile(eps_fd_field, coords_1d, nbins=16)

    frac2d, kx_shift, ky_shift, frac_flat = modewise_longitudinal_fraction(Bper, spacing)
    shell_k, shell_frac, total_frac, _, _, _, _ = longitudinal_fraction_spectrum(Bper, spacing, nbins=16)

    fig, axs = plt.subplots(2, 3, figsize=(16.0, 8.8))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.10, top=0.86, wspace=0.28, hspace=0.34)
    figure_title(
        fig,
        'Real-space and spectral divergence diagnostics',
        r'First row: finite-difference $\epsilon=\ell(\nabla\!\cdot\!B)/B_{\rm rms}$. '
        r'Second row: modewise longitudinal power fraction $f_L(\mathbf{k})$ and its shell summary.'
    )

    vmax = np.nanpercentile(np.abs(eps_fd_field[:, :, zmid]), 99)
    im0 = axs[0, 0].imshow(eps_fd_field[:, :, zmid].T, origin='lower', extent=[0, 1, 0, 1], aspect='equal', vmin=-vmax, vmax=vmax)
    axs[0, 0].set_title(r'Mid-plane finite-difference $\epsilon(x,y,z_0)$')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    cb0 = fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.03)
    cb0.set_label(r'$\epsilon$')

    rc = shell_fd['rcent']
    axs[0, 1].plot(rc, shell_fd['mean'], 'o-', lw=1.8, ms=4.0, label=r'shell mean $\langle\epsilon\rangle$')
    axs[0, 1].fill_between(rc, shell_fd['mean'] - shell_fd['stderr'], shell_fd['mean'] + shell_fd['stderr'], alpha=0.20, label='standard error')
    axs[0, 1].plot(rc, shell_fd['rms'], '--', lw=2.0, label=r'shell rms $\sqrt{\langle\epsilon^2\rangle}$')
    axs[0, 1].axhline(0.0, ls='--', lw=1.0, color='0.4')
    axs[0, 1].set_title('Radial shell profile')
    axs[0, 1].set_xlabel(r'radius $r$')
    axs[0, 1].set_ylabel(r'$\epsilon$')
    axs[0, 1].legend(loc='upper right')

    outer = shell_fd['outer_removed_mask']
    lo = np.nanpercentile(eps_fd_field, 0.2)
    hi = np.nanpercentile(eps_fd_field, 99.8)
    bins = np.linspace(lo, hi, 70)
    axs[0, 2].hist(eps_fd_field.ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='full grid')
    axs[0, 2].hist(eps_fd_field[outer].ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='outer shell removed')
    axs[0, 2].axvline(0.0, ls='--', lw=1.2, color='0.45')
    axs[0, 2].set_title('Distribution of normalised divergence')
    axs[0, 2].set_xlabel(r'$\epsilon=\ell(\nabla\!\cdot\!B)/B_{\rm rms}$')
    axs[0, 2].set_ylabel('normalised frequency density')
    axs[0, 2].legend(loc='upper right')

    vmaxf = np.nanpercentile(frac2d, 99.5)
    im1 = axs[1, 0].imshow(frac2d.T, origin='lower', extent=[kx_shift.min(), kx_shift.max(), ky_shift.min(), ky_shift.max()], aspect='equal', vmin=0.0, vmax=vmaxf, cmap='magma')
    axs[1, 0].set_title(r'Modewise longitudinal fraction $f_L(k_x,k_y,k_z{=}0)$')
    axs[1, 0].set_xlabel(r'$k_x$')
    axs[1, 0].set_ylabel(r'$k_y$')
    cb1 = fig.colorbar(im1, ax=axs[1, 0], fraction=0.046, pad=0.03)
    cb1.set_label(r'$f_L(\mathbf{k})$')

    axs[1, 1].plot(shell_k, shell_frac, 'o-', lw=1.8, ms=4.0)
    axs[1, 1].axhline(total_frac, ls='--', lw=1.2, color='0.4', label=fr'total = {total_frac:.3f}')
    axs[1, 1].set_title('Longitudinal power fraction by $|k|$ shell')
    axs[1, 1].set_xlabel(r'$|k|$')
    axs[1, 1].set_ylabel(r'$E_L/E_{\rm tot}$')
    axs[1, 1].legend(loc='upper right')

    binsf = np.linspace(0.0, np.nanpercentile(frac_flat, 99.8), 70)
    axs[1, 2].hist(frac_flat, bins=binsf, density=True, histtype='step', linewidth=2.0, color='tab:red')
    axs[1, 2].axvline(total_frac, ls='--', lw=1.2, color='0.45', label=fr'total = {total_frac:.3f}')
    axs[1, 2].set_title('Distribution of modewise longitudinal fraction')
    axs[1, 2].set_xlabel(r'$f_L(\mathbf{k})$')
    axs[1, 2].set_ylabel('normalised frequency density')
    axs[1, 2].legend(loc='upper right')

    fig.savefig(OUTDIR / 'figure_divergence_fd_fft.png', bbox_inches='tight')
    plt.close(fig)
    print('Completed figure_divergence_fd_fft.png')


# ------------------------------
# Main data generation
# ------------------------------

def generate_fields_and_metrics(n=8, ell=0.08, n0=216, k=27, seeds=(0, 1, 2, 3), sigma2=1.0):
    points, coords_1d = make_points_3d(n)
    spacing = (1.0 / n, 1.0 / n, 1.0 / n)
    cov = gp.extras.make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=True)

    rows = []
    shell_curves = {'standard': [], 'periodic': []}
    example_fields = {}

    for seed in seeds:
        xi = jr.normal(jr.key(seed), (points.shape[0], 3))
        for periodic_graph, label in [(False, 'standard'), (True, 'periodic')]:
            graph, *_ = build_graph_local(points, n0=n0, k=k, periodic=periodic_graph, boxsize=1.0)
            B = np.asarray(gp.generate_vector(graph, cov, xi, fast_jit=True)).reshape((n, n, n, 3))
            if not np.isfinite(B).all():
                raise RuntimeError(f'NaNs encountered for graph={label}, seed={seed}.')
            if seed == seeds[0]:
                example_fields[label] = B.copy()

            Bmag = np.sqrt(np.sum(B ** 2, axis=-1))
            Brms = float(np.sqrt(np.mean(Bmag ** 2)))
            div_fd = divergence_fd_periodic(B, spacing)
            eps_fd = float(ell * np.sqrt(np.mean(div_fd ** 2)) / Brms)
            div_fft, _ = divergence_fft(B, spacing)
            eps_fft = float(ell * np.sqrt(np.mean(div_fft ** 2)) / Brms)
            shell_k, shell_frac, total_frac, *_ = longitudinal_fraction_spectrum(B, spacing, nbins=14)
            shell_curves[label].append(shell_frac)
            sx, sy, sz = seam_ratio(B)
            rows.append({
                'seed': seed,
                'graph_search': label,
                'B_rms': Brms,
                'eps_fd': eps_fd,
                'eps_fft': eps_fft,
                'longitudinal_fraction_total': float(total_frac),
                'seam_ratio_x': sx,
                'seam_ratio_y': sy,
                'seam_ratio_z': sz,
            })

    df = pd.DataFrame(rows)
    return df, shell_k, shell_curves, example_fields, coords_1d, spacing


def write_summary(df, n, n0, k, ell):
    summary = []
    summary.append('Standard covariance + periodic-tree FFT comparison (extended)')
    summary.append('===========================================================')
    summary.append(f'3D grid: n={n}, total points={n**3}, n0={n0}, k={k}, ell={ell}')
    summary.append('Covariance: extras.make_div_free_rbf_covariance(..., periodic=True)')
    summary.append('Graph comparison: standard search vs periodic predecessor search')
    summary.append('')
    for key in ['standard', 'periodic']:
        sub = df[df['graph_search'] == key]
        summary.append(f'[{key}]')
        for col in ['eps_fd', 'eps_fft', 'longitudinal_fraction_total', 'seam_ratio_x', 'seam_ratio_y', 'seam_ratio_z']:
            summary.append(f'  {col}: {sub[col].mean():.6f} ± {sub[col].std(ddof=1):.6f}')
        summary.append('')
    summary.append('Interpretation: seam ratio close to 1 means the cross-boundary jump is comparable to an interior one-cell jump.')
    summary.append('A lower FFT longitudinal fraction indicates a more divergence-free spectral realization.')
    (OUTDIR / 'summary.txt').write_text('\n'.join(summary) + '\n')


# ------------------------------
# Main
# ------------------------------

def main():
    n = 30
    ell = 0.08
    n0 = 5000
    k = 9
    sigma2 = 1.0
    seeds = (0, 1, 2, 3)

    # predecessor-search comparison figures
    figure_predecessor_search_2d(n=20, n0=250, k=27)
    figure_predecessor_search_3d(n=10, n0=200, k=27)

    # field generation and metrics
    df, shell_k, shell_curves, example_fields, coords_1d, spacing = generate_fields_and_metrics(
        n=n, ell=ell, n0=n0, k=k, seeds=seeds, sigma2=sigma2
    )
    df.to_csv(OUTDIR / 'field_metrics.csv', index=False)
    write_summary(df, n=n, n0=n0, k=k, ell=ell)

    # figure set requested for the demo
    figure_field_periodicity_checks(example_fields['standard'], example_fields['periodic'], df)
    figure_fft_and_k_mapping(df, shell_k, shell_curves, spacing)
    figure_realspace_correlations(example_fields['standard'], example_fields['periodic'], ell, sigma2=sigma2)
    figure_divergence_fd_fft(example_fields['periodic'], coords_1d, ell, spacing)

    print('Wrote outputs to', OUTDIR)


if __name__ == '__main__':
    main()
