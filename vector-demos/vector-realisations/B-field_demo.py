import pathlib
import gc

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import graphgp as gp

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import colors as mcolors

OUTDIR = pathlib.Path('temp-figures')
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


def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=False):
    ell2 = ell * ell

    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)
        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))

    return cov_elem


def build_grid(n):
    xs = (jnp.arange(n) + 0.5) / n
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing='ij')
    pts = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))
    return pts, np.asarray(xs)


def divergence_fd_nonperiodic(Bgrid, spacing):
    dx, dy, dz = spacing
    Bx = np.asarray(Bgrid[..., 0])
    By = np.asarray(Bgrid[..., 1])
    Bz = np.asarray(Bgrid[..., 2])
    return (
        np.gradient(Bx, dx, axis=0, edge_order=2)
        + np.gradient(By, dy, axis=1, edge_order=2)
        + np.gradient(Bz, dz, axis=2, edge_order=2)
    )


def overlapping_shift_pair(Bgrid, axis, h):
    if h == 0:
        return np.asarray(Bgrid), np.asarray(Bgrid)
    sl0 = [slice(None)] * 4
    sl1 = [slice(None)] * 4
    sl0[axis] = slice(0, Bgrid.shape[axis] - h)
    sl1[axis] = slice(h, Bgrid.shape[axis])
    return np.asarray(Bgrid[tuple(sl0)]), np.asarray(Bgrid[tuple(sl1)])


def correlation_tensor_axis_lags(Bgrid, max_lag):
    lags = np.arange(0, max_lag + 1)
    Ct = np.zeros((3, 3, 3, len(lags)), dtype=np.float64)
    for a in range(3):
        for ih, h in enumerate(lags):
            A, C = overlapping_shift_pair(Bgrid, a, int(h))
            for i in range(3):
                for j in range(3):
                    Ct[a, i, j, ih] = np.mean(A[..., i] * C[..., j])
    return lags, Ct


def summarize_correlations(Bgrid, ell):
    n = Bgrid.shape[0]
    max_lag = min(n // 2, max(6, int(np.ceil(9.0 * ell * n))))
    lags, Ct = correlation_tensor_axis_lags(Bgrid, max_lag)
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


def figure_title(fig, title, eq):
    fig.text(0.06, 0.985, title, ha='left', va='top', fontsize=15.5, fontweight='bold')
    fig.text(0.06, 0.952, eq, ha='left', va='top', fontsize=11.4)


def decorate_xy(ax):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')


def face_rgba(arr, norm, cmap='viridis', alpha=1.0):
    rgba = plt.get_cmap(cmap)(norm(arr))
    rgba[..., -1] = alpha
    return rgba


def projected_face_textures(Bgrid, mode='mean'):
    Bmag = np.linalg.norm(Bgrid, axis=-1)
    if mode == 'mean':
        tex_xy = Bmag.mean(axis=2)
        tex_xz = Bmag.mean(axis=1)
        tex_yz = Bmag.mean(axis=0)
    elif mode == 'max':
        tex_xy = Bmag.max(axis=2)
        tex_xz = Bmag.max(axis=1)
        tex_yz = Bmag.max(axis=0)
    else:
        raise ValueError(f'unknown projection mode: {mode}')
    return tex_xy, tex_xz, tex_yz, Bmag


def draw_box_edges(ax, lw=2.0, color='k', alpha=1.0):
    corners = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        pa, pb = corners[a], corners[b]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            lw=lw,
            color=color,
            alpha=alpha,
            solid_capstyle='round',
        )


def plot_projected_cube(ax, Bgrid, ell, sigma2, metrics, projection_mode='mean'):
    tex_xy, tex_xz, tex_yz, Bmag = projected_face_textures(Bgrid, mode=projection_mode)
    vmin = np.percentile(Bmag, 3)
    vmax = np.percentile(Bmag, 99)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'

    n = Bgrid.shape[0]
    s = np.linspace(0.0, 1.0, n)

    # z-faces
    Xz, Yz = np.meshgrid(s, s, indexing='ij')
    Ztop = np.ones_like(Xz)
    Zbot = np.zeros_like(Xz)
    fc_xy = face_rgba(tex_xy.T, norm, cmap=cmap)
    ax.plot_surface(Xz, Yz, Ztop, facecolors=fc_xy, shade=False, linewidth=0, antialiased=False)
    ax.plot_surface(Xz, Yz, Zbot, facecolors=fc_xy, shade=False, linewidth=0, antialiased=False)

    # y-faces
    Xy, Zy = np.meshgrid(s, s, indexing='ij')
    Yfront = np.zeros_like(Xy)
    Yback = np.ones_like(Xy)
    fc_xz = face_rgba(np.flipud(tex_xz.T), norm, cmap=cmap)
    ax.plot_surface(Xy, Yfront, Zy, facecolors=fc_xz, shade=False, linewidth=0, antialiased=False)
    ax.plot_surface(Xy, Yback, Zy, facecolors=fc_xz, shade=False, linewidth=0, antialiased=False)

    # x-faces
    Yx, Zx = np.meshgrid(s, s, indexing='ij')
    Xleft = np.zeros_like(Yx)
    Xright = np.ones_like(Yx)
    fc_yz = face_rgba(np.flipud(tex_yz.T), norm, cmap=cmap)
    ax.plot_surface(Xleft, Yx, Zx, facecolors=fc_yz, shade=False, linewidth=0, antialiased=False)
    ax.plot_surface(Xright, Yx, Zx, facecolors=fc_yz, shade=False, linewidth=0, antialiased=False)

    draw_box_edges(ax, lw=2.0, color='k', alpha=1.0)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=23, azim=-58)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    ax.set_title(r'Projected, mean magnetic field amplitude')

    note = (
        rf'$\ell={ell:.3f},\ \sigma^2={sigma2:.2f}$' + '\n' +
        rf'$B_{{\rm rms}}={metrics["B_rms"]:.3f}$' + '\n' +
        rf'$\epsilon_{{\rm rms}}={metrics["epsilon_rms"]:.3f}$'
    )
    ax.text2D(
        0.02, 0.96, note,
        transform=ax.transAxes,
        ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='0.55')
    )


def plot_slice_figure(Bgrid, coords_1d, ell, sigma2, spacing, metrics, prefix):
    n = Bgrid.shape[0]
    zslice = n // 2
    z0 = coords_1d[zslice]

    Bx = np.asarray(Bgrid[:, :, zslice, 0])
    By = np.asarray(Bgrid[:, :, zslice, 1])
    Bz = np.asarray(Bgrid[:, :, zslice, 2])
    Babs = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    # Use one explicit plotting convention for both quiver and streamlines.
    xplot = np.asarray(coords_1d)
    yplot = np.asarray(coords_1d)
    Xplot, Yplot = np.meshgrid(xplot, yplot, indexing='xy')
    U = Bx.T
    V = By.T
    mag = np.hypot(U, V)
    Udir = U / np.where(mag > 0.0, mag, 1.0)
    Vdir = V / np.where(mag > 0.0, mag, 1.0)

    fig = plt.figure(figsize=(15.4, 7.8))
    figure_title(
        fig,
        'Slice anatomy and 3D structure of the sampled magnetic field',
        r'$M_{ij}(r)=\sigma^2[a(r)\delta_{ij}+b(r)r_i r_j]$, '
        r'$a(r)=\left(1-r^2/(2\ell^2)\right)e^{-r^2/(2\ell^2)}$, '
        r'$b(r)=e^{-r^2/(2\ell^2)}/(2\ell^2)$, such that $\nabla\!\cdot\!\mathbf{B}=0$.'
    )

    subfigs = fig.subfigures(1, 2, width_ratios=[1.02, 1.00], wspace=0.04)

    left = subfigs[0]
    left.subplots_adjust(left=0.04, right=0.98, bottom=0.09, top=0.88)
    ax3d = left.add_subplot(111, projection='3d')
    plot_projected_cube(ax3d, Bgrid, ell=ell, sigma2=sigma2, metrics=metrics, projection_mode='mean')

    right = subfigs[1]
    axs = right.subplots(2, 2)
    right.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.86, wspace=0.24, hspace=0.28)
    extent = [0.0, 1.0, 0.0, 1.0]

    im0 = axs[0, 0].imshow(Babs.T, origin='lower', extent=extent, aspect='equal')
    axs[0, 0].set_title(r'Field amplitude $|\mathbf{B}|(x,y,z_0)$')
    right.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.03)

    axs[0, 1].imshow(Babs.T, origin='lower', extent=extent, aspect='equal', alpha=0.28)
    axs[0, 1].quiver(
        Xplot, Yplot, Udir, Vdir,
        angles='xy', scale_units='xy', scale=45.0,
        color='k', width=0.0020, alpha=0.90, pivot='mid'
    )
    axs[0, 1].set_title(r'$(B_x,B_y)$ quiver at $z_0$')

    vmax_bz = np.nanpercentile(np.abs(Bz), 99)
    im2 = axs[1, 0].imshow(
        Bz.T, origin='lower', extent=extent, aspect='equal',
        vmin=-vmax_bz, vmax=vmax_bz
    )
    axs[1, 0].set_title(r'Signed out-of-plane component $B_z(x,y,z_0)$')
    right.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.03)

    axs[1, 1].imshow(Babs.T, origin='lower', extent=extent, aspect='equal', alpha=0.28)
    axs[1, 1].streamplot(
        xplot, yplot, U, V,
        density=1.1, color='k', linewidth=0.75
    )
    axs[1, 1].set_title(r'Streamlines of $(B_x,B_y)$')

    for ax in axs.ravel():
        decorate_xy(ax)

    fig.savefig(OUTDIR / f'{prefix}_slice_validation.png', bbox_inches='tight')
    plt.close(fig)


def plot_realspace_figure(Bgrid, ell, sigma2, prefix):
    corr = summarize_correlations(Bgrid, ell)
    r = corr['r']
    x = r / ell
    cpar_th, cperp_th, ctrace_th = theory_parallel_perp_trace(r, ell, sigma2)
    ctrace = corr['cpar_mean'] + 2.0 * corr['cperp_mean']
    ctrace_std = np.sqrt(corr['cpar_std'] ** 2 + 4.0 * corr['cperp_std'] ** 2)

    fig, axs = plt.subplots(2, 3, figsize=(15.4, 8.0), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.89, bottom=0.13, top=0.86, wspace=0.25, hspace=0.28)
    figure_title(
        fig,
        'Real-space correlation structure of the sampled magnetic field',
        r'$C_{\parallel}(r)=\sigma^2 e^{-r^2/(2\ell^2)}$, '
        r'$C_{\perp}(r)=\sigma^2(1-r^2/(2\ell^2))e^{-r^2/(2\ell^2)}$, '
        r'${\rm tr}\,C=C_{\parallel}+2C_{\perp}$, and $C_{xy}=C_{xz}=C_{yz}=0$.'
    )

    legend_handles = None
    legend_labels = None
    top = [
        (axs[0, 0], corr['cpar_mean'] / sigma2, corr['cpar_std'] / sigma2, cpar_th / sigma2, r'$C_{\parallel}(r)/\sigma^2$', None),
        (axs[0, 1], corr['cperp_mean'] / sigma2, corr['cperp_std'] / sigma2, cperp_th / sigma2, r'$C_{\perp}(r)/\sigma^2$', np.sqrt(2.0)),
        (axs[0, 2], ctrace / sigma2, ctrace_std / sigma2, ctrace_th / sigma2, r'${\rm tr}\,C(r)/\sigma^2$', None),
    ]
    for ax, mean, std, theory, title, xline in top:
        h1 = ax.plot(x, mean, 'o', ms=4.2, label='sample mean')
        h2 = ax.fill_between(x, mean - std, mean + std, alpha=0.20, label='axis scatter')
        h3 = ax.plot(x, theory, lw=2.1, label='theory')
        ax.axhline(0.0, ls='--', lw=1.0, color='0.4')
        if xline is not None:
            ax.axvline(xline, ls=':', lw=1.2, color='0.35')
            ax.text(xline + 0.10, 0.055, r'$r/\ell=\sqrt{2}$', fontsize=10)
        ax.set_title(title)
        ax.set_ylabel('correlation')
        if legend_handles is None:
            legend_handles = [h1[0], h2, h3[0]]
            legend_labels = ['sample mean', 'axis scatter', 'theory']

    bottom = [
        (axs[1, 0], corr['cxy_mean'] / sigma2, corr['cxy_std'] / sigma2, r'$C_{xy}(r)/\sigma^2$'),
        (axs[1, 1], corr['cxz_mean'] / sigma2, corr['cxz_std'] / sigma2, r'$C_{xz}(r)/\sigma^2$'),
        (axs[1, 2], corr['cyz_mean'] / sigma2, corr['cyz_std'] / sigma2, r'$C_{yz}(r)/\sigma^2$'),
    ]
    for ax, mean, std, title in bottom:
        ax.plot(x, mean, 'o', ms=4.0)
        ax.fill_between(x, mean - std, mean + std, alpha=0.20)
        ax.axhline(0.0, color='tab:orange', lw=2.0)
        ax.set_title(title)
        ax.set_xlabel(r'$r/\ell$')
        ax.set_ylabel('correlation')

    for ax in axs[0, :]:
        ax.set_xlabel(r'$r/\ell$')

    # Single figure-level legend, outside the axes region.
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper left',
        bbox_to_anchor=(0.905, 0.905),
        frameon=True,
        borderaxespad=0.0,
    )
    fig.savefig(OUTDIR / f'{prefix}_realspace_validation.png', bbox_inches='tight')
    plt.close(fig)


def plot_divergence_figure(Bgrid, coords_1d, metrics, prefix):
    n = Bgrid.shape[0]
    zslice = n // 2
    eps = np.asarray(metrics['epsilon_fd'])
    eps_mid = eps[:, :, zslice]
    shell = radial_shell_profile(eps, coords_1d, nbins=16)
    outer = shell['outer_removed_mask']

    fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.8))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.84, wspace=0.26)
    figure_title(fig, 'Divergence diagnostics via finite differences', r'The normalised finite difference measure is given by $\epsilon=\ell(\nabla\!\cdot\!B)/B_{\rm rms}$.')

    vmax = np.nanpercentile(np.abs(eps_mid), 99)
    im = axs[0].imshow(eps_mid.T, origin='lower', extent=[0, 1, 0, 1], aspect='equal', vmin=-vmax, vmax=vmax)
    axs[0].set_title(r'Mid-plane finite-difference $\epsilon(x,y,z_0)$')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
    cb.set_label(r'$\epsilon$')

    rc = shell['rcent']
    mean = shell['mean']
    stderr = shell['stderr']
    rms = shell['rms']
    axs[1].plot(rc, mean, 'o-', lw=1.8, ms=4.0, label=r'shell mean $\langle\epsilon\rangle$')
    axs[1].fill_between(rc, mean - stderr, mean + stderr, alpha=0.20, label='standard error')
    axs[1].plot(rc, rms, '--', lw=2.0, label=r'shell rms $\sqrt{\langle\epsilon^2\rangle}$')
    axs[1].axhline(0.0, ls='--', lw=1.0, color='0.4')
    axs[1].set_title('Radial shell profile')
    axs[1].set_xlabel(r'radius $r$')
    axs[1].legend(loc='upper right')

    lo = np.nanpercentile(eps, 0.2)
    hi = np.nanpercentile(eps, 99.8)
    bins = np.linspace(lo, hi, 70)
    axs[2].hist(eps.ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='full grid')
    axs[2].hist(eps[outer].ravel(), bins=bins, density=True, histtype='step', linewidth=2.0, label='outer shell removed')
    axs[2].axvline(0.0, ls='--', lw=1.2, color='0.45')
    axs[2].set_title('Distribution of normalised divergence')
    axs[2].set_xlabel(r'$\epsilon=\ell(\nabla\!\cdot\!B)/B_{\rm rms}$')
    axs[2].set_ylabel(r'normalised frequency density')
    axs[2].legend(loc='upper right')

    fig.savefig(OUTDIR / f'{prefix}_divergence_aux.png', bbox_inches='tight')
    plt.close(fig)


def generate_field(n=40, n0=220, k=10, ell=0.053, sigma2=1.0, periodic=False, seed=0):
    pts, coords_1d = build_grid(n)
    graph = gp.build_graph(pts, n0=n0, k=k)
    cov = make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=periodic)
    xi = jr.normal(jr.key(seed), (pts.shape[0], 3))
    B = gp.generate_vector(graph, cov, xi, fast_jit=True)
    Bgrid = np.asarray(B).reshape((n, n, n, 3))
    spacing = (1.0 / n, 1.0 / n, 1.0 / n)
    div_fd = divergence_fd_nonperiodic(Bgrid, spacing)
    Bmag = np.linalg.norm(Bgrid, axis=-1)
    Brms = float(np.sqrt(np.mean(Bmag ** 2)))
    eps = ell * div_fd / Brms
    metrics = {
        'B_rms': Brms,
        'div_fd': div_fd,
        'div_fd_rms': float(np.sqrt(np.mean(div_fd ** 2))),
        'epsilon_fd': eps,
        'epsilon_rms': float(np.sqrt(np.mean(eps ** 2))),
    }
    return Bgrid, coords_1d, spacing, metrics


def main():
    prefix = 'demo_vector_paper_v6'
    n = 50
    n0 = 220
    k = 10
    ell = (100/n) * 0.04
    sigma2 = 1.0
    periodic = True

    Bgrid, coords_1d, spacing, metrics = generate_field(
        n=n, n0=n0, k=k, ell=ell, sigma2=sigma2, periodic=periodic, seed=0
    )
    print('Field shape:', Bgrid.shape)
    print('B_rms:', metrics['B_rms'])
    print('epsilon_rms:', metrics['epsilon_rms'])

    plot_slice_figure(Bgrid, coords_1d, ell, sigma2, spacing, metrics, prefix)
    gc.collect()
    plot_realspace_figure(Bgrid, ell, sigma2, prefix)
    gc.collect()
    plot_divergence_figure(Bgrid, coords_1d, metrics, prefix)
    gc.collect()
    print('Saved figures to:', OUTDIR.resolve())

   


if __name__ == '__main__':
    main()
