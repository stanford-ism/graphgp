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


def generate_field(n=40, n0=220, k=10, ell=0.053, sigma2=1.0, periodic=False, seed=0):
    pts, coords_1d = build_grid(n)
    graph = gp.build_graph(pts, n0=n0, k=k)
    cov = make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=periodic)
    xi = jr.normal(jr.key(seed), (pts.shape[0], 3))
    B = gp.generate_vector(graph, cov, xi, fast_jit=True)
    Bgrid = np.asarray(B).reshape((n, n, n, 3))
    spacing = (1.0 / n, 1.0 / n, 1.0 / n)
    return Bgrid, coords_1d, spacing


def main():
    prefix = 'Bfield_realspace'
    n = 50
    n0 = 220
    k = 10
    ell = (100/n) * 0.04
    sigma2 = 1.0
    periodic = True

    Bgrid, coords_1d, spacing = generate_field(
        n=n, n0=n0, k=k, ell=ell, sigma2=sigma2, periodic=periodic, seed=0
    )
    print('Field shape:', Bgrid.shape)

    plot_realspace_figure(Bgrid, ell, sigma2, prefix)
    gc.collect()
    print('Saved figures to:', OUTDIR.resolve())

if __name__ == '__main__':
    main()
