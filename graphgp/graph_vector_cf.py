"""NIFTy-style non-parametric correlated *vector* fields for GraphGP.

This module extends the scalar ``CFGridKernel``/``GraphCF`` idea to a
three-component magnetic field.  The model learns a positive, isotropic
non-parametric spectral amplitude and turns it into the non-helical magnetic
covariance tensor

    M_ij(r) = M_N(r) delta_ij + (M_L(r) - M_N(r)) rhat_i rhat_j,

with the divergence-free Fourier projector

    Mhat_ij(k) = P_B(k) (delta_ij - khat_i khat_j).

The resulting callable can be passed directly to ``graphgp.generate_vector``.
"""

from __future__ import annotations

import operator
from dataclasses import field
from functools import partial, reduce
from typing import Callable, Mapping, Optional, Tuple

import jax.numpy as jnp
import nifty.re as jft
import numpy as np

from .graph import Graph, build_graph
from .kernels import k_log_grid, non_parametric_amplitude, r_log_grid
from .refine import generate_vector


Array = jnp.ndarray


def _as_prior_callable(value, *, kind: str, name: str, allow_none: bool = False):
    """Accept either an existing callable prior or a ``(mean, std)`` tuple."""
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"missing required prior for `{name}`")
    if isinstance(value, (tuple, list)):
        if kind == "lognormal":
            value = jft.prior.lognormal_prior(*value)
        elif kind == "normal":
            value = jft.prior.normal_prior(*value)
        else:
            raise ValueError(f"unknown prior kind {kind!r}")
    if not callable(value):
        raise TypeError(f"invalid `{name}` specified; got {type(value)!r}")
    return value


def _maybe_wrapped_lognormal(value, *, name: str):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        value = jft.prior.lognormal_prior(*value)
    if not callable(value):
        raise TypeError(f"invalid `{name}` specified; got {type(value)!r}")
    return jft.WrappedCall(value, name=name, white_init=True)


def _spherical_j0(x: Array) -> Array:
    """Stable spherical Bessel j_0(x)."""
    x = jnp.asarray(x)
    ax = jnp.abs(x)
    small = ax < 1.0e-4
    x2 = x * x
    series = 1.0 - x2 / 6.0 + x2 * x2 / 120.0 - x2 * x2 * x2 / 5040.0
    xs = jnp.where(small, 1.0, x)
    direct = jnp.sin(xs) / xs
    return jnp.where(small, series, direct)


def _spherical_j2(x: Array) -> Array:
    """Stable spherical Bessel j_2(x)."""
    x = jnp.asarray(x)
    ax = jnp.abs(x)
    small = ax < 1.0e-3
    x2 = x * x
    # j2(x) = x^2/15 - x^4/210 + x^6/7560 + O(x^8)
    series = x2 / 15.0 - x2 * x2 / 210.0 + x2 * x2 * x2 / 7560.0
    xs = jnp.where(small, 1.0, x)
    direct = ((3.0 / xs**3 - 1.0 / xs) * jnp.sin(xs)) - (3.0 * jnp.cos(xs) / xs**2)
    return jnp.where(small, series, direct)


def nonhelical_cross_spectrum_matrix(
    kvec: Array,
    transverse_power: Array,
    longitudinal_power: Array | float = 0.0,
) -> Array:
    """Return the isotropic, non-helical 3x3 vector cross-spectrum matrix.

    ``transverse_power`` multiplies ``I - khat khat^T``.  For a magnetic
    divergence-free prior, keep ``longitudinal_power=0``.  A non-zero
    ``longitudinal_power`` is useful for a general Helmholtz-decomposed vector
    field, where it multiplies ``khat khat^T``.
    """
    kvec = jnp.asarray(kvec)
    k2 = jnp.sum(kvec * kvec)
    safe_k2 = jnp.where(k2 > 0.0, k2, 1.0)
    kk = jnp.outer(kvec, kvec) / safe_k2
    eye = jnp.eye(3, dtype=jnp.result_type(kvec, transverse_power, longitudinal_power))
    # At k=0 the direction is undefined; this convention is harmless for binned spectra.
    kk = jnp.where(k2 > 0.0, kk, jnp.zeros_like(kk))
    return transverse_power * (eye - kk) + longitudinal_power * kk


def longitudinal_normal_from_projector_spectra(
    transverse_spectrum: Array,
    k_edges: Array,
    r_grid: Array,
    *,
    longitudinal_spectrum: Array | None = None,
    fourier_factor: float = 2.0 * np.pi,
    tiny: float = 1.0e-30,
) -> Tuple[Array, Array]:
    """Convert isotropic Fourier projector spectra to real-space M_L and M_N.

    The Fourier-space tensor is

        P_T(k) (delta_ij - khat_i khat_j) + P_L(k) khat_i khat_j.

    For the magnetic/non-helical/divergence-free case, use only ``P_T``.  The
    returned arrays are normalized so that ``M_L(0)=M_N(0)=1`` before the
    external fluctuation amplitude is applied.

    Parameters
    ----------
    transverse_spectrum:
        Positive values on intervals ``[k_edges[i], k_edges[i+1]]``.
    k_edges:
        Monotonic bin edges.  The first edge may be zero.
    r_grid:
        Radii at which to evaluate the real-space covariance functions.
    longitudinal_spectrum:
        Optional positive longitudinal/irrotational spectrum.  Leave as ``None``
        for magnetic fields.
    fourier_factor:
        Use ``2*pi`` when ``k`` is in cycles per coordinate unit, matching the
        existing scalar ``weights`` implementation; use ``1`` for angular wave
        numbers.
    """
    k_edges = jnp.asarray(k_edges)
    r_grid = jnp.asarray(r_grid)
    p_t = jnp.asarray(transverse_spectrum)
    if longitudinal_spectrum is None:
        p_l = jnp.zeros_like(p_t)
    else:
        p_l = jnp.asarray(longitudinal_spectrum)

    k_mid = 0.5 * (k_edges[1:] + k_edges[:-1])
    dk = k_edges[1:] - k_edges[:-1]
    shell = k_mid**2 * dk
    x = fourier_factor * r_grid[:, None] * k_mid[None, :]

    j0 = _spherical_j0(x)
    j2 = _spherical_j2(x)

    # Angular factors with r along the z-axis:
    #   I0      = 2 j0
    #   I2      = (2/3) j0 - (4/3) j2
    #   I0-I2   = (4/3) (j0 + j2)
    #   I0+I2   = (4/3) (2 j0 - j2)
    raw_ml = shell[None, :] * (
        p_t[None, :] * (4.0 / 3.0) * (j0 + j2)
        + p_l[None, :] * ((2.0 / 3.0) * j0 - (4.0 / 3.0) * j2)
    )
    raw_mn = shell[None, :] * (
        p_t[None, :] * (2.0 / 3.0) * (2.0 * j0 - j2)
        + p_l[None, :] * (2.0 / 3.0) * (j0 + j2)
    )

    # Component variance at r=0 for either diagonal component.
    denom = jnp.sum(shell * ((4.0 / 3.0) * p_t + (2.0 / 3.0) * p_l))
    denom = jnp.maximum(denom, tiny)
    ml = jnp.sum(raw_ml, axis=-1) / denom
    mn = jnp.sum(raw_mn, axis=-1) / denom
    return ml, mn


def magnetic_longitudinal_normal_from_spectrum(
    spectrum: Array,
    k_edges: Array,
    r_grid: Array,
    *,
    fourier_factor: float = 2.0 * np.pi,
) -> Tuple[Array, Array]:
    """Magnetic shorthand: one positive transverse spectrum -> M_L, M_N."""
    return longitudinal_normal_from_projector_spectra(
        spectrum,
        k_edges,
        r_grid,
        longitudinal_spectrum=None,
        fourier_factor=fourier_factor,
    )


def magnetic_tensor_from_radial_functions(
    rvec: Array,
    r_grid: Array,
    m_longitudinal: Array,
    m_normal: Array,
    *,
    variance: Array | float = 1.0,
    corrlen: Array | float = 1.0,
    twiddle_factor: float = 0.0,
    right: float = 0.0,
) -> Array:
    """Evaluate M_ij(r) from tabulated M_L and M_N."""
    rvec = jnp.asarray(rvec)
    corrlen = jnp.asarray(corrlen)
    rr = jnp.linalg.norm(rvec)
    rr_eff = rr / corrlen

    ml = jnp.interp(rr_eff, r_grid, m_longitudinal, left=m_longitudinal[0], right=right)
    mn = jnp.interp(rr_eff, r_grid, m_normal, left=m_normal[0], right=right)

    safe_rr = jnp.where(rr > 0.0, rr, 1.0)
    rhat = rvec / safe_rr
    rhat = jnp.where(rr > 0.0, rhat, jnp.zeros_like(rhat))
    eye = jnp.eye(3, dtype=jnp.result_type(rvec, ml, mn, variance))

    mat = mn * eye + (ml - mn) * jnp.outer(rhat, rhat)
    mat = variance * mat
    if twiddle_factor != 0.0:
        mat = mat + jnp.where(rr == 0.0, variance * twiddle_factor, 0.0) * eye
    return mat


def make_magnetic_covariance_element(
    r_grid: Array,
    m_longitudinal: Array,
    m_normal: Array,
    *,
    variance: Array | float = 1.0,
    corrlen: Array | float = 1.0,
    periodic: bool = False,
    boxsize: Array | float = 1.0,
    twiddle_factor: float = 1.0e-6,
) -> Callable[[Array, Array], Array]:
    """Create ``cov_elem(x, y) -> (3, 3)`` for GraphGP's vector generator."""
    r_grid = jnp.asarray(r_grid)
    m_longitudinal = jnp.asarray(m_longitudinal)
    m_normal = jnp.asarray(m_normal)
    box = jnp.asarray(boxsize)

    def cov_elem(x: Array, y: Array) -> Array:
        rvec = y - x
        if periodic:
            rvec = rvec - box * jnp.round(rvec / box)
        return magnetic_tensor_from_radial_functions(
            rvec,
            r_grid,
            m_longitudinal,
            m_normal,
            variance=variance,
            corrlen=corrlen,
            twiddle_factor=twiddle_factor,
        )

    return cov_elem


def magnetic_trace_correlation_length(
    r_grid: Array,
    m_longitudinal: Array,
    m_normal: Array,
    *,
    corrlen: Array | float = 1.0,
) -> Array:
    """Estimate lambda_B = integral dr w(r)/w(0) from tabulated functions."""
    r_grid = jnp.asarray(r_grid)
    trace = m_longitudinal + 2.0 * m_normal
    trace0 = jnp.where(jnp.abs(trace[0]) > 0.0, trace[0], 1.0)
    # jnp.trapezoid exists in recent JAX; fall back to manual trapezoid expression.
    y = trace / trace0
    dr = r_grid[1:] - r_grid[:-1]
    integral = jnp.sum(0.5 * (y[1:] + y[:-1]) * dr)
    return corrlen * integral


class MagneticCFGridKernel(jft.Model):
    """Non-parametric, non-helical magnetic covariance tensor model.

    The learned scalar spectral amplitude replaces the fixed power law/RBF shape.
    In Fourier space the 3x3 cross-spectrum matrix is assembled as
    ``P_B(k) (I - khat khat.T)``.  In real space, this becomes a pair of tabulated
    functions ``M_L(r)`` and ``M_N(r)`` used by GraphGP's block-vector covariance.

    ``corrlen`` is optional.  If provided, it is a log-normal prior or callable
    model that rescales physical separations as ``r / corrlen``.  This gives a
    learnable global correlation length on top of the non-parametric spectral
    shape; because the non-parametric spectrum can also shift power in ``k``, it
    should usually be given a moderately informative prior.
    """

    fluctuations: jft.Model = field(metadata=dict(static=False))
    spectrum: jft.Model = field(metadata=dict(static=False))
    corrlen: Optional[jft.Model] = field(default=None, metadata=dict(static=False))
    twiddle_factor: float = field(metadata=dict(static=True))
    rs: Array = field(metadata=dict(static=True))
    ks: Array = field(metadata=dict(static=True))
    periodic: bool = field(metadata=dict(static=True))
    boxsize: Array | float = field(metadata=dict(static=True))
    fourier_factor: float = field(metadata=dict(static=True))
    name: str = field(metadata=dict(static=True))

    def __init__(
        self,
        rmin: float,
        rmax: float,
        nbins: int,
        fluctuations,
        loglogavgslope,
        flexibility=None,
        *,
        asperity=None,
        corrlen=None,
        twiddle_factor: float = 1.0e-6,
        periodic: bool = False,
        boxsize: Array | float = 1.0,
        fourier_factor: float = 2.0 * np.pi,
        name: str = "b_kernel",
    ):
        self.twiddle_factor = float(twiddle_factor)
        self.rs = jnp.asarray(r_log_grid(rmin, rmax, nbins))
        self.ks = jnp.asarray(k_log_grid(rmin, rmax, nbins))
        self.periodic = bool(periodic)
        self.boxsize = boxsize
        self.fourier_factor = float(fourier_factor)
        self.name = name

        flu = _as_prior_callable(fluctuations, kind="lognormal", name="fluctuations")
        slp = _as_prior_callable(loglogavgslope, kind="normal", name="loglogavgslope")
        flx = _as_prior_callable(flexibility, kind="lognormal", name="flexibility", allow_none=True)
        asp = _as_prior_callable(asperity, kind="lognormal", name="asperity", allow_none=True)

        fluctuations_model, spectrum_model = non_parametric_amplitude(
            self.ks,
            flu,
            slp,
            flx,
            asp,
            prefix=name,
        )
        self.fluctuations = fluctuations_model
        self.spectrum = spectrum_model
        self.corrlen = _maybe_wrapped_lognormal(corrlen, name=name + "corrlen")

        domain = self.fluctuations.domain | self.spectrum.domain
        if self.corrlen is not None:
            domain = domain | self.corrlen.domain
        super().__init__(domain=domain, white_init=True, target=callable)

    def normalized_spectrum(self, x: Mapping) -> Array:
        """Spectrum values on the stored k-bin edges, normalized internally."""
        return self.spectrum(x)

    def _corrlen_value(self, x: Mapping) -> Array:
        return 1.0 if self.corrlen is None else self.corrlen(x)

    def radial_kernels(self, x: Mapping) -> Tuple[Array, Array]:
        """Return physical-variance ``(M_L, M_N)`` arrays on ``self.rs``."""
        sigma = self.fluctuations(x)
        spectrum_edges = self.spectrum(x)
        spectrum_mid = 0.5 * (spectrum_edges[1:] + spectrum_edges[:-1])
        ml, mn = magnetic_longitudinal_normal_from_spectrum(
            spectrum_mid,
            self.ks,
            self.rs,
            fourier_factor=self.fourier_factor,
        )
        variance = sigma**2
        return variance * ml, variance * mn

    def correlation_length(self, x: Mapping) -> Array:
        """A diagnostic estimate of lambda_B from the current real-space trace."""
        ml, mn = self.radial_kernels(x)
        sigma = self.fluctuations(x)
        variance = jnp.maximum(sigma**2, 1.0e-30)
        return magnetic_trace_correlation_length(
            self.rs,
            ml / variance,
            mn / variance,
            corrlen=self._corrlen_value(x),
        )

    def covariance_element(self, x: Mapping) -> Callable[[Array, Array], Array]:
        """Return a GraphGP-compatible ``cov_elem(x_point, y_point)`` callable."""
        sigma = self.fluctuations(x)
        spectrum_edges = self.spectrum(x)
        spectrum_mid = 0.5 * (spectrum_edges[1:] + spectrum_edges[:-1])
        ml, mn = magnetic_longitudinal_normal_from_spectrum(
            spectrum_mid,
            self.ks,
            self.rs,
            fourier_factor=self.fourier_factor,
        )
        return make_magnetic_covariance_element(
            self.rs,
            ml,
            mn,
            variance=sigma**2,
            corrlen=self._corrlen_value(x),
            periodic=self.periodic,
            boxsize=self.boxsize,
            twiddle_factor=self.twiddle_factor,
        )

    def __call__(self, x: Mapping) -> Callable[[Array, Array], Array]:
        return self.covariance_element(x)


def get_coordinates(grid) -> np.ndarray:
    """Coordinates of the finest level of a NIFTy/JFT grid as ``shape + (ndim,)``."""
    grid_at_level = grid.at(-1)
    idx = np.meshgrid(
        *(np.arange(int(s), dtype=int) for s in grid_at_level.shape),
        indexing="ij",
    )
    idx = np.stack(idx, axis=0)
    coords = grid_at_level.index2coord(idx)
    return np.moveaxis(np.asarray(coords), 0, -1)


class GraphVectorCF(jft.Model):
    """GraphGP realization model for a learned 3-component correlated field."""

    points: Array = field(metadata=dict(static=False))
    neighbors: Array = field(metadata=dict(static=False))
    indices: Array = field(metadata=dict(static=False))
    kernel: jft.Model = field(metadata=dict(static=False))
    offsets: Tuple[int, ...] = field(metadata=dict(static=True))

    def __init__(
        self,
        points: Array,
        neighbors: Array,
        indices: Array,
        offsets: Tuple[int, ...],
        kernel: MagneticCFGridKernel,
        shape: Tuple[int, ...],
        *,
        cuda: bool = False,
        fast_jit: bool = True,
        key: str = "graph_b_xi",
        dtype=jnp.float64,
    ):
        self.points = jnp.asarray(points).astype(jnp.float32)
        self.neighbors = jnp.asarray(neighbors).astype(jnp.int32)
        self.indices = jnp.asarray(indices).astype(jnp.int32) if indices is not None else None
        self.offsets = tuple(int(o) for o in offsets)
        self.cuda = bool(cuda)
        self.fast_jit = bool(fast_jit)
        self.shape = tuple(int(s) for s in shape)
        self.kernel = kernel
        self.key = key

        xi_shape = self.shape + (3,)
        domain = {key: jft.ShapeWithDtype(xi_shape, dtype)}
        domain = domain | kernel.domain
        init = jft.Initializer({key: partial(jft.random_like, primals=domain[key])})
        init = init | kernel.init
        super().__init__(domain=domain, init=init)

    def __call__(self, x: Mapping):
        cov_elem = self.kernel(x)
        graph = Graph(self.points, self.neighbors, self.offsets, self.indices)
        xi = x[self.key].reshape((-1, 3))
        values = generate_vector(
            graph,
            cov_elem,
            xi,
            cuda=self.cuda,
            fast_jit=self.fast_jit,
        )
        return [values.reshape(self.shape + (3,))]

    @staticmethod
    def from_points(
        points: Array,
        kernel: MagneticCFGridKernel,
        *,
        n0: int,
        k: int,
        shape: Tuple[int, ...] | None = None,
        cuda: bool = False,
        fast_jit: bool = True,
        periodic_graph: bool = False,
        boxsize: Array | float | None = None,
        key: str = "graph_b_xi",
        dtype=jnp.float64,
    ) -> "GraphVectorCF":
        points = jnp.asarray(points).astype(jnp.float32)
        if shape is None:
            shape = (int(points.shape[0]),)
        n0 = min(int(n0), int(points.shape[0]))
        build_kwargs = dict(n0=n0, k=int(k), cuda=bool(cuda))
        if periodic_graph:
            build_kwargs.update(periodic=True, boxsize=boxsize)
        graph = build_graph(points, **build_kwargs)
        return GraphVectorCF(
            graph.points,
            graph.neighbors,
            graph.indices,
            graph.offsets,
            kernel,
            tuple(shape),
            cuda=cuda,
            fast_jit=fast_jit,
            key=key,
            dtype=dtype,
        )

    @staticmethod
    def from_grid(
        grid,
        kernel: MagneticCFGridKernel,
        *,
        n0: int,
        k: int,
        cuda: bool = False,
        fast_jit: bool = True,
        periodic_graph: bool = False,
        boxsize: Array | float | None = None,
        key: str = "graph_b_xi",
        dtype=jnp.float64,
    ) -> "GraphVectorCF":
        coords = get_coordinates(grid)
        shape = coords.shape[:-1]
        n0 = min(int(n0), reduce(operator.mul, shape))
        flat = coords.reshape((-1, coords.shape[-1])).astype(np.float32)
        return GraphVectorCF.from_points(
            flat,
            kernel,
            n0=n0,
            k=k,
            shape=shape,
            cuda=cuda,
            fast_jit=fast_jit,
            periodic_graph=periodic_graph,
            boxsize=boxsize,
            key=key,
            dtype=dtype,
        )


__all__ = [
    "GraphVectorCF",
    "MagneticCFGridKernel",
    "get_coordinates",
    "longitudinal_normal_from_projector_spectra",
    "magnetic_longitudinal_normal_from_spectrum",
    "magnetic_tensor_from_radial_functions",
    "make_magnetic_covariance_element",
    "magnetic_trace_correlation_length",
    "nonhelical_cross_spectrum_matrix",
]
