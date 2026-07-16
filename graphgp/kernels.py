import numpy as np
import nifty.re as jft
import jax.numpy as jnp
from typing import Callable, Mapping, Optional
from functools import partial
from dataclasses import field


class SimpleKernel(jft.Model):
    a: jft.Model = field(metadata=dict(static=False))
    sc: jft.Model = field(metadata=dict(static=False))
    cutoff: jft.Model = field(metadata=dict(static=False))
    twiddle_factor: float = field(metadata=dict(static=True))

    def __init__(self, a, sc, cutoff, name="kernel", twiddle_factor=1e-6):
        self.a = jft.LogNormalPrior(a[0], a[1], name=name + "_a")
        self.sc = jft.LogNormalPrior(sc[0], sc[1], name=name + "_sc")
        self.cutoff = jft.LogNormalPrior(cutoff[0], cutoff[1], name=name + "_cutoff")
        self.twiddle_factor = twiddle_factor
        domain = self.a.domain | self.sc.domain | self.cutoff.domain
        init = self.a.init | self.sc.init | self.cutoff.init
        super().__init__(domain=domain, init=init, target=callable)

    def __call__(self, x):
        a = self.a(x)
        sc = self.sc(x)
        cutoff = self.cutoff(x)

        def kernel(r):
            res = a**2 / (1.0 + (r / cutoff) ** 2) ** (sc / 2)
            res = jnp.where(r == 0.0, res * (1.0 + self.twiddle_factor), res)
            return res

        return kernel


# %%
def r_log_grid(rmin, rmax, nbins):
    assert rmin > 0.0
    assert rmax > rmin
    return np.concatenate(
        [
            np.array(
                [
                    0.0,
                ]
            ),
            np.exp(np.linspace(np.log(rmin), np.log(rmax), nbins - 1)),
        ]
    )


def k_log_grid(rmin, rmax, nbins):
    return r_log_grid(1.0 / rmax, 1.0 / rmin, nbins)


def lin_k_grid(rmin, rmax, nbins):
    return np.linspace(0.0, 1.0 / rmin, nbins)


def weights(k, r):
    fct = 2.0 * np.pi * r
    wgt = (np.sin(fct * k[..., 1:]) - np.sin(fct * k[..., :-1])) / fct**2
    wgt -= (
        k[..., 1:] * np.cos(fct * k[..., 1:]) - k[..., :-1] * np.cos(fct * k[..., :-1])
    ) / fct
    wgt /= r
    return wgt


def norm(k):
    return (k[1:] ** 3 - k[:-1] ** 3) * (2 * np.pi / 3)


def kernel(Fk, weights, norm):
    res = weights @ Fk
    res0 = (norm * Fk).sum()
    return jnp.concatenate(
        [
            jnp.array(
                [
                    1.0,
                ]
            ),
            res / res0,
        ]
    )


def kernel_eval(r, rs, Fr):
    idx_r = jnp.searchsorted(rs, r)
    idx_l = idx_r - 1
    maskl = idx_l < 0
    maskr = idx_r >= rs.size
    # idx_l, idx_r = idx_l.clip(0, rs.size - 1), idx_r.clip(0, rs.size - 1)
    norm = rs[idx_r] - rs[idx_l]
    norm = jnp.where(maskl | maskr, 1.0, norm)
    ep = (r - rs[idx_l]) / norm
    res = Fr[idx_r] * ep + Fr[idx_l] * (1.0 - ep)
    res = jnp.where(~maskl, res, Fr[0])
    res = jnp.where(~maskr, res, 0.0)
    return res


def _remove_slope(rel_log_mode_dist, x):
    sc = rel_log_mode_dist / rel_log_mode_dist[-1]
    return x - x[-1] * sc


def _log_modes(m_length):
    lm = jnp.log(m_length)
    um = jnp.concatenate(
        (
            jnp.array(
                [
                    0.0,
                ]
            ),
            lm[1:] - lm[1],
        )
    )
    assert um[0] == 0.0
    log_vol = um[2:] - um[1:-1]
    assert um.shape[0] - 2 == log_vol.shape[0]
    return um, log_vol


def non_parametric_amplitude(
    k_array,
    fluctuations: Callable,
    loglogavgslope: Callable,
    flexibility: Optional[Callable] = None,
    asperity: Optional[Callable] = None,
    prefix: str = "",
    kind: str = "amplitude",
) -> jft.Model:
    """Constructs a function computing the amplitude of a non-parametric power
    spectrum

    See
    :class:`nifty8.re.correlated_field.CorrelatedFieldMaker.add_fluctuations`
    for more details on the parameters.

    See also
    --------
    `Variable structures in M87* from space, time and frequency resolved
    interferometry`, Arras, Philipp and Frank, Philipp and Haim, Philipp
    and Knollmüller, Jakob and Leike, Reimar and Reinecke, Martin and
    Enßlin, Torsten, `<https://arxiv.org/abs/2002.05218>`_
    `<http://dx.doi.org/10.1038/s41550-021-01548-0>`_
    """
    # totvol = grid.total_volume
    # rel_log_mode_len = grid.harmonic_grid.relative_log_mode_lengths
    # mode_multiplicity = grid.harmonic_grid.mode_multiplicity
    # log_vol = grid.harmonic_grid.log_volume
    rel_log_mode_len, log_vol = _log_modes(k_array)

    fluctuations = jft.WrappedCall(
        fluctuations, name=prefix + "fluctuations", white_init=True
    )

    loglogavgslope = jft.WrappedCall(
        loglogavgslope, name=prefix + "loglogavgslope", white_init=True
    )
    ptree = loglogavgslope.domain.copy()
    if flexibility is not None and (log_vol.size > 0):
        flexibility = jft.WrappedCall(
            flexibility, name=prefix + "flexibility", white_init=True
        )
        assert log_vol is not None
        assert rel_log_mode_len.ndim == log_vol.ndim == 1
        if asperity is not None:
            asperity = jft.WrappedCall(
                asperity, name=prefix + "asperity", white_init=True
            )
        deviations = jft.IntegratedWienerProcess(
            jnp.zeros((2,)),
            flexibility,
            log_vol,
            name=prefix + "spectrum",
            asperity=asperity,
        )
        ptree.update(deviations.domain)
    else:
        deviations = None

    def correlate(primals: Mapping) -> jnp.ndarray:
        slope = loglogavgslope(primals)
        slope *= rel_log_mode_len
        ln_spectrum = slope

        if deviations is not None:
            twolog = deviations(primals)
            # Prepend zeromode
            twolog = jnp.concatenate((jnp.zeros((1,)), twolog[:, 0]))
            ln_spectrum += _remove_slope(rel_log_mode_len, twolog)

        # Exponentiate and norm the power spectrum
        spectrum = jnp.exp(ln_spectrum)
        spectrum = spectrum.at[0].set(spectrum[1])
        return spectrum

    return fluctuations, jft.Model(
        correlate, domain=ptree, init=partial(jft.random_like, primals=ptree)
    )


# %%
class CFKernel(jft.Model):
    fluctuations: jft.Model = field(metadata=dict(static=False))
    spectrum: jft.Model = field(metadata=dict(static=False))
    weights: jnp.ndarray = field(metadata=dict(static=False))
    norm: jnp.ndarray = field(metadata=dict(static=False))

    twiddle_factor: float = field(metadata=dict(static=True))
    rs: jnp.ndarray = field(metadata=dict(static=True))
    ks: jnp.ndarray = field(metadata=dict(static=True))
    name: str = field(metadata=dict(static=True))

    def __init__(
        self,
        rmin,
        rmax,
        nbins,
        fluctuations,
        loglogavgslope,
        flexibility,
        twiddle_factor,
        asperity=None,
        name="kernel",
    ):
        self.twiddle_factor = twiddle_factor
        self.rs = jnp.array(r_log_grid(rmin, rmax, nbins))
        self.ks = jnp.array(k_log_grid(rmin, rmax, nbins))
        self.name = name

        flu = fluctuations
        if isinstance(flu, (tuple, list)):
            flu = jft.prior.lognormal_prior(*flu)
        elif not callable(flu):
            te = f"invalid `fluctuations` specified; got '{type(fluctuations)}'"
            raise TypeError(te)
        slp = loglogavgslope
        if isinstance(slp, (tuple, list)):
            slp = jft.prior.normal_prior(*slp)
        elif not callable(slp):
            te = f"invalid `loglogavgslope` specified; got '{type(loglogavgslope)}'"
            raise TypeError(te)

        flx = flexibility
        if isinstance(flx, (tuple, list)):
            flx = jft.prior.lognormal_prior(*flx)
        elif flx is not None and not callable(flx):
            te = f"invalid `flexibility` specified; got '{type(flexibility)}'"
            raise TypeError(te)
        asp = asperity
        if isinstance(asp, (tuple, list)):
            asp = jft.prior.lognormal_prior(*asp)
        elif asp is not None and not callable(asp):
            te = f"invalid `asperity` specified; got '{type(asperity)}'"
            raise TypeError(te)

        fluctuations, spectrum = non_parametric_amplitude(
            self.ks, flu, slp, flx, asp, prefix=name
        )
        self.fluctuations = fluctuations
        self.spectrum = spectrum
        self.weights = weights(self.ks[np.newaxis, :], self.rs[1:, np.newaxis])
        self.norm = norm(self.ks)
        domain = fluctuations.domain | spectrum.domain
        super().__init__(domain=domain, white_init=True, target=callable)

    def normalized_spectrum(self, x):
        return self.spectrum(x)

    def __call__(self, x):
        fluctuations = self.fluctuations(x)
        spectrum = self.spectrum(x)
        spectrum = 0.5 * (spectrum[1:] + spectrum[:-1])
        ker = kernel(spectrum, self.weights, self.norm) * fluctuations**2

        def my_kernel_eval(r):
            res = kernel_eval(r, self.rs, ker)
            res = jnp.where(r == 0.0, res * (1.0 + self.twiddle_factor), res)
            return res

        return my_kernel_eval


"""
#class CorrelatedFieldKernel(jft.Model):
#    def __init__(self, rmin, rmax, nbins, name="kernel"):


# %%
from functools import partial

scale = 0.09
beta = 1.5
def fk(k):
    return 1./(1.+ (k/scale)**2)**beta

rmin = 0.01
rmax = 10000
nbins = 512
rs = r_log_grid(rmin, rmax, nbins)
ks = k_log_grid(rmin, rmax, nbins)
Fk = fk(ks[:-1])
wgts = weights(ks[np.newaxis, :], rs[1:, np.newaxis])
nm = norm(ks)
Fr = kernel(Fk, wgts, nm)
print(Fr.shape)

myker = partial(kernel_eval, rs=rs, Fr=Fr)
# %%
myr = np.linspace(0., rmax, 10000)
import matplotlib.pyplot as plt
Fr = np.array(Fr)
plt.plot(rs, Fr)
plt.plot(myr, myker(myr), '--')
plt.xscale('log')
plt.show()

# %%
import jax
jax.config.update("jax_enable_x64", True)
fluct = (1.,0.4)
slp = (-3., 0.1)
flex = (0.005, 0.00001)
myker = CFKernel(0.01, 10000, 1000, fluct, slp, flex,
                 twiddle_factor=0.01,
                 name="dib")
key = jax.random.PRNGKey(0)

rnds = []
for kk in jax.random.split(key, 10):
    rnds.append(myker.init(kk))
specfunc = [myker(rnd) for rnd in rnds]
spec = [myker.normalized_spectrum(rnd) for rnd in rnds]
ks = myker.ks

for sp in spec:
    plt.plot(ks, sp)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
myr = np.linspace(0., 10000., 1000)
for sp in specfunc:
    sp = jax.jit(sp)
    plt.plot(myr, sp(myr))
#plt.xscale('log')
plt.show()

# %%
"""
