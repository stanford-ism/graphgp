"""
Semidefinite Cholesky decomposition as a JAX primitive.

Pivots that fall below floating point resolution, ``d_j <= n * eps * A_jj``, are dropped: ``L_jj`` and the column
below it are set to zero, so that point carries no information to later rows. This replaces adding jitter to the
diagonal for (near-)duplicate points. Below that scale the computed pivot is dominated by rounding error, so keeping
it would divide by noise.

Derivatives treat the set of dropped pivots as fixed, which is the derivative of the model with those points removed.
The forward pass, its JVP (linear in the matrix tangent), and the VJP (transpose of the JVP) are separate primitives.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

__all__ = ["cholesky", "solve_lower", "solve_lower_transpose", "unit_pivot"]


def cholesky(A):
    """
    Lower Cholesky factor of symmetric positive semidefinite ``A`` of shape ``(..., n, n)``. Only the lower triangle is
    read. Dropped pivots have ``L_jj = 0`` and a zero column below the diagonal.
    """
    return cholesky_p.bind(A)


def unit_pivot(L):
    """Replace zero (dropped) diagonal entries with one, so standard triangular solves skip those points."""
    diag = jnp.diagonal(L, axis1=-2, axis2=-1)
    return L + jnp.where(diag == 0, 1, 0)[..., None] * jnp.eye(L.shape[-1], dtype=L.dtype)


def solve_lower(L, b):
    """Solve ``L x = b`` for ``b`` of shape ``(..., n)``, with ``x_j = 0`` for dropped pivots."""
    x = jax.scipy.linalg.solve_triangular(unit_pivot(L), b[..., None], lower=True)[..., 0]
    return jnp.where(jnp.diagonal(L, axis1=-2, axis2=-1) == 0, 0, x)


def solve_lower_transpose(L, b):
    """Solve ``L^T x = b`` for ``b`` of shape ``(..., n)``, with ``x_j = 0`` for dropped pivots."""
    dropped = jnp.diagonal(L, axis1=-2, axis2=-1) == 0
    b = jnp.where(dropped, 0, b)
    return jax.scipy.linalg.solve_triangular(unit_pivot(L), b[..., None], lower=True, trans=1)[..., 0]


# ===================== Implementations =====================


def _cholesky_impl(A):
    n = A.shape[-1]
    rows = jnp.arange(n)
    diag_A = jnp.diagonal(A, axis1=-2, axis2=-1)
    tol = n * jnp.finfo(A.dtype).eps * diag_A

    def body(j, L):
        # columns >= j of L are still zero, so the matvec only sums over k < j
        s = A[..., :, j] - jnp.einsum("...ik,...k->...i", L, L[..., j, :])
        d = s[..., j]
        keep = d > tol[..., j]
        L_jj = jnp.sqrt(jnp.where(keep, d, 1))
        col = jnp.where(keep[..., None] & (rows > j), s / L_jj[..., None], 0)
        col = jnp.where(rows == j, jnp.where(keep, L_jj, 0)[..., None], col)
        return L.at[..., :, j].set(col)

    return lax.fori_loop(0, n, body, jnp.zeros_like(A))


def _cholesky_jvp_impl(L, dA):
    """Linear in ``dA``. Column-by-column derivative of the recurrence in ``_cholesky_impl``."""
    n = L.shape[-1]
    rows = jnp.arange(n)
    diag_L = jnp.diagonal(L, axis1=-2, axis2=-1)

    def body(j, dL):
        # columns >= j of dL are still zero
        s = (
            dA[..., :, j]
            - jnp.einsum("...ik,...k->...i", L, dL[..., j, :])
            - jnp.einsum("...ik,...k->...i", dL, L[..., j, :])
        )
        L_jj = diag_L[..., j]
        keep = L_jj > 0
        inv = jnp.where(keep, 1 / jnp.where(keep, L_jj, 1), 0)
        dL_jj = s[..., j] * inv / 2
        col = jnp.where(rows > j, (s - L[..., :, j] * dL_jj[..., None]) * inv[..., None], 0)
        col = jnp.where(rows == j, dL_jj[..., None], col)
        return dL.at[..., :, j].set(col)

    return lax.fori_loop(0, n, body, jnp.zeros_like(dA))


def _cholesky_vjp_impl(L, ct):
    """Transpose of ``_cholesky_jvp_impl`` with respect to ``dA``."""
    (dA,) = jax.linear_transpose(partial(_cholesky_jvp_impl, L), L)(ct)
    return dA


def _symmetrize(dA):
    return (dA + jnp.swapaxes(dA, -1, -2)) / 2


# ===================== Primitives =====================

cholesky_p = Primitive("graphgp_cholesky")
cholesky_jvp_p = Primitive("graphgp_cholesky_jvp")
cholesky_vjp_p = Primitive("graphgp_cholesky_vjp")


def _same_shape(x, *_):
    return ShapedArray(x.shape, x.dtype)


for prim, impl, abstract in [
    (cholesky_p, _cholesky_impl, _same_shape),
    (cholesky_jvp_p, _cholesky_jvp_impl, lambda L, dA: _same_shape(dA)),
    (cholesky_vjp_p, _cholesky_vjp_impl, lambda L, ct: _same_shape(ct)),
]:
    prim.def_impl(jax.jit(impl))
    prim.def_abstract_eval(abstract)
    mlir.register_lowering(prim, mlir.lower_fun(impl, multiple_results=False))


def _cholesky_jvp_rule(primals, tangents):
    (A,), (dA,) = primals, tangents
    L = cholesky_p.bind(A)
    if type(dA) is ad.Zero:
        return L, ad.Zero.from_primal_value(L)
    return L, cholesky_jvp_p.bind(L, _symmetrize(dA))


ad.primitive_jvps[cholesky_p] = _cholesky_jvp_rule


def _cholesky_jvp_transpose(ct, L, dA):
    assert not ad.is_undefined_primal(L) and ad.is_undefined_primal(dA)
    if type(ct) is ad.Zero:
        return None, ad.Zero(dA.aval)
    return None, cholesky_vjp_p.bind(L, ct)


def _cholesky_vjp_transpose(ct, L, ct_in):
    assert not ad.is_undefined_primal(L) and ad.is_undefined_primal(ct_in)
    if type(ct) is ad.Zero:
        return None, ad.Zero(ct_in.aval)
    return None, cholesky_jvp_p.bind(L, ct)


ad.primitive_transposes[cholesky_jvp_p] = _cholesky_jvp_transpose
ad.primitive_transposes[cholesky_vjp_p] = _cholesky_vjp_transpose


def _linear_jvp(prim):
    # both linear primitives are linear in their second argument; derivatives with respect to L are not implemented
    def rule(primals, tangents):
        L, x = primals
        dL, dx = tangents
        if type(dL) is not ad.Zero:
            raise NotImplementedError("Second derivatives through graphgp cholesky are not implemented.")
        out = prim.bind(L, x)
        if type(dx) is ad.Zero:
            return out, ad.Zero.from_primal_value(out)
        return out, prim.bind(L, dx)

    return rule


ad.primitive_jvps[cholesky_jvp_p] = _linear_jvp(cholesky_jvp_p)
ad.primitive_jvps[cholesky_vjp_p] = _linear_jvp(cholesky_vjp_p)


def _batch_rule(prim):
    # impls broadcast over leading dimensions, so move all batch axes to the front
    def rule(args, dims):
        size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
        args = [batching.bdim_at_front(a, d, size) for a, d in zip(args, dims)]
        return prim.bind(*args), 0

    return rule


for prim in [cholesky_p, cholesky_jvp_p, cholesky_vjp_p]:
    batching.primitive_batchers[prim] = _batch_rule(prim)
