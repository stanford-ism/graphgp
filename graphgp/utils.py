import jax
import jax.numpy as jnp


def _clip_v(v, A):
    info = jnp.finfo(A.dtype)
    tol = A.shape[0] * info.eps * jnp.linalg.norm(A, ord=2)
    return jnp.clip(v, tol, None)


def _get_sqrt(v, U):
    vsq = jnp.sqrt(v)
    return U @ (vsq[:, jnp.newaxis] * U.T)


@jax.custom_jvp
def _sqrtm(M):
    v, U = jnp.linalg.eigh(M)
    v = _clip_v(v, M)
    return _get_sqrt(v, U)


@_sqrtm.defjvp
def _sqrtm_jvp(M, dM):
    # Note: Only stable 1st derivative!
    M, dM = M[0], dM[0]
    v, U = jnp.linalg.eigh(M)
    v = _clip_v(v, M)

    dM = U.T @ dM @ U
    vsq = jnp.sqrt(v)
    dres = dM / (vsq[:, jnp.newaxis] + vsq[jnp.newaxis, :])
    return _get_sqrt(v, U), U @ dres @ U.T
