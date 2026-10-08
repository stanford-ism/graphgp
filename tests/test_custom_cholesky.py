import jax
import jax.numpy as jnp
import jax.random as jr

from graphgp.custom_cholesky import cholesky

jax.config.update("jax_enable_x64", True)

rng = jr.key(7)


def _pd_batch():
    X = jr.normal(rng, (5, 6, 6))
    return X @ X.transpose(0, 2, 1) + 0.1 * jnp.eye(6)


def _degenerate(dtype=jnp.float64):
    # exact duplicates and a near-duplicate below float resolution
    x = jnp.array([0.0, 0.3, 0.3, 1.0, 0.3 + 1e-9, 1.7, 0.0])
    return (0.8 * jnp.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)).astype(dtype)


def test_matches_jnp_on_pd():
    A = _pd_batch()
    dA = jr.normal(jr.key(1), A.shape)
    dA = dA + dA.transpose(0, 2, 1)
    ct = jr.normal(jr.key(2), A.shape)
    assert jnp.allclose(cholesky(A), jnp.linalg.cholesky(A), atol=1e-12)
    assert jnp.allclose(jax.jvp(cholesky, (A,), (dA,))[1], jax.jvp(jnp.linalg.cholesky, (A,), (dA,))[1], atol=1e-12)
    g1 = jax.grad(lambda A: jnp.sum(jnp.tril(cholesky(A)) * ct))(A)
    g2 = jax.grad(lambda A: jnp.sum(jnp.tril(jnp.linalg.cholesky(A)) * ct))(A)
    assert jnp.allclose(g1, g2, atol=1e-12)


def test_degenerate_drops_pivots():
    A = _degenerate()
    L = cholesky(A)
    kept = jnp.diagonal(L) > 0
    assert jnp.array_equal(kept, jnp.array([True, True, False, True, False, True, False]))
    assert jnp.allclose((L @ L.T - A)[kept][:, kept], 0, atol=1e-14)
    assert jnp.all(L[:, ~kept] == 0)  # dropped columns are zero, including the diagonal


def test_degenerate_jvp_vjp_adjoint():
    A = _degenerate()
    dA = jr.normal(jr.key(3), A.shape)
    dA = dA + dA.T
    ct = jnp.tril(jr.normal(jr.key(4), A.shape))
    jvp = jax.jvp(cholesky, (A,), (dA,))[1]
    (vjp,) = jax.vjp(cholesky, A)[1](ct)
    assert jnp.isclose(jnp.sum(ct * jvp), jnp.sum(vjp * dA), rtol=1e-12)


def test_float32_finite():
    A = _degenerate(jnp.float32)
    assert jnp.all(jnp.isnan(jnp.linalg.cholesky(A)[-1]))  # stock Cholesky fails here
    assert jnp.all(jnp.isfinite(cholesky(A)))
    grad = jax.grad(lambda A: jnp.sum(cholesky(A)))(A)
    assert jnp.all(jnp.isfinite(grad))


def test_vmap_jit():
    A = _pd_batch()
    assert jnp.allclose(jax.jit(jax.vmap(cholesky))(A), cholesky(A))
    grad = jax.vmap(jax.grad(lambda A: jnp.sum(cholesky(A))))(A)
    assert jnp.allclose(grad, jax.grad(lambda A: jnp.sum(cholesky(A)))(A))
