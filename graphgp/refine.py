from typing import Tuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import Array
from jax import lax

import numpy as np

from .graph import Graph

CovElem = Callable[[Array, Array], Array]  # (d,), (d,) -> (ncomp, ncomp)

try:
    import graphgp_cuda

    has_cuda = True
except ImportError:
    has_cuda = False

def generate(
    graph: Graph,
    covariance: Tuple[Array, Array],
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
) -> Array:
    """
    Generate a GP with dense Cholesky for the first layer followed by conditional refinement.
    It is recommended to JIT compile before use.

    Args:
        graph: An instance of ``Graph``, can be checked for validity with ``check_graph``.
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        xi: Unit normal distributed parameters of shape ``(N,).``
        reorder: Whether to reorder parameters and values according to the original order of the points. Default is ``True``.
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer slightly.

    Returns:
        The generated values of shape ``(N,).``
    """
    if len(xi) != len(graph.points):
        raise ValueError("Length of xi must match number of points in graph.")
    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        xi = xi[graph.indices]
    initial_values = generate_dense(graph.points[:n0], covariance, xi[:n0])
    values = refine(
        graph.points, graph.neighbors, graph.offsets, covariance, initial_values, xi[n0:], cuda=cuda, fast_jit=fast_jit
    )
    if graph.indices is not None:
        values = jnp.empty_like(values).at[graph.indices].set(values, unique_indices=True)
    values = jnp.where(jnp.any(jnp.isnan(values)), jnp.full_like(values, jnp.nan), values)
    return values


def generate_dense(points: Array, covariance: Tuple[Array, Array], xi: Array) -> Array:
    """
    Generate a GP with a dense Cholesky decomposition. Note that to compare with the GraphGP values,
    the points must be provided in tree order.

    Args:
        points: Locations of points to model of shape ``(N, d)``
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        xi: Unit normal distributed parameters of shape ``(N,).``
    Returns:
        The generated values of shape ``(N,).``
    """
    if len(xi) != len(points):
        raise ValueError("Length of xi must match number of points.")
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    values = L @ xi
    return values


def refine(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Tuple[Array, Array],
    initial_values: Array,
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
) -> Array:
    """
    Conditionally generate using initial values according to GraphGP algorithm. Most users can use ``generate``, which
    automatically generates the initial values and accepts a ``Graph`` object as input. This function is provided
    if more flexibility is needed, for example to conditionally upsample already generated values.

    It is recommended to JIT compile before use due to internal for loops.

    Args:
        points: Modeled points in tree order of shape ``(N, d)``.
        neighbors: Indices of the neighbors of shape ``(N - offsets[0], k)``.
        offsets: Tuple of length ``B`` representing the end index of each batch.
        covariance: Tuple of arrays (cov_bins, cov_vals) storing discretized covariance. If using your own covariance, inflate k(0) by a small factor to ensure positive definite.
        initial_values: Initial values of shape ``(offsets[0],).``
        xi: Unit normal distributed parameters of shape ``(N - offsets[0],).``
        cuda: Whether to use optional CUDA extension, if installed. Will still use CUDA GPU via JAX if available. Default is ``False`` but recommended if possible for performance.
        fast_jit: Whether to use version of refinement that compiles faster, if cuda=False. Default is ``True`` but runtime performance and memory usage will suffer.

    Returns:
        The refined values of shape ``(N,).``

    """
    n0 = len(points) - len(neighbors)  # should equal offsets[0]
    if len(initial_values) != n0:
        raise ValueError("Length of initial_values must match number of initial points.")
    if len(xi) != len(points) - n0:
        raise ValueError("Length of xi must match number of refined points.")
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        values = graphgp_cuda.refine(
            points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance, initial_values, xi
        )

    elif fast_jit:
        k = neighbors.shape[1]
        max_batch = np.max(np.diff(np.array(offsets)))
        values = jnp.zeros(len(points))
        values = values.at[:n0].set(initial_values)

        # Precompute matrix factorizations for all points
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        L = jnp.linalg.cholesky(K)
        mean_vec = jnp.linalg.solve(L[:, :k, :k].transpose(0, 2, 1), L[:, k, :k][..., None]).squeeze(-1)
        std = L[:, k, k]

        # For each batch defined by offsets, dot neighbor values with mean_vec and add noise
        def step(values, start):
            neighbor_values = values[lax.dynamic_slice(neighbors, (start - n0, 0), (max_batch, k))]
            mean_slice = jnp.sum(lax.dynamic_slice(mean_vec, (start - n0, 0), (max_batch, k)) * neighbor_values, axis=1)
            noise_slice = lax.dynamic_slice(std * xi, (start - n0,), (max_batch,))
            values = lax.dynamic_update_slice(values, mean_slice + noise_slice, (start,))
            return values, None

        values, _ = lax.scan(step, values, jnp.array(offsets[:-1]))

    else:
        values = initial_values
        for i in range(1, len(offsets)):
            start = offsets[i - 1]
            end = offsets[i]
            coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)
            coarse_values = jnp.take(values, neighbors[start - n0 : end - n0], axis=0)
            fine_point = points[start:end]
            fine_xi = xi[start - n0 : end - n0]
            mean, std = jax.vmap(Partial(_conditional_mean_std, covariance))(coarse_points, coarse_values, fine_point)
            values = jnp.concatenate([values, mean + std * fine_xi], axis=0)

    return values


def generate_inv(
    graph: Graph,
    covariance: Tuple[Array, Array],
    values: Array,
    *,
    cuda: bool = False,
) -> Array:
    """
    Inverse of ``generate``. Ensure that the choice for ``reorder`` is the same. Recommended to JIT compile.
    """
    if len(values) != len(graph.points):
        raise ValueError("Length of values must match number of points in graph.")
    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        values = values[graph.indices]
    initial_values, xi = refine_inv(graph.points, graph.neighbors, graph.offsets, covariance, values, cuda=cuda)
    initial_xi = generate_dense_inv(graph.points[:n0], covariance, initial_values)
    xi = jnp.concatenate([initial_xi, xi], axis=0)
    if graph.indices is not None:
        xi = jnp.empty_like(xi).at[graph.indices].set(xi, unique_indices=True)
    xi = jnp.where(jnp.any(jnp.isnan(xi)), jnp.full_like(xi, jnp.nan), xi)
    return xi


def generate_dense_inv(points: Array, covariance: Tuple[Array, Array], values: Array) -> Array:
    """
    Inverse of ``generate_dense``.
    """
    if len(values) != len(points):
        raise ValueError("Length of values must match number of points.")
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    xi = jnp.linalg.solve(L, values)
    return xi


def refine_inv(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Tuple[Array, Array],
    values: Array,
    *,
    cuda: bool = False,
) -> Tuple[Array, Array]:
    """
    Inverse of ``refine``.
    """
    n0 = len(points) - len(neighbors)  # should equal offsets[0]
    if len(values) != len(points):
        raise ValueError("Length of values must match number of points.")
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        initial_values, xi = graphgp_cuda.refine_inv(
            points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance, values
        )
    else:
        k = neighbors.shape[1]
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        L = jnp.linalg.cholesky(K)
        mean_vec = jnp.linalg.solve(L[:, :k, :k].transpose(0, 2, 1), L[:, k, :k][..., None]).squeeze(-1)
        mean = jnp.sum(mean_vec * values[neighbors], axis=1)
        std = L[:, k, k]

        xi = (values[n0:] - mean) / std
        initial_values = values[:n0]
    return initial_values, xi


def generate_logdet(graph: Graph, covariance: Tuple[Array, Array], *, cuda: bool = False) -> Array:
    """
    Log determinant of ``generate``.
    """
    n0 = len(graph.points) - len(graph.neighbors)
    dense_logdet = generate_dense_logdet(graph.points[:n0], covariance)
    return dense_logdet + refine_logdet(graph.points, graph.neighbors, graph.offsets, covariance, cuda=cuda)


def generate_dense_logdet(points: Array, covariance: Tuple[Array, Array]) -> Array:
    """
    Log determinant of ``generate_dense``.
    """
    K = compute_cov_matrix(covariance, points, points)
    return jnp.linalg.slogdet(K)[1] / 2


def refine_logdet(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    covariance: Tuple[Array, Array],
    *,
    cuda: bool = False,
) -> Array:
    """
    Log determinant of ``refine``.
    """
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        logdet = graphgp_cuda.refine_logdet(points, neighbors, jnp.asarray(offsets, dtype=neighbors.dtype), *covariance)
    else:
        n0 = len(points) - len(neighbors)
        k = neighbors.shape[1]
        coarse_points = points[neighbors]
        joint_points = jnp.concatenate([coarse_points, points[n0:, None]], axis=1)
        K = jax.vmap(compute_cov_matrix, in_axes=(None, 0, 0))(covariance, joint_points, joint_points)
        L = jnp.linalg.cholesky(K)
        std = L[:, k, k]
        logdet = jnp.sum(jnp.log(std))
    return logdet


def _conditional_mean_std(covariance, coarse_points, coarse_values, fine_point):
    k = len(coarse_points)
    joint_points = jnp.concatenate([coarse_points, fine_point[jnp.newaxis]], axis=0)
    K = compute_cov_matrix(covariance, joint_points, joint_points)
    L = jnp.linalg.cholesky(K)
    mean = L[k, :k] @ jnp.linalg.solve(L[:k, :k], coarse_values)
    std = L[k, k]
    return mean, std


def compute_cov_matrix(covariance: Tuple[Array, Array], points_a: Array, points_b: Array) -> Array:
    distances = jnp.expand_dims(points_a, -2) - jnp.expand_dims(points_b, -3)
    distances = jnp.linalg.norm(distances, axis=-1)
    if isinstance(covariance, Tuple) and isinstance(covariance[0], Array) and isinstance(covariance[1], Array):
        cov_bins, cov_vals = covariance
        return cov_lookup(distances, cov_bins, cov_vals)
    else:
        raise ValueError("Invalid covariance specification.")


def cov_lookup(r, cov_bins, cov_vals):
    """
    Look up covariance in array of sampled `cov_vals` at radii `cov_bins` (equal-sized arrays).
    If `r` is inside of bounds, a linearly interpolated value is returned.
    If `r` is below the first bin, the first value is returned. But really the first bin should always be 0.0.
    If `r` is above the last bin, the last value is returned. Maybe the last value should be zero.
    """
    return jnp.interp(r, cov_bins, cov_vals)


# =========================
# Vector-valued extensions
# =========================


def _get_cuda_vector_kernel_params(cov_elem: CovElem, *, jitter: float, dtype, int_dtype):
    """
    Extract parameters for the CUDA vector-field kernel from a covariance callable.

    The CUDA path cannot call an arbitrary Python ``cov_elem`` from device code.
    It therefore supports the 3-component divergence-free RBF tensor whose
    parameters are carried as lightweight metadata on the Python callable.
    """
    meta = getattr(cov_elem, "_graphgp_cuda", None)
    if isinstance(meta, dict):
        kernel = meta.get("kernel")
        if kernel not in ("divfree_rbf", "div_free_rbf_vector_3"):
            kernel = None
        if bool(meta.get("image_sum", False)):
            raise NotImplementedError(
                "cuda=True for vector fields supports the min-image div-free RBF kernel, "
                "but not the 27-image periodic sum variant yet."
            )
        ell = meta.get("ell")
        sigma2 = meta.get("sigma2", 1.0)
        periodic = meta.get("periodic", True)
    else:
        kernel = getattr(cov_elem, "_graphgp_cuda_kernel", None)
        ell = getattr(cov_elem, "_graphgp_cuda_ell", None)
        sigma2 = getattr(cov_elem, "_graphgp_cuda_sigma2", 1.0)
        periodic = getattr(cov_elem, "_graphgp_cuda_periodic", True)

    if kernel != "div_free_rbf_vector_3" and kernel != "divfree_rbf":
        raise NotImplementedError(
            "cuda=True for vector fields currently supports only the magnetic "
            "3x3 divergence-free RBF tensor created by "
            "graphgp.extras.make_div_free_rbf_covariance(...), or an equivalent "
            "callable carrying _graphgp_cuda metadata."
        )
    if ell is None:
        raise ValueError("CUDA vector covariance metadata is missing the correlation length 'ell'.")

    params = jnp.asarray([float(ell), float(sigma2), float(jitter), 1.0 if bool(periodic) else 0.0], dtype=dtype)
    return params.astype(dtype), jnp.asarray(3, dtype=int_dtype)


def compute_cov_matrix_elem(cov_elem: CovElem, points_a: Array, points_b: Array) -> Array:
    """
    Build the *block* covariance matrix for a vector-valued GP.

    Args:
        cov_elem: callable cov_elem(x, y) -> (ncomp, ncomp)
        points_a: (Na, d)
        points_b: (Nb, d)

    Returns:
        Block covariance of shape (Na*ncomp, Nb*ncomp)
    """
    if points_a.ndim != 2 or points_b.ndim != 2:
        raise ValueError("points_a and points_b must have shape (N, d)")

    # blocks[a, b] = cov_elem(points_a[a], points_b[b])  -> (Na, Nb, ncomp, ncomp)
    blocks = jax.vmap(
        lambda xa: jax.vmap(lambda yb: cov_elem(xa, yb), in_axes=0, out_axes=0)(points_b),
        in_axes=0,
        out_axes=0,
    )(points_a)

    ncomp = blocks.shape[-1]
    # reshape (Na, Nb, ncomp, ncomp) -> (Na*ncomp, Nb*ncomp)
    blocks = jnp.transpose(blocks, (0, 2, 1, 3))  # (Na, ncomp, Nb, ncomp)
    return blocks.reshape((points_a.shape[0] * ncomp, points_b.shape[0] * ncomp))


def make_separable_vector_covariance(
    covariance: Tuple[Array, Array],
    component_cov: Array,
) -> CovElem:
    """
    Convenience: lift a scalar, radial GraphGP covariance (bins/vals) to a vector covariance
    via a *constant* component covariance matrix C:

        Cov(v_i(x), v_j(y)) = k(|x-y|) * C_ij
    """
    cov_bins, cov_vals = covariance
    C = jnp.asarray(component_cov)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("component_cov must be a square (ncomp, ncomp) matrix")

    def cov_elem(x: Array, y: Array) -> Array:
        r = jnp.linalg.norm(x - y)
        k = cov_lookup(r, cov_bins, cov_vals)
        return k * C

    return cov_elem


def generate_vector(
    graph: Graph,
    cov_elem: CovElem,
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
) -> Array:
    """
    Vector-valued analogue of `generate`:

    - `xi` has shape (N, ncomp)
    - returns values of shape (N, ncomp)
    - `cov_elem(x, y)` returns (ncomp, ncomp)
    """
    if xi.ndim != 2:
        raise ValueError(f"xi must have shape (N, ncomp). Got {xi.shape}.")
    if xi.shape[0] != len(graph.points):
        raise ValueError("Leading dimension of xi must match number of points in graph.")

    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        xi = xi[graph.indices]

    initial_values = generate_dense_vector(graph.points[:n0], cov_elem, xi[:n0])
    values = refine_vector(
        graph.points,
        graph.neighbors,
        graph.offsets,
        cov_elem,
        initial_values,
        xi[n0:],
        cuda=cuda,
        fast_jit=fast_jit,
    )

    if graph.indices is not None:
        values = jnp.empty_like(values).at[graph.indices].set(values, unique_indices=True)
    values = jnp.where(jnp.any(jnp.isnan(values)), jnp.full_like(values, jnp.nan), values)
    return values


def generate_dense_vector(points: Array, cov_elem: CovElem, xi: Array) -> Array:
    """
    Dense Cholesky generation for vector-valued GP on `points` (in tree order).

    points: (N, d)
    xi:     (N, ncomp)
    returns (N, ncomp)
    """
    if xi.ndim != 2:
        raise ValueError(f"xi must have shape (N, ncomp). Got {xi.shape}.")
    if xi.shape[0] != len(points):
        raise ValueError("Leading dimension of xi must match number of points.")

    K = compute_cov_matrix_elem(cov_elem, points, points)  # (N*ncomp, N*ncomp)
    L = jnp.linalg.cholesky(K)
    vals = L @ xi.reshape((-1,))
    return vals.reshape((len(points), xi.shape[1]))


def _local_conditional_mats_vector(cov_elem: CovElem, joint_points: Array, *, k: int, ncomp: int, jitter: float):
    """
    joint_points: (k+1, d)
    returns:
      R:      (ncomp, k*ncomp)  so mean = R @ vec(coarse_values)
      cholD:  (ncomp, ncomp)    so noise = cholD @ xi_f
    """
    K = compute_cov_matrix_elem(cov_elem, joint_points, joint_points)
    m = k * ncomp
    Kcc = K[:m, :m]
    Kcf = K[:m, m:]       # (m, ncomp)
    Kff = K[m:, m:]       # (ncomp, ncomp)

    # Solve Kcc * A = Kcf  => A = Kcc^{-1} Kcf
    A = jnp.linalg.solve(Kcc, Kcf)              # (m, ncomp)
    D = Kff - Kcf.T @ A                         # (ncomp, ncomp)
    if jitter != 0.0:
        D = D + jitter * jnp.eye(ncomp, dtype=D.dtype)
    cholD = jnp.linalg.cholesky(D)
    R = A.T                                     # (ncomp, m)
    return R, cholD


def refine_vector(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    cov_elem: CovElem,
    initial_values: Array,
    xi: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
    jitter: float = 0.0,
) -> Array:
    """
    Vector-valued analogue of `refine`.

    points:         (N, d) in tree/depth order
    neighbors:      (N-n0, k)
    initial_values: (n0, ncomp)
    xi:             (N-n0, ncomp)
    returns:        (N, ncomp)
    """
    n0 = len(points) - len(neighbors)
    if initial_values.shape[0] != n0:
        raise ValueError("initial_values must have leading dimension n0.")
    if initial_values.ndim != 2:
        raise ValueError("initial_values must have shape (n0, ncomp).")
    if xi.ndim != 2 or xi.shape[0] != (len(points) - n0):
        raise ValueError("xi must have shape (N-n0, ncomp).")

    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        params, ncomp_cuda = _get_cuda_vector_kernel_params(
            cov_elem, jitter=jitter, dtype=points.dtype, int_dtype=neighbors.dtype
        )
        return graphgp_cuda.refine_vector(
            points,
            neighbors,
            jnp.asarray(offsets, dtype=neighbors.dtype),
            initial_values,
            xi,
            params,
        )

    k = int(neighbors.shape[1])
    ncomp = int(initial_values.shape[1])

    if fast_jit:
        import numpy as np

        max_batch = int(np.max(np.diff(np.array(offsets))))

        values = jnp.zeros((len(points), ncomp), dtype=initial_values.dtype)
        values = values.at[:n0, :].set(initial_values)

        coarse_points = points[neighbors]                         # (N-n0, k, d)
        joint_points = jnp.concatenate([coarse_points, points[n0:, None, :]], axis=1)  # (N-n0, k+1, d)

        # Precompute R and cholD for all refined points
        local = jax.vmap(
            lambda jp: _local_conditional_mats_vector(cov_elem, jp, k=k, ncomp=ncomp, jitter=jitter),
            in_axes=0,
            out_axes=(0, 0),
        )
        R_all, cholD_all = local(joint_points)                    # (N-n0, ncomp, k*ncomp), (N-n0, ncomp, ncomp)

        m = k * ncomp

        def step(values, start):
            i0 = start - n0

            neigh = lax.dynamic_slice(neighbors, (i0, 0), (max_batch, k))                 # (B, k)
            neigh_vals = values[neigh]                                                    # (B, k, ncomp)
            cvec = neigh_vals.reshape((max_batch, m))                                      # (B, m)

            R = lax.dynamic_slice(R_all, (i0, 0, 0), (max_batch, ncomp, m))               # (B, ncomp, m)
            mean = jnp.einsum("bim,bm->bi", R, cvec)                                       # (B, ncomp)

            cholD = lax.dynamic_slice(cholD_all, (i0, 0, 0), (max_batch, ncomp, ncomp))   # (B, ncomp, ncomp)
            xi_slice = lax.dynamic_slice(xi, (i0, 0), (max_batch, ncomp))                 # (B, ncomp)
            noise = jnp.einsum("bij,bj->bi", cholD, xi_slice)                              # (B, ncomp)

            out_slice = mean + noise
            values = lax.dynamic_update_slice(values, out_slice, (start, 0))
            return values, None

        values, _ = lax.scan(step, values, jnp.array(offsets[:-1]))
        return values

    # Non-fast path (python loop, mirrors scalar implementation’s slow branch)
    values = initial_values
    for i in range(1, len(offsets)):
        start = offsets[i - 1]
        end = offsets[i]
        coarse_points = jnp.take(points, neighbors[start - n0 : end - n0], axis=0)    # (B,k,d)
        coarse_values = jnp.take(values, neighbors[start - n0 : end - n0], axis=0)    # (B,k,ncomp)
        fine_points = points[start:end]                                               # (B,d)
        fine_xi = xi[start - n0 : end - n0]                                           # (B,ncomp)

        def one(cp, cv, fp, z):
            jp = jnp.concatenate([cp, fp[None, :]], axis=0)
            R, cholD = _local_conditional_mats_vector(cov_elem, jp, k=k, ncomp=ncomp, jitter=jitter)
            mean = R @ cv.reshape((-1,))
            return mean + cholD @ z

        batch_vals = jax.vmap(one)(coarse_points, coarse_values, fine_points, fine_xi)  # (B,ncomp)
        values = jnp.concatenate([values, batch_vals], axis=0)
    return values


def generate_vector_inv(
    graph: Graph,
    cov_elem: CovElem,
    values: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
    jitter: float = 0.0,
) -> Array:
    """
    Inverse of `generate_vector`. Returns xi with shape (N, ncomp).
    """
    if values.ndim != 2:
        raise ValueError("values must have shape (N, ncomp).")
    if values.shape[0] != len(graph.points):
        raise ValueError("Leading dimension of values must match number of points in graph.")

    n0 = len(graph.points) - len(graph.neighbors)
    if graph.indices is not None:
        values = values[graph.indices]

    initial_values, xi_ref = refine_vector_inv(
        graph.points, graph.neighbors, graph.offsets, cov_elem, values,
        cuda=cuda, fast_jit=fast_jit, jitter=jitter
    )
    xi0 = generate_dense_vector_inv(graph.points[:n0], cov_elem, initial_values)
    xi = jnp.concatenate([xi0, xi_ref], axis=0)

    if graph.indices is not None:
        xi = jnp.empty_like(xi).at[graph.indices].set(xi, unique_indices=True)
    xi = jnp.where(jnp.any(jnp.isnan(xi)), jnp.full_like(xi, jnp.nan), xi)
    return xi


def generate_dense_vector_inv(points: Array, cov_elem: CovElem, values: Array) -> Array:
    if values.ndim != 2:
        raise ValueError("values must have shape (N, ncomp).")
    if values.shape[0] != len(points):
        raise ValueError("Leading dimension of values must match number of points.")
    K = compute_cov_matrix_elem(cov_elem, points, points)
    L = jnp.linalg.cholesky(K)
    xi = jnp.linalg.solve(L, values.reshape((-1,)))
    return xi.reshape(values.shape)


def refine_vector_inv(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    cov_elem: CovElem,
    values: Array,
    *,
    cuda: bool = False,
    fast_jit: bool = True,
    jitter: float = 0.0,
) -> Tuple[Array, Array]:
    """
    Inverse of `refine_vector`. Returns (initial_values, xi_ref).
    """
    n0 = len(points) - len(neighbors)
    if values.ndim != 2 or values.shape[0] != len(points):
        raise ValueError("values must have shape (N, ncomp).")
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        params, ncomp_cuda = _get_cuda_vector_kernel_params(
            cov_elem, jitter=jitter, dtype=points.dtype, int_dtype=neighbors.dtype
        )
        return graphgp_cuda.refine_vector_inv(
            points,
            neighbors,
            jnp.asarray(offsets, dtype=neighbors.dtype),
            values,
            params,
        )

    k = int(neighbors.shape[1])
    ncomp = int(values.shape[1])
    m = k * ncomp

    coarse_points = points[neighbors]
    joint_points = jnp.concatenate([coarse_points, points[n0:, None, :]], axis=1)  # (N-n0, k+1, d)

    local = jax.vmap(
        lambda jp: _local_conditional_mats_vector(cov_elem, jp, k=k, ncomp=ncomp, jitter=jitter),
        in_axes=0,
        out_axes=(0, 0),
    )
    R_all, cholD_all = local(joint_points)  # (N-n0, ncomp, m), (N-n0, ncomp, ncomp)

    neigh_vals = values[neighbors]                           # (N-n0, k, ncomp)
    cvec = neigh_vals.reshape((len(neigh_vals), m))          # (N-n0, m)
    mean = jnp.einsum("bim,bm->bi", R_all, cvec)             # (N-n0, ncomp)

    resid = values[n0:] - mean                               # (N-n0, ncomp)
    xi = jax.vmap(lambda L, r: jnp.linalg.solve(L, r))(cholD_all, resid)
    return values[:n0], xi


def generate_vector_logdet(
    graph: Graph,
    cov_elem: CovElem,
    *,
    cuda: bool = False,
    jitter: float = 0.0,
) -> Array:
    n0 = len(graph.points) - len(graph.neighbors)
    return generate_dense_vector_logdet(graph.points[:n0], cov_elem) + refine_vector_logdet(
        graph.points, graph.neighbors, graph.offsets, cov_elem, cuda=cuda, jitter=jitter
    )


def generate_dense_vector_logdet(points: Array, cov_elem: CovElem) -> Array:
    K = compute_cov_matrix_elem(cov_elem, points, points)
    L = jnp.linalg.cholesky(K)
    return jnp.sum(jnp.log(jnp.diagonal(L)))


def refine_vector_logdet(
    points: Array,
    neighbors: Array,
    offsets: Tuple[int, ...],
    cov_elem: CovElem,
    *,
    cuda: bool = False,
    jitter: float = 0.0,
) -> Array:
    if cuda:
        if not has_cuda:
            raise ImportError("CUDA extension not installed, cannot use cuda=True.")
        params, ncomp_cuda = _get_cuda_vector_kernel_params(
            cov_elem, jitter=jitter, dtype=points.dtype, int_dtype=neighbors.dtype
        )
        return graphgp_cuda.refine_vector_logdet(
            points,
            neighbors,
            jnp.asarray(offsets, dtype=neighbors.dtype),
            params,
        )

    n0 = len(points) - len(neighbors)
    k = int(neighbors.shape[1])

    # Infer ncomp by evaluating once on first point-pair
    C00 = cov_elem(points[0], points[0])
    ncomp = int(C00.shape[0])

    coarse_points = points[neighbors]
    joint_points = jnp.concatenate([coarse_points, points[n0:, None, :]], axis=1)
    _, cholD_all = jax.vmap(
        lambda jp: _local_conditional_mats_vector(cov_elem, jp, k=k, ncomp=ncomp, jitter=jitter),
        in_axes=0,
        out_axes=(0, 0),
    )(joint_points)
    return jnp.sum(jnp.log(jnp.diagonal(cholD_all, axis1=-2, axis2=-1)))
