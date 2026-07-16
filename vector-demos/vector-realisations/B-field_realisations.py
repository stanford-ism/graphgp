# %%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import graphgp as gp
from jax import Array
import itertools
from pathlib import Path


# %%
def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=True):
    ell2 = ell * ell
    def cov_elem(x, y):
        r = y - x
        if periodic:
            r = r - jnp.round(r)  # assumes coords in [0,1)
        rr2 = jnp.dot(r, r)
        base = jnp.exp(-0.5 * rr2 / ell2)

        I = jnp.eye(3, dtype=base.dtype)
        a = (1.0 - 0.5 * rr2 / ell2) * base
        b = (0.5 / ell2) * base
        return sigma2 * (a * I + b * jnp.outer(r, r))
    return cov_elem

def make_div_free_rbf_covariance(ell, sigma2=1.0, periodic=True, image_sum=True):
    """
    PSD divergence-free kernel in 3D built from RBF scalar kernel.
    sigma2 is the per-component variance at r=0, so K(x,x)=sigma2*I.
    If periodic and image_sum=True, uses 27-image periodic summation (PSD-safe).
    """
    ell2 = ell * ell
    ell4 = ell2 * ell2
    I3 = jnp.eye(3)

    # internal amplitude so that K(0) = sigma2 * I
    # For 3D: K(0) = A * (2/ell^2) I  =>  A = sigma2 * ell^2 / 2
    A = sigma2 * ell2 / 2.0

    shifts = [(0, 0, 0)]
    if periodic and image_sum:
        shifts = list(itertools.product([-1, 0, 1], repeat=3))

    def cov_elem(x, y):
        r0 = y - x  # points assumed in [0,1)^3 if periodic
        K = jnp.zeros((3, 3), dtype=r0.dtype)

        for s in shifts:
            s = jnp.array(s, dtype=r0.dtype)
            r = r0 + s

            # If you prefer min-image instead of image sum:
            if periodic and (not image_sum):
                r = r - jnp.round(r)

            rr2 = jnp.dot(r, r)
            base = jnp.exp(-0.5 * rr2 / ell2)

            a = (2.0 / ell2 - rr2 / ell4) * base
            b = (1.0 / ell4) * base
            K = K + A * (a * I3 + b * jnp.outer(r, r))

        return K
    
    # Attach metadata the CUDA dispatcher will read
    cov_elem._graphgp_cuda = {
        "kernel": "divfree_rbf",
        "ell": float(ell),
        "sigma2": float(sigma2),
        "periodic": bool(periodic),
        "image_sum": bool(image_sum),
    }

    return cov_elem

def divergence_central(B, spacing):
    dx, dy, dz = spacing
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    dBx_dx = (jnp.roll(Bx, -1, axis=0) - jnp.roll(Bx,  1, axis=0)) / (2.0 * dx)
    dBy_dy = (jnp.roll(By, -1, axis=1) - jnp.roll(By,  1, axis=1)) / (2.0 * dy)
    dBz_dz = (jnp.roll(Bz, -1, axis=2) - jnp.roll(Bz,  1, axis=2)) / (2.0 * dz)
    return dBx_dx + dBy_dy + dBz_dz

def divergence_nonperiodic(B, dx):
    # B: (n,n,n,3)
    dBx = (B[2:,1:-1,1:-1,0] - B[:-2,1:-1,1:-1,0]) / (2*dx)
    dBy = (B[1:-1,2:,1:-1,1] - B[1:-1,:-2,1:-1,1]) / (2*dx)
    dBz = (B[1:-1,1:-1,2:,2] - B[1:-1,1:-1,:-2,2]) / (2*dx)
    return dBx + dBy + dBz

# %%
rng = jr.key(0)

# Regular periodic grid of points in [0,1)^3
n = 50 #10
xs = 1*(jnp.arange(n) + 0.5) / n
print(xs)
X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
points = jnp.stack([X, Y, Z], axis=-1).reshape((-1, 3))
N = points.shape[0]

# Build GraphGP dependency graph
graph = gp.build_graph(points, n0=250, k=5) #k=20

n0 = points.shape[0] - graph.neighbors.shape[0]
fine = points[n0:]                       # (N-n0, 3)
coarse = points[graph.neighbors]         # (N-n0, k, 3)
dmax = jnp.max(jnp.linalg.norm(coarse - fine[:, None, :], axis=-1))
print("max neighbor distance:", float(dmax))

ell = (100/n)*0.04
sigma2=1
# Divergence-free covariance element
cov_elem = make_div_free_rbf_covariance(ell=ell, sigma2=sigma2, periodic=False) #0.4

""""
K = gp.compute_cov_matrix_elem(cov_elem, points, points)
K = 0.5 * (K + K.T)
evals = jnp.linalg.eigvalsh(K)
print("min eigenvalue:", float(evals[0]))
print("K finite:", bool(jnp.all(jnp.isfinite(K))))
print("K sym err:", float(jnp.max(jnp.abs(K - K.T))))
print("diag min/max:", float(jnp.min(jnp.diag(K))), float(jnp.max(jnp.diag(K))))
"""

# White parameters xi (N, 3)
rng, key = jr.split(rng)
xi = jr.normal(key, (N, 3))

# Generate vector field values (N, 3)
B = gp.generate_vector(graph, cov_elem, xi, cuda=False, fast_jit=True)

#B = gp.generate_dense_vector(points, cov_elem, xi)

# Reshape to grid and compute diagnostics
Bgrid = B.reshape((n, n, n, 3))
spacing = (1.0 / n, 1.0 / n, 1.0 / n)
dx = spacing[0]
output_dir = Path("temp-figures")
output_dir.mkdir(exist_ok=True)

# %%
# Plot a mid-plane slice
zslice = n // 2
Bx = Bgrid[:, :, zslice, 0]
By = Bgrid[:, :, zslice, 1]
Bz = Bgrid[:, :, zslice, 2]
Babs = jnp.linalg.norm(Bgrid[:, :, zslice, :], axis=-1)

# Quiver of (Bx, By) on the slice
xx = jnp.arange(n)
yy = jnp.arange(n)
XX, YY = jnp.meshgrid(xx, yy, indexing="ij")
plt.figure(figsize=(5, 5))
plt.quiver(XX, YY, Bx, By, scale=50)
plt.title(f"Vector field at z={zslice}")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.savefig(output_dir / "demo_vector_quiver.png")
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
axs[0].imshow(Bx);   axs[0].set_title("B_x")
axs[1].imshow(By);   axs[1].set_title("B_y")
axs[2].imshow(Bz);   axs[2].set_title("B_z")
axs[3].imshow(Babs); axs[3].set_title("|B|")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "demo_vector_fields.png")
plt.show()
