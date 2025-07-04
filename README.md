# GraphGP

GraphGP... \
✓ generates approximate Gaussian process realizations for stationary, decaying kernels \
✓ scales to billions of points in linear time and with minimal memory overhead \
✓ effortlessly handles points at arbitrary locations with many orders of magnitude of dynamic range \
✓ uses JAX, with an optional CUDA extension that is recommended for large problems \
✓ supports forward and backward differentiation of both realization and kernel parameters \
✓ has an exact inverse and determinant available

The underlying theory and implementation is described in two papers ([1], [2]). It is an evolution of Iterative Charted Refinement [3], which was first implemented in the NIFTy package [4].

We wrote this package for our applications in astrophysics, but we hope others across the physical sciences will find it useful! Please do not hesitate to open an issue or discussion for questions or feedback :)

## Installation
To install, use pip. The only dependency is JAX. For large problems, it is recommended to install with the CUDA extension as shown below, which will require CMake and the CUDA compiler (nvcc) installed on your system and take some additional time to build. Please let us know if you run into issues there.
```
python -m pip install graphgp
python -m pip install graphgp[cuda]
```

## Usage

```python
import jax
import jax.random as jr
import graphgp as gg

kp, kx = jr.split(jr.key(100))
points = jr.normal(kp, shape=(1_000_000, 2))
xi = jr.normal(kx, shape=(1_000_000,))

graph = gg.build_graph(points, n_initial=1000, k=8, cuda=True)
covariance = gg.matern_covariance_discrete(r_min=1e-3, r_max=1e1, n=100)
values = gg.generate(graph, covariance, xi, cuda=True)
```

## Q & A

_Can I use this with NIFTy, blackjax, numpyro, etc?_ \
Yes! It should be easy to plug GraphGP into any JAX-based inference framework. We recommend MGVI and geoVI as implemented in NIFTy as probably the only algorithms capable of handling inference at scale.

_How fast is it?_ \
This depends a lot on the hardware, but we get _ points per second on an H100 GPU and _ points per second on a laptop with _. The runtime is dominated by the random memory access needed to load values of conditioning points. Computation is comparatively cheap and parallel enough to easily saturate any GPU for large problems.

_What are the memory requirements for $N$ points in $D$ dimensions using $k$ neighbors?_ \
The graph requires $ND$ floats to store the point locations, $Nk$ integers to store the indices of neighbors, and $N$ integers to store the original point order. The CUDA version of `gg.generate` has essentially no memory overhead, it only requires $N$ floats for the white noise parameters and $N$ floats for the output values. This is achieved by generating small refinement matrices on the fly and storing them only in registers. There is also a small dense matrix for the initial points and two arrays for the covariance, but these are typically negligible. Derivatives will typically need $2N$ additional floats for the parameter and value tangents. In a realistic workflow, one probably needs space for a full copy of all of these arrays, though it could be avoided with care. For extremely memory-constrained settings, it is possible to reduce the footprint of the graph. If the points lie on a geometric grid, their locations can be computed on the fly from a single integer index. This also happens to be significantly faster than loading points from memory. Nearest neighbors can also be queried on the fly to eliminate the neighbor index array, although with a performance hit.

_Is the tree order optimal?_ \
No. The optimal order will depend on the distribution of points, and is probably intractable to determine in general. The tree order is a "good guess" for arbitrary point distributions which also happens to be necessary for efficient neighbor queries. See the software paper for more discussion. If you have a better idea, feel free to construct your own graph! We make it transparent and offer a helper function to ensure your graph is valid.

_Can I differentiate with respect to the locations of points?_ \
Yes and no. It is our eventual goal to enable adaptive resolution by optimizing the locations of points. However, the simplest version of this results in points collapsing on each other as a local minimum, which needs to be solved first. Additionally, as points move, it is critical that the topological graph order remain the same so that the parameters retain their meaning. But our tree-based order will certainly change, and querying preceding neighbors on an arbitrary order of points is challenging, especially if we have to do it frequently as points move. For these reasons, the CUDA version of the code does not currently support differentiation with respect to points. When a working approach is developed in JAX, we will happily fill in the CUDA implementation.

_Can I distribute the GP across multiple GPUs?_ \
Please try! With some understanding of the structure, it's not too difficult to split a graph across multiple machines. The main challenge is how to build the graph in the first place if it cannot fit in on a single machine. Our automatic tools require global sorts which are not easily distributed. Instead, users must rely on knowledge of their point distribution to pre-determine the division of points among machines, after which the automatic tools can be used. There is an example notebook demonstrating this for for a healpix spherical grid.

_Is there TPU support?_ \
Only via the pure JAX version. It would be awesome to have a custom TPU implementation as well. Perhaps if Google offers us tons of free TPUs to make the world's largest dust map we'll put in the effort ;)