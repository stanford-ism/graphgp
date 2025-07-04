<p align="center">
  <img src="logo.png" alt="Project Logo" height="150" />
</p>

"Gaussian processes don't scale," they said.

***GraphGP***... \
✅ generates Gaussian process realizations with approximately stationary, decaying kernels \
✅ scales to billions of points in linear time and with minimal memory overhead \
✅ effortlessly handles points at arbitrary locations with many orders of magnitude of dynamic range \
✅ uses JAX, with a optional CUDA extension that is recommended for large problems \
✅ supports forward and backward mode differentiation of both white noise and kernel parameters \
✅ has an exact inverse and determinant available

The underlying theory and implementation is described in two papers ([1], [2]). It is an evolution of Iterative Charted Refinement [3], which was first implemented in the NIFTy package [4].

We wrote this software for applications in astrophysics, but we hope others across the physical sciences will find it useful! Please do not hesitate to open an issue or discussion for questions and feedback :)

## Installation
To install, use pip. The only dependency is JAX. For large problems, it is recommended to install with the CUDA extension as shown below, which will require CMake and the CUDA compiler (nvcc) installed on your system and take some additional time to build. Please let us know if you run into issues there.
```
python -m pip install graphgp
python -m pip install graphgp[cuda]
```

## Usage

```python
import jax
import graphgp as gp

kp, kx = jax.random.split(jax.random.key(99))
points = jax.random.normal(kp, shape=(1_000_000, 2))
xi = jax.random.normal(kx, shape=(1_000_000,))

graph = gp.build_graph(points, n_initial=1000, k=8, cuda=True)
covariance = gp.matern_covariance_discrete(r_min=1e-3, r_max=20, n=100)
values = gp.generate(graph, covariance, xi, cuda=True)
```

## Q&A

*How does it work?* \
The most straightforward way to generate a GP at $N$ arbitrary points is to construct the dense $N\times N$ covariance matrix, compute a matrix square root via Cholesky decomposition, and then apply it to a vector of white noise. Alternatively, but equivalently, we can generate each value sequentially, conditioning on all previously generated values using the Gaussian conditioning formulas. The main approximation we make is to condition only on the values of *k* previously generated neighbors, rather than up to $N$. There are a couple of subtleties to mention:
 1. Sequential generation is slow. Observe that the nearest-neighbor relations define a directed acyclic graph. We can compute the depth of each point in the graph, and generate values for all points at the same level in parallel. Alternatively, 
 2. Computing the *k* preceding neighbors for all points in an arbitrary order is difficult. Thus, we use a k-d tree order.
 3. The order of points will affect the accuracy of the approximation. Conveniently, the k-d tree order works well as it distributes points "coarse" to "fine".

For more details, see the software paper

*How accurate is it?* \
This is dependent on many factors. First, note that the output is exactly linear in the white noise input, and thus exactly Gaussian. The covariance kernel is approximated such that some pairs of points may have slightly more or less covariance than the kernel prescribes. For context, we typically use GPs in settings where sub-percent uncertainty is unnecessary, and even then only as a prior. Errors from approximate inference are often larger than errors in the GP itself. The design of the software reflects this, with small variability between different implementations that may be greater than machine precision. For `strict` graphs, *k* is the only parameter to tune. For non-`strict` graphs, the expansion factor can also be tuned. See the software paper for more quantitative metrics.

*How fast is it?* \
This depends a lot on the hardware, but we get _ points per second on an H100 GPU and _ points per second on a laptop with _. The runtime is dominated by the random memory access needed to load values of conditioning points. Computation is comparatively cheap and parallel enough to easily saturate any GPU for large problems.

*What are the memory requirements for $N$ points in $D$ dimensions using $k$ neighbors?* \
The graph requires $ND$ floats to store the point locations, $Nk$ integers to store the indices of neighbors, and $N$ integers to store the original point order. The CUDA version of `gg.generate` has essentially no memory overhead, it only requires $N$ floats for the white noise parameters and $N$ floats for the output values. This is achieved by generating small refinement matrices on the fly and storing them only in registers. There is also a small dense matrix for the initial points and two arrays for the covariance, but these are typically negligible. Derivatives will typically need $2N$ additional floats for the parameter and value tangents. In a realistic workflow, one probably needs space for a full copy of all of these arrays, though it could be avoided with care. For extremely memory-constrained settings, it is possible to reduce the footprint of the graph. If the points lie on a geometric grid, their locations can be computed on the fly from a single integer index. This also happens to be significantly faster than loading points from memory. Nearest neighbors can also be queried on the fly to eliminate the neighbor index array, although with a performance hit.

*Can I use this with NIFTy, blackjax, numpyro, etc?* \
Yes! It should be easy to plug GraphGP into any JAX-based inference framework. We recommend MGVI and geoVI as implemented in NIFTy as probably the only algorithms capable of handling inference at scale.

*Is the tree order optimal?* \
No. The optimal order will depend on the distribution of points, and is probably intractable to determine in general. The tree order is a "good guess" for arbitrary point distributions which also happens to be necessary for efficient neighbor queries. See the software paper for more discussion. If you have a better idea, feel free to construct your own graph! We make it transparent and offer a helper function to ensure your graph is valid.

*Can I differentiate with respect to the locations of points?* \
Yes and no. It is our eventual goal to enable adaptive resolution by optimizing the locations of points. However, the simplest version of this results in points collapsing on each other as a local minimum, which needs to be solved first. Additionally, as points move, it is critical that the topological graph order remain the same so that the parameters retain their meaning. But our tree-based order will certainly change, and querying preceding neighbors on an arbitrary order of points is challenging, especially if we have to do it frequently as points move. For these reasons, the CUDA version of the code does not currently support differentiation with respect to points. When a working approach is developed in JAX, we will happily fill in the CUDA implementation.

*Can I use a non-stationary kernel?* \
Mild anisotropy or inhomogeneity should be fine as far as the algorithm is concerned. The key question is whether or not the nearest neighbors in space are ones with roughly the largest influence. (In principle, one could try to find neighbors in kernel space, see the theory paper for a discussion.) We have not yet implemented any non-stationary kernels, nor is there a convenient discretization that will work for general non-stationary kernels. If you have a specific kernel you wish to use, please submit an issue and we will consider it. Adding kernels to the JAX implementation should be straightforward.

*Can I distribute the GP across multiple GPUs?* \
Please try! With some understanding of the structure, it's not too difficult to split a graph across multiple machines. The main challenge is how to build the graph in the first place if it cannot fit in on a single machine. Our automatic tools require global sorts which are not easily distributed. Instead, users must rely on knowledge of their point distribution to pre-determine the division of points among machines, after which the automatic tools can be used. There is an example notebook demonstrating this for for a healpix spherical grid.

*Is there TPU support?* \
Only via the pure JAX version. It would be awesome to have a custom TPU implementation as well. Perhaps if Google offers us tons of free TPUs to make the world's largest dust map we'll put in the effort ;)

*Who made this?* \
Insert some note about contributions. Mention NIFTy, KIPAC, all authors of papers.

If you use this software in your work, please cite both the software and theory papers.