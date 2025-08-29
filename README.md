<p align="center">
  <img src="logo.png" alt="Project Logo" height="150" />
</p>

"Gaussian processes don't scale."

***GraphGP***... \
✅ generates Gaussian process realizations with approximately stationary, decaying kernels \
✅ scales to billions of parameters with linear time and memory requirements \
✅ effortlessly handles arbitrary point distributions with large dynamic range \
✅ uses JAX, with a faster CUDA extension that also supports derivatives \
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

graph = gp.build_graph(points, n_initial=1000, k=8)
covariance = gp.matern_covariance_discrete(r_min=1e-3, r_max=20, n=100)
values = gp.generate(graph, covariance, xi)
```

## Q&A

*How does it work?* \
The most straightforward way to generate a GP at *N* arbitrary points is to construct a dense *N x N* covariance matrix, compute a matrix square root via Cholesky decomposition, and then apply it to a vector of white noise. This can be interpeted as generating each value sequentially, conditioned on all previously generated values using the Gaussian conditioning formulas. The main approximation made in GraphGP is to condition only on the values of *k* previously generated neighbors, rather than up to *N*. When *N* is billions and *k* is eight, this makes a big difference. There are a couple of subtleties to mention:
 1. The order of points has a significant effect on the accuracy of the approximation. We use a modified k-d tree order which can be seen as first modeling "coarse" long-range correlations and then filling in the "fine" details. This is where the assumption of a decaying kernel is important.
 2. Finding the *k* preceding neighbors for points in an arbitrary order is expensive. Fortunately, the points are in k-d tree order so queries are fast.
 3. Sequential generation is slow, so we provide two methods for parallelization. The more accurate `strict=True` method considers the directed acyclic graph implied by the nearest neighbor relations. We can compute the depth of each point in the graph and generate values at the same level in parallel. The size of the levels depends strongly on the order of points, hence our modifications to the standard k-d tree order. The approximate `strict=False` method works for any user-specified levels. If a point depends on another point at the same level, it generates an approximate value on the fly. This approach can be quite accurate with our special order. It's faster to set up and the level sizes don't depend on the point distribution (a big factor for JIT compilation), so this is the default.

*How accurate is it?* \
This is dependent on many factors. First, note that the output is exactly linear in the white noise input, and thus exactly Gaussian. The covariance kernel is approximated such that some pairs of points may have slightly more or less covariance than the kernel prescribes, often at the _ % level. For context, our applications often do not require sub-percent accuracy in the GP prior since the data is not constraining to that level and errors from approximate inference are often larger than errors in the prior itself. The design of the software reflects this, with small variability between different implementations that may be greater than machine precision. For `strict` graphs, *k* is the only accuracy parameter to tune. For non-`strict` graphs, the expansion factor can also be tuned. See the software paper for more quantitative metrics.

*How fast is it?* \
This depends a lot on the hardware, but we get _ points per second on an H100 GPU and _ points per second on a laptop with _. The runtime is dominated by the random memory access needed to load values of conditioning points. Computation is comparatively cheap and parallel enough to easily saturate any GPU for large problems.

*What are the memory requirements for *N* points in *D* dimensions using *k* neighbors?* \
The graph requires *ND* floats to store the point locations, *Nk* integers to store the indices of neighbors, and *N* integers to store the original point order. The CUDA version of `generate` has essentially no memory overhead, it only requires *N* floats for the white noise parameters and *N* floats for the output values. This is achieved by generating small refinement matrices on the fly in registers so they never have to be written to main memory. There is also a small dense matrix for the initial points and two arrays for the covariance, but these are typically negligible. Derivatives will typically need *2N* additional floats for the parameter and value tangents. In a realistic workflow, one probably needs space for a full copy of all of these arrays, though it could be avoided with care. For extremely memory-constrained settings, it is possible to reduce the footprint of the graph. If the points lie on a geometric grid, their locations can be computed on the fly from a single integer index. This also happens to be significantly faster than loading points from memory. Nearest neighbors can also be queried on the fly to eliminate the neighbor index array, although with a performance hit.

*What is the main difference between the JAX and CUDA versions?* \
The JAX version must store refinement matrices in main memory. The CUDA version generates them on the fly. This can yield 30x improvement in both speed and memory efficiency for large problems. Graph building is also dramatically faster in the CUDA version, though this is less relevant as it only happens once.

*What are the main differences between GraphGP and NIFTy Iterative Charted Refinement?* \
Rather than requiring a coordinate chart which maps the point distribution to a Euclidean grid, we provide an automatic way to generate the refinement structure based on a tree order and nearest-neighbor relations. This promises to scale better in higher dimensions since fewer neighbors can be used, and is just simpler to set up in practice. Additionally, we generate each fine point independently, which removes artificial boundaries and enables the inverse and determinant. Finally, among other optimizations, the custom CUDA implementation generates refinement matrices on the fly, eliminating the main memory bottleneck of ICR and delivering substantial performance improvements for large problems. All of this being said, ICR was a huge leap forward and is still an excellent option. Users who are familiar with NIFTy should feel no need to switch if it is working for them.

*Can I use this with NIFTy, blackjax, numpyro, etc?* \
Yes! It should be easy to plug GraphGP into any JAX-based inference framework. We recommend MGVI and geoVI as implemented in NIFTy as probably the only algorithms capable of handling inference at scale.

*Is the tree order optimal?* \
No. The optimal order will depend on the distribution of points, and is probably intractable to determine in general. The tree order is a "good guess" for arbitrary point distributions which also happens to be necessary for efficient neighbor queries. See the software paper for more discussion. If you have a better idea, feel free to construct your own graph! We make it transparent and offer a helper function to ensure your graph is valid.

*Can I differentiate with respect to the locations of points?* \
Yes and no. It is our eventual goal to enable adaptive resolution by optimizing the locations of points. However, the simplest attempt results in local minima where points collapse on each other. Additionally, the nearest neighbor search requires tree order which changes as the points move. But the graph order must remain fixed for parameters to retain their meaning. There are possible solutions to both of these problems but there is more work to be done. So for now, only the JAX version supports differentiation with respect to the location of points, and users should not expect this kind of optimization to work easily.

*Can I use a non-stationary kernel?* \
Mild anisotropy or inhomogeneity should be fine as far as the algorithm is concerned. The key question is whether or not the nearest neighbors in space are ones with roughly the largest influence. (In principle, one could try to find neighbors in kernel space, see the theory paper for a discussion.) We have not yet implemented any non-stationary kernels, nor is there a convenient discretization that will work for general non-stationary kernels. If you have a specific kernel you wish to use, please submit an issue and we will consider it. Adding kernels to the JAX implementation should be straightforward.

*Can I distribute the GP across multiple GPUs?* \
Please try! With some understanding of the structure, it's not too difficult to split a graph across multiple machines. The main challenge is how to build the graph in the first place if it cannot fit in on a single machine. Our automatic tools require global sorts which are not easily distributed. Instead, users must rely on knowledge of their point distribution to pre-determine the division of points among machines, after which the automatic tools can be used. There is an example notebook demonstrating this for for a healpix spherical grid.

*Why can't I just use a non-uniform fast Fourier transform?* \
Non-uniform FFTs work by first interpolating onto a regular grid. This is handled very carefully but does not really solve our problem since the dynamic range will be limited by the dynamic range of the regular grid. One could think of GraphGP as a specialized NUFFT which is truly non-uniform but only accurate for decaying power spectra.

*Is there TPU support?* \
Only via the pure JAX version. It would be awesome to have a custom TPU implementation as well. Perhaps if Google offers us tons of free TPUs to make the world's largest dust map we'll put in the effort ;)

*What do you have planned for the future?* \
Faster graph construction. Moving points. Multi-GPU. Line integrals with no memory overhead and autodiff support. Non-stationary kernels. We do not plan to add inference tools to this package.

*Who made this?* \
Insert some note about contributions. Mention NIFTy, KIPAC, all authors of papers.

If you use this software in your work, please cite both the software and theory papers.