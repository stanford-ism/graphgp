# Distributed GraphGP design

This document describes the exact distributed GraphGP backend on the
`revive_shard_graph` branch.  It is the correctness reference for future
communication-saving approximations.  The implementation is pure JAX at
runtime and is designed to return a globally shaped, spatially sharded
`jax.Array`.

## Scope and status

The branch distributes an existing GraphGP dependency graph over a one-axis
JAX device mesh.  Every graph node has exactly one owner, and every conditional
is evaluated by that owner.  Remote parent values are exchanged as the graph
is traversed.  The result is mathematically the same GraphGP realization as
the ordinary single-device generator, up to floating-point roundoff.

This is an exact baseline.  It does not yet implement halos, truncated
conditioning, coarse-global/fine-local correlations, shared separators, or a
sharded line-of-sight operator.

## Mathematical recap

Let $x_i\in\mathbb{R}^d$ be the locations at which a field is represented and
let $k(r)$ be the stationary covariance kernel.  A dense Gaussian process
would have

$$
  f \sim \mathcal{N}(0, K), \qquad K_{ij}=k(\lVert x_i-x_j\rVert).
$$

GraphGP uses the Vecchia/nearest-neighbor factorization.  The points are put
in an ordering such that every non-initial point has a set of earlier parent
points $P_i$.  Instead of conditioning on all earlier points, GraphGP keeps
only $k$ preceding neighbors.  This makes the factorization sparse and gives
linear storage and execution cost for fixed $k$.

### Dense seed

The first $n_0$ points are treated densely.  If $K_0$ is their covariance
matrix and $K_0=L_0L_0^\mathsf{T}$, then for white-noise excitations
$\xi_0\sim\mathcal{N}(0,I)$,

$$
  f_0 = L_0\xi_0.
$$

The seed is the only part deliberately replicated by the distributed
implementation; it should remain small compared with the full field.

### Conditional refinement

For a later point $i$, form the covariance matrix of its parents and the
point itself:

$$
  C_i = K(P_i\cup\{x_i\}, P_i\cup\{x_i\}) = L_iL_i^\mathsf{T}.
$$

Writing $k=|P_i|$, the implementation obtains the conditional weights and
standard deviation from the Cholesky factor:

$$
  w_i = L_i[P_i,P_i]^{-\mathsf{T}} L_i[x_i,P_i],
  \qquad
  \sigma_i = L_i[x_i,x_i].
$$

The generated value is then

$$
  f_i = w_i^\mathsf{T}f_{P_i} + \sigma_i\xi_i.
$$

Thus the distributed path must provide exactly the same parent values and use
exactly the same covariance/conditional calculation as the ordinary
`graphgp.generate` path.  Covariance inputs are discretized radii and values;
the kernel is linearly interpolated between radii.

## Graph and tree structure

`build_graph` first constructs a GPU-friendly k-d-tree ordering.  It retains
the permutation back to the caller's original point order.  For every point
after the seed it queries $k$ nearest points that precede it in this order.

The result is a DAG represented by:

- `points`: points in topological/tree order;
- `neighbors`: the $k$ preceding parents for every non-seed point;
- `offsets`: boundaries of dependency levels/batches;
- `indices`: the permutation between original and internal point order.

The `offsets` levels have an important invariant: a point in a level never
depends on another point in that same level.  Therefore all points in one
level can be generated in parallel once earlier levels are available.

The structure is tree-derived but the execution graph is a bounded-parent DAG:
a node can be used by many later nodes, and each node has up to $k$ parents.
The dependency level, rather than the k-d-tree depth alone, determines the
safe execution order.

## Partition compilation

The compiler is called as follows:

```python
plan = graphgp.distributed.partition_graph(
    graph,
    owners,                 # one owner per node, in original point order
    output_shape=(...),
)
```

`owners` may describe arbitrary geometric domains.  It is converted into
topological order using the graph's original-index permutation.  The compiler
then creates a `DistributedGraph` PyTree containing:

- owned node IDs in topological and original order;
- per-level local node slots and active masks;
- local parent slots;
- remote parent source-partition and receive-slot tables;
- per-level export slots for boundary values;
- seed excitation routing tables;
- output send/receive routing tables;
- static shape metadata and padding widths.

JAX requires rectangular array shapes, while partitions can own different
numbers of nodes.  The compiler therefore pads each partition to a common
width.  Padding entries are filled with safe dummy rows and are always guarded
by active masks.  Padding never creates an additional semantic GraphGP node.

At least one dimension of `output_shape` must be divisible by the number of
partitions.  The first such dimension is used as the spatial sharding axis.
This is an output-layout constraint, not a restriction on how `owners` must be
constructed.

## Exact distributed execution

```python
values = graphgp.distributed.generate(
    plan,
    covariance,
    xi,
    mesh=mesh,
    axis_name="space",
)
```

The runtime sequence is:

1. **Route excitations.**  An `all_to_all` sends each white-noise value to the
   partition that owns its graph node.
2. **Reconstruct the seed.**  The seed values are exchanged with an
   `all_gather`; every device evaluates the small dense Cholesky seed.
3. **Traverse dependency levels.**  A padded `lax.scan` visits graph levels in
   order.
4. **Exchange boundaries.**  At each level, each device packs the parent values
   needed remotely.  An `all_gather` makes the padded boundary buffer visible
   to every device.
5. **Evaluate local conditionals.**  Local parents are read from local slots;
   remote parents are indexed from the gathered buffer.  Each active owned
   node is evaluated once with the shared conditional-weight helper.
6. **Route outputs.**  An `all_to_all` maps owned values into the requested
   spatial output layout.

The returned object has the global shape but retains a `NamedSharding` whose
`space` axis is partitioned.  The full field is not gathered onto one device
or materialized on the host.

The first implementation intentionally uses padded `all_gather` for boundary
exchange.  A parent can be transmitted more than once if later levels need it
again.  Sparse `all_to_all` and export-once caching are planned optimizations,
not changes to the mathematical model.

## Correctness oracle

`generate_recompute(plan, covariance, xi)` is a deliberately non-scalable
reference implementation.  For each partition it computes the transitive
ancestor closure of that partition's owned nodes, builds a temporary local
GraphGP graph, and runs the ordinary generator.  The owned results are then
placed into the global output.

The intended validation chain is:

```python
single = graphgp.generate(graph, covariance, xi)
oracle = graphgp.distributed.generate_recompute(plan, covariance, xi)
exact = graphgp.distributed.generate(plan, covariance, xi, mesh=mesh)
```

`single`, `oracle`, and `exact` should agree within the tolerance appropriate
to the chosen floating-point type.  The oracle may replicate ancestors and
perform much more work; its purpose is to make partitioning errors obvious.

## Autodiff and JAX transformations

The execution kernel is built from JAX primitives (`lax.scan`,
`lax.all_gather`, `lax.all_to_all`, array indexing, and linear algebra).  The
communication operations therefore appear in the JAX program and participate
in forward- and reverse-mode differentiation.

The test suite exercises:

- `jit` compilation;
- JVP and VJP with respect to excitations;
- gradients with respect to covariance values;
- leading batch/sample axes;
- `vmap` combined with JVP/VJP;
- the adjoint identity between a tangent and cotangent.

The partition plan is static with respect to topology and shapes, while its
array leaves are JAX values.  This keeps the plan compatible with PyTree-based
transformations without rebuilding host-side graph metadata inside a traced
function.

## Communication and memory accounting

`partition_stats(plan)` reports:

- `owned_nodes`: active nodes per partition;
- `cross_partition_parents`: number of graph edges crossing ownership cuts;
- `communicated_values_per_evaluation`: padded communication estimate for the
  current implementation;
- padding fractions for owned, level, and boundary arrays;
- `ancestor_replication_factor` for the recompute oracle.

The important scaling target is approximately (N/P) graph state and field
values per device, excluding the replicated dense seed and padding.  Exact
exchange communication is driven by graph cuts and padded boundary widths,
not by the full field size.

The benchmark scaffold in `benchmarks/distributed_scaling.py` records compile
time, repeated runtime, JAX memory-analysis estimates, device count, and
partition statistics as JSON.  It is intended to be extended with measured
communication volume and peak HBM usage.

## Current limitations

The branch currently has these deliberate limitations:

- one-dimensional device mesh (`space` by convention);
- host-side partition compilation;
- padded `all_gather` boundary exchange;
- replicated covariance arrays and dense seed;
- no multi-host CI test yet;
- no spatially sharded LOS/physics operator;
- CUDA extension remains single-device only;
- no approximate correlation decomposition yet;
- no NIFTy.re `GraphGPField` adapter in this repository.

The last item belongs in the coordinated NIFTy `graphgpfield` branch.  That
adapter should treat this module as an ordinary composable field model while
preserving the `space` sharding of the excitation and generated field.

## Next research stages

Once exact exchange is stable, the intended progression is:

1. measure exact exchange on 2, 4, and 8 GPUs;
2. replace padded boundary `all_gather` with sparse `all_to_all` where useful;
3. add spatially local physics and LOS reduction operators;
4. retain explicit refinement-scale metadata in `Graph`;
5. implement coarse-global/fine-local correlations;
6. add fixed-width halos and bounded remote-parent conditioning;
7. investigate shared separator/inducing variables;
8. add custom VJP/CUDA kernels only after pure-JAX behavior is validated.

Approximation benchmarks should report runtime, peak memory, communication,
paired-excitation error, boundary-correlation error, power-spectrum error, and
small-problem covariance KL divergence relative to this exact backend.
