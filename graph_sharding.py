import numpy as np
import jax
import hugegp as gp


def graph_shard(
    graph: gp.Graph,
    n_shards: int,
    shape: tuple[int, ...] | None = None,
    axis: int = 0,
    cuda: bool = False,
    _debug: bool = False,
) -> tuple[list[gp.Graph], list[tuple[np.ndarray, ...]]]:
    """Sharding for GraphGP.

    Builds and returns a list of graphs and lists of extra gather indices that
    build a valid graph for gp generation of each shard. The extra indices are
    neighbors (and neighbors of neighbors, etc.) that are not included in the
    tensor shard but need to be included in the graph for gp generation.

    TODO: Check performance for large graphs. Requires operations on full graph.
    """
    if cuda:
        raise NotImplementedError("CUDA not implemented")
        # FIXME @dodgebc: inconsistent behavior in hugegp.tree.query_neighbors vs hugegp-cuda.tree.query_neighbors

    n0 = graph.offsets[0]

    if shape is None:
        shape = graph.points.shape[:1]
    if axis >= len(shape):
        raise ValueError(f"Axis {axis} is out of bounds for shape {shape}")
    if shape[axis] % n_shards != 0:
        raise ValueError(
            f"Shape {shape} is not divisible by {n_shards} for axis {axis}"
        )
    shard_size = shape[axis] // n_shards

    graph_ids = graph.indices
    assert graph_ids is not None
    graph_inv_ids = np.empty_like(graph_ids)
    graph_inv_ids[graph_ids] = np.arange(graph_ids.size)

    ids = list(np.arange(sh, dtype=np.int_) for sh in shape)
    graphs = []
    gathers = []
    for i in range(n_shards):
        # Create shard in original tensor
        ids[axis] = np.arange(i * shard_size, (i + 1) * shard_size, dtype=np.int_)
        fids = np.meshgrid(*ids, indexing="ij")
        # Ravel into 1D flattened tensor
        fids = np.ravel_multi_index(fids, shape).flatten()
        # Get ids in tree order
        graph_fids = graph_inv_ids[fids]
        graph_fids.sort()
        if _debug:
            assert np.all(np.unique(graph_fids) == graph_fids)
        # Always include all initial points
        missing = np.setdiff1d(np.arange(n0), graph_fids, assume_unique=True)
        all_ids = graph_fids[graph_fids >= n0]
        nbrs = all_ids
        search = True
        while search:
            # Add all missing neighbors to `missing` until graph shard is valid
            nbrs = graph.neighbors[nbrs - n0].flatten()
            nbrs = nbrs[nbrs >= n0]
            nbrs = np.unique(nbrs)
            nbrs = np.setdiff1d(nbrs, all_ids, assume_unique=True)
            # TODO @dodgebc: can we do the preceding operations on the tree
            # without uniqifying and getting all neighbors?
            missing = np.concatenate((missing, nbrs))
            all_ids = np.concatenate((all_ids, nbrs))
            all_ids.sort()
            search = nbrs.size > 0
        missing.sort()
        if _debug:
            assert np.all(np.unique(missing) == missing)

        all_ids = np.concatenate((graph_inv_ids[fids], missing))
        sorting = np.argsort(all_ids)  # New tree order
        all_ids = all_ids[sorting]

        new_neighbors = graph.neighbors[all_ids[n0:] - n0]
        if _debug:
            assert (
                np.setdiff1d(np.unique(new_neighbors), all_ids, assume_unique=True).size
                == 0
            )
            assert np.all(all_ids[:n0] == np.arange(n0))
            assert np.all(np.unique(all_ids) == all_ids)
        new_points = graph.points[all_ids]
        new_neighbors = graph.neighbors[all_ids[n0:] - n0]

        # Index into new tree
        all_inv_ids = np.full(all_ids.max() + 1, -1)
        all_inv_ids[all_ids] = np.arange(all_ids.size)
        new_neighbors = all_inv_ids[new_neighbors]
        assert np.all(new_neighbors != -1)
        # New depths
        depths = gp.compute_depths(new_neighbors, n0=n0, cuda=cuda)
        offsets = np.searchsorted(depths, np.arange(1, np.max(depths) + 2))
        offsets = tuple(int(o) for o in offsets)
        # Get missing ids to index into input tensor″
        missing = graph_ids[missing]
        missing = np.unravel_index(missing, shape)
        gathers.append(missing)
        new_graph = gp.Graph(new_points, new_neighbors, offsets, indices=sorting)
        if _debug:
            gp.check_graph(new_graph)
        graphs.append(new_graph)
    return graphs, gathers


def generate_sharded(
    graphs: list[gp.Graph],
    gathers: list[tuple[np.ndarray, ...]],
    cov,
    xis,
    shard_axis=0,
    cuda=False,
    _debug=True,
):
    if not _debug:
        raise NotImplementedError("Only demo mode with debug enabled implemented!")
    assert len(graphs) == len(gathers)
    nshard = len(graphs)
    results = []
    shard_size = xis.shape[shard_axis] // nshard
    for i, (graph, gather) in enumerate(zip(graphs, gathers)):
        # TODO replace with jax sharding and all_to_all for gather
        print("Getting shard", i)
        xis_shard = xis[i * shard_size : (i + 1) * shard_size]
        print("with shard size", xis_shard.size)
        xis_in = jnp.concatenate((xis_shard.flatten(), xis[gather]))
        print("with gather size", xis_in.size)
        values_shard = gp.generate(graph, cov, xis_in, cuda=cuda)
        values_shard = values_shard[: xis_shard.size].reshape(xis_shard.shape)
        results.append(values_shard)
    results = jnp.concatenate(results, axis=shard_axis)
    return results


if __name__ == "__main__":
    import jax.numpy as jnp
    # 2D grid test with sharding along one axis
    npix1 = 128
    npix2 = 64

    key = jax.random.key(137)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (npix1,))
    x = jnp.sort(x)
    y = jax.random.uniform(k2, (npix2,))
    points = jnp.meshgrid(x, y, indexing="ij")
    points = jnp.stack(points, axis=-1)

    graph = gp.build_graph(points.reshape(-1, 2), n0=128, k=3, cuda=False)
    xis = jax.random.normal(k3, (npix1, npix2))
    cov = gp.MaternCovariance(p=1)
    cov_bins = gp.make_cov_bins(r_min=1e-5, r_max=1, n_bins=512)
    cov_vals = cov(cov_bins, cutoff=0.1)
    cov = (cov_bins, cov_vals)
    values = gp.generate(graph, cov, xis.flatten(), cuda=False)
    values = values.reshape(xis.shape)

    shard_axis = 0
    nshard = 4
    graphs, gathers = graph_shard(
        graph, nshard, shape=xis.shape, axis=shard_axis, cuda=False,
        _debug=True,
    )
    results = generate_sharded(
        graphs, gathers, cov, xis, shard_axis=shard_axis, cuda=False
    )
    print(results[:10,0], values[:10,0])
    assert jnp.allclose(results, values)
