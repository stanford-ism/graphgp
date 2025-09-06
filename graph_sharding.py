import numpy as np
import jax
import hugegp as gp

def graph_shard(graph: gp.Graph, n_shards: int, shape: tuple[int, ...] | None = None, axis: int = 0, cuda: bool = False) -> tuple[list[gp.Graph], list[tuple[np.ndarray, tuple[np.ndarray, ...]]]]:
    """Sharding for GraphGP.
    Builds and returns a list of graphs
    """
    if cuda:
        raise NotImplementedError("CUDA not implemented")
        #FIXME inconsistencies in hugegp.tree.query_neighbors vs hugegp-cuda.tree.query_neighbors

    n0 = graph.offsets[0]

    depths = gp.compute_depths(graph.neighbors, n0=n0, cuda=cuda)

    if shape is None:
        shape = graph.points.shape[:1]
    if axis >= len(shape):
        raise ValueError(f"Axis {axis} is out of bounds for shape {shape}")
    if shape[axis] % n_shards != 0:
        raise ValueError(f"Shape {shape} is not divisible by {n_shards} for axis {axis}")
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
        ids[axis] = np.arange(i*shard_size, (i+1)*shard_size, dtype=np.int_)
        fids = np.meshgrid(*ids, indexing='ij')
        # Ravel into 1D flattened tensor
        fids = np.ravel_multi_index(fids, shape)
        # Get ids in tree order
        graph_fids = graph_inv_ids[fids]
        graph_fids.sort()
        assert np.all(np.unique(graph_fids) == graph_fids) #TODO make debug optional
        # Always include all initial points
        missing = np.setdiff1d(np.arange(n0), graph_fids, assume_unique=True)
        all_ids = np.concatenate((np.arange(n0), graph_fids[graph_fids >= n0]))
        active_ids = np.copy(all_ids)[n0:]
        search = True
        while search:
            # Add all missing neighbors to `missing` until graph shard is valid
            nbrs = graph.neighbors[active_ids - n0].flatten()
            nbrs = nbrs[nbrs >= n0]
            nbrs = np.unique(nbrs)
            # TODO check against searchsorted method
            nbrs = np.setdiff1d(nbrs, all_ids, assume_unique=True)
            missing = np.concatenate((missing, nbrs))
            missing.sort()
            all_ids = np.concatenate((all_ids, nbrs))
            all_ids.sort()
            active_ids = nbrs
            search = active_ids.size > 0
        assert np.all(np.setdiff1d(all_ids, missing, assume_unique=True) == graph_fids) #TODO make debug optional
        assert np.all(np.unique(missing) == missing)

        #TODO unify with above
        all_ids = np.concatenate((graph_inv_ids[fids], missing))
        sorting = np.argsort(all_ids)
        all_ids = all_ids[sorting]
        print(all_ids)

        new_neighbors = graph.neighbors[all_ids[n0:] - n0]
        #TODO make debug optional
        assert np.setdiff1d(np.unique(new_neighbors), all_ids, assume_unique=True).size == 0 
        assert np.all(all_ids[:n0] == np.arange(n0))
        assert np.all(np.unique(all_ids) == all_ids)
        new_points = graph.points[all_ids]
        new_neighbors = graph.neighbors[all_ids[n0:] - n0]

        all_inv_ids = np.full(all_ids.max() + 1, -1)
        all_inv_ids[all_ids] = np.arange(all_ids.size)
        new_neighbors = all_inv_ids[new_neighbors]
        assert np.all(new_neighbors != -1)

        depths = gp.compute_depths(new_neighbors, n0=n0, cuda=cuda)
        offsets = np.searchsorted(depths, np.arange(1, np.max(depths) + 2))
        offsets = tuple(int(o) for o in offsets)

        missing = graph_ids[missing]
        missing = np.unravel_index(missing, shape)
        gathers.append(missing)
        new_graph = gp.Graph(new_points, new_neighbors, offsets, indices=sorting)
        gp.check_graph(new_graph)
        graphs.append(new_graph)
    return graphs, gathers
