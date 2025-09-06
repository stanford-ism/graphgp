import numpy as np
import jax
import hugegp as gp

def graph_shard(graph: gp.Graph, n_shards: int, shape: tuple[int, ...] | None = None, axis: int = 0, cuda: bool = False) -> list[gp.Graph]:
    """Sharding for GraphGP.
    Builds and returns a list of graphs
    """
    if cuda:
        raise NotImplementedError("CUDA not implemented")
        #FIXME inconsistencies in hugegp.tree.query_neighbors vs hugegp-cuda.tree.query_neighbors

    n0 = graph.offsets[0]

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
            nbrs.sort()
            nbrs = nbrs[nbrs >= n0]
            # TODO check against searchsorted method
            nbrs = np.setdiff1d(nbrs, all_ids, assume_unique=True)
            missing = np.concatenate((missing, nbrs))
            missing.sort()
            all_ids = np.concatenate((all_ids, nbrs))
            all_ids.sort()
            active_ids = nbrs
            search = active_ids.size > 0
        assert np.all(np.setdiff1d(all_ids, missing, assume_unique=True) == graph_fids) #TODO make debug optional

    return graphs