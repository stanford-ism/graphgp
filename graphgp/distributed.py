"""Exact distributed execution for GraphGP dependency graphs.

Graph construction and partition planning happen on the host.  Runtime
generation is a pure-JAX SPMD computation: every graph node is evaluated by
its owner and remote parent values are exchanged between dependency levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.tree_util import register_dataclass

try:  # JAX >= 0.6
    from jax import shard_map as _shard_map
except ImportError:  # JAX 0.5
    from jax.experimental.shard_map import shard_map as _shard_map

_SHARD_MAP_SUPPORTS_CHECK_REP = "check_rep" in inspect.signature(_shard_map).parameters


def _distributed_shard_map(fun, **kwargs):
    """Call shard_map across the JAX 0.5 and current APIs."""
    if _SHARD_MAP_SUPPORTS_CHECK_REP:
        kwargs["check_rep"] = False
    try:
        return _shard_map(fun, **kwargs)
    except TypeError as error:
        # Some newer JAX releases retain a permissive signature while
        # rejecting this removed keyword when the map is constructed.
        if "check_rep" not in kwargs or "check_rep" not in str(error):
            raise
        kwargs.pop("check_rep")
        return _shard_map(fun, **kwargs)

from .graph import Graph, check_graph
from .refine import (
    _conditional_weights_std,
    compute_cov_matrix,
    generate as generate_single,
)


@register_dataclass
@dataclass
class DistributedGraph:
    """Padded execution plan for an exact partition of a :class:`Graph`.

    Array fields whose first dimension is ``n_partitions`` are sharded over
    the runtime mesh.  Node identifiers are in GraphGP topological order;
    excitation/output identifiers are in the original input order.
    """

    graph: Graph
    owners: Array
    owned_topological: Array
    owned_original: Array
    owned_active: Array
    seed_local_slots: Array
    seed_active: Array
    seed_source: Array
    seed_source_slot: Array
    level_local_slots: Array
    level_active: Array
    level_points: Array
    level_parent_points: Array
    parent_local_slots: Array
    parent_remote_source: Array
    parent_remote_slot: Array
    parent_is_remote: Array
    export_local_slots: Array
    export_active: Array
    output_send_local_slots: Array
    output_send_active: Array
    output_receive_offsets: Array
    output_receive_active: Array
    output_shape: tuple[int, ...] = field(metadata={"static": True})
    output_axis: int = field(metadata={"static": True})
    n_partitions: int = field(metadata={"static": True})
    n_points: int = field(metadata={"static": True})
    n0: int = field(metadata={"static": True})
    k: int = field(metadata={"static": True})
    n_levels: int = field(metadata={"static": True})
    max_owned: int = field(metadata={"static": True})
    max_level: int = field(metadata={"static": True})
    max_seed: int = field(metadata={"static": True})
    max_boundary: int = field(metadata={"static": True})
    max_output_send: int = field(metadata={"static": True})


@dataclass(frozen=True)
class PartitionStats:
    owned_nodes: tuple[int, ...]
    cross_partition_parents: int
    communicated_values_per_evaluation: int
    owned_padding_fraction: float
    level_padding_fraction: float
    boundary_padding_fraction: float
    ancestor_replication_factor: float


def _as_numpy(x):
    return np.asarray(jax.device_get(x))


def _fill_unique_padding(active_slots, width, n_available):
    active_slots = list(active_slots)
    unused = [i for i in range(n_available) if i not in set(active_slots)]
    needed = width - len(active_slots)
    if needed > len(unused):
        raise ValueError("insufficient local slots for padding")
    return np.asarray(active_slots + unused[:needed], dtype=np.int32)


def partition_graph(graph: Graph, owners, *, output_shape=None) -> DistributedGraph:
    """Compile an exact owner-computes execution plan.

    ``owners`` contains one integer owner per node in the original input order.
    Owner identifiers must form the dense range ``0 .. n_partitions - 1``.
    """

    check_graph(graph)
    points = _as_numpy(graph.points)
    neighbors = _as_numpy(graph.neighbors).astype(np.int32, copy=False)
    indices = np.arange(len(points), dtype=np.int32) if graph.indices is None else _as_numpy(graph.indices)
    indices = indices.astype(np.int32, copy=False)
    owners = np.asarray(owners, dtype=np.int32)
    n = len(points)
    if owners.shape != (n,):
        raise ValueError(f"owners must have shape ({n},), got {owners.shape}")
    unique_owners = np.unique(owners)
    if unique_owners.size == 0 or not np.array_equal(unique_owners, np.arange(unique_owners.size)):
        raise ValueError("owner identifiers must form the dense range 0 .. n_partitions - 1")
    n_partitions = int(unique_owners.size)
    if output_shape is None:
        output_shape = (n,)
    output_shape = tuple(int(i) for i in output_shape)
    if int(np.prod(output_shape)) != n:
        raise ValueError("the product of output_shape must match the number of graph nodes")

    n0 = int(graph.offsets[0])
    k = int(neighbors.shape[1])
    owner_topological = owners[indices]
    owned = [np.flatnonzero(owner_topological == p).astype(np.int32) for p in range(n_partitions)]
    max_owned = max(max(map(len, owned)), 1)

    owned_topological = np.full((n_partitions, max_owned), -1, dtype=np.int32)
    owned_original = np.zeros((n_partitions, max_owned), dtype=np.int32)
    owned_active = np.zeros((n_partitions, max_owned), dtype=bool)
    local_maps = []
    for p, ids in enumerate(owned):
        owned_topological[p, : len(ids)] = ids
        owned_original[p, : len(ids)] = indices[ids]
        owned_active[p, : len(ids)] = True
        local_maps.append({int(node): slot for slot, node in enumerate(ids)})

    seed_ids = [ids[ids < n0] for ids in owned]
    max_seed = max(max(map(len, seed_ids)), 1)
    seed_local_slots = np.zeros((n_partitions, max_seed), dtype=np.int32)
    seed_active = np.zeros((n_partitions, max_seed), dtype=bool)
    seed_source = np.empty(n0, dtype=np.int32)
    seed_source_slot = np.empty(n0, dtype=np.int32)
    for p, ids in enumerate(seed_ids):
        slots = [local_maps[p][int(i)] for i in ids]
        padded = _fill_unique_padding(slots, max_seed, max_owned)
        seed_local_slots[p] = padded
        seed_active[p, : len(ids)] = True
        for send_slot, node in enumerate(ids):
            seed_source[node] = p
            seed_source_slot[node] = send_slot

    ranges = [np.arange(graph.offsets[i], graph.offsets[i + 1], dtype=np.int32) for i in range(len(graph.offsets) - 1)]
    n_levels = len(ranges)
    level_nodes = [[r[owner_topological[r] == p] for r in ranges] for p in range(n_partitions)]
    max_level = max((len(ids) for per_owner in level_nodes for ids in per_owner), default=0)
    max_level = max(max_level, 1)

    exports = [[[] for _ in range(n_levels)] for _ in range(n_partitions)]
    for level, r in enumerate(ranges):
        for node in r:
            dst = int(owner_topological[node])
            for parent in neighbors[node - n0]:
                src = int(owner_topological[parent])
                if src != dst:
                    exports[src][level].append(int(parent))
    for p in range(n_partitions):
        for level in range(n_levels):
            exports[p][level] = sorted(set(exports[p][level]))
    max_boundary = max((len(v) for per_owner in exports for v in per_owner), default=0)
    max_boundary = max(max_boundary, 1)

    shp = (n_partitions, n_levels, max_level)
    level_local_slots = np.zeros(shp, dtype=np.int32)
    level_active = np.zeros(shp, dtype=bool)
    level_points = np.empty(shp + (points.shape[1],), dtype=points.dtype)
    level_parent_points = np.empty(shp + (k, points.shape[1]), dtype=points.dtype)
    parent_local_slots = np.zeros(shp + (k,), dtype=np.int32)
    parent_remote_source = np.zeros(shp + (k,), dtype=np.int32)
    parent_remote_slot = np.zeros(shp + (k,), dtype=np.int32)
    parent_is_remote = np.zeros(shp + (k,), dtype=bool)
    export_local_slots = np.zeros((n_partitions, n_levels, max_boundary), dtype=np.int32)
    export_active = np.zeros((n_partitions, n_levels, max_boundary), dtype=bool)

    if n > n0:
        exemplar = n0
        dummy_point = points[exemplar]
        dummy_parent_points = points[neighbors[0]]
    else:
        dummy_point = points[0]
        dummy_parent_points = np.broadcast_to(points[0], (k, points.shape[1])).copy()

    export_lookup = [[{} for _ in range(n_levels)] for _ in range(n_partitions)]
    for p in range(n_partitions):
        for level in range(n_levels):
            exp = exports[p][level]
            slots = [local_maps[p][node] for node in exp]
            padded = _fill_unique_padding(slots, max_boundary, max_owned)
            export_local_slots[p, level] = padded
            export_active[p, level, : len(exp)] = True
            export_lookup[p][level] = {node: slot for slot, node in enumerate(exp)}

    output_axis = next(
        (axis for axis, size in enumerate(output_shape) if size % n_partitions == 0),
        None,
    )
    if output_axis is None:
        raise ValueError(
            "at least one output_shape axis must be divisible by the number "
            "of partitions"
        )
    local_output_shape = list(output_shape)
    local_output_shape[output_axis] //= n_partitions
    local_output_shape = tuple(local_output_shape)
    output_routes = [[[] for _ in range(n_partitions)] for _ in range(n_partitions)]
    for src, ids in enumerate(owned):
        for local_slot, node in enumerate(ids):
            original = int(indices[node])
            coordinate = list(np.unravel_index(original, output_shape))
            dst = coordinate[output_axis] // local_output_shape[output_axis]
            coordinate[output_axis] %= local_output_shape[output_axis]
            local_offset = np.ravel_multi_index(coordinate, local_output_shape)
            output_routes[src][dst].append((local_slot, int(local_offset)))
    max_output_send = max((len(route) for routes in output_routes for route in routes), default=0)
    max_output_send = max(max_output_send, 1)
    output_send_local_slots = np.zeros((n_partitions, n_partitions, max_output_send), dtype=np.int32)
    output_send_active = np.zeros((n_partitions, n_partitions, max_output_send), dtype=bool)
    output_receive_offsets = np.zeros((n_partitions, n_partitions, max_output_send), dtype=np.int32)
    output_receive_active = np.zeros((n_partitions, n_partitions, max_output_send), dtype=bool)
    for src in range(n_partitions):
        for dst in range(n_partitions):
            route = output_routes[src][dst]
            slots = [slot for slot, _ in route]
            padded = _fill_unique_padding(slots, max_output_send, max_owned)
            output_send_local_slots[src, dst] = padded
            output_send_active[src, dst, : len(route)] = True
            output_receive_offsets[dst, src, : len(route)] = [offset for _, offset in route]
            output_receive_active[dst, src, : len(route)] = True

    for p in range(n_partitions):
        for level in range(n_levels):
            nodes = level_nodes[p][level]
            slots = [local_maps[p][int(node)] for node in nodes]
            level_local_slots[p, level] = _fill_unique_padding(slots, max_level, max_owned)
            level_active[p, level, : len(nodes)] = True
            level_points[p, level] = dummy_point
            level_parent_points[p, level] = dummy_parent_points
            for row, node in enumerate(nodes):
                parents = neighbors[node - n0]
                level_points[p, level, row] = points[node]
                level_parent_points[p, level, row] = points[parents]
                for col, parent in enumerate(parents):
                    src = int(owner_topological[parent])
                    if src == p:
                        parent_local_slots[p, level, row, col] = local_maps[p][int(parent)]
                    else:
                        parent_is_remote[p, level, row, col] = True
                        parent_remote_source[p, level, row, col] = src
                        parent_remote_slot[p, level, row, col] = export_lookup[src][level][int(parent)]

    return DistributedGraph(
        graph=graph,
        owners=jnp.asarray(owners),
        owned_topological=jnp.asarray(owned_topological),
        owned_original=jnp.asarray(owned_original),
        owned_active=jnp.asarray(owned_active),
        seed_local_slots=jnp.asarray(seed_local_slots),
        seed_active=jnp.asarray(seed_active),
        seed_source=jnp.asarray(seed_source),
        seed_source_slot=jnp.asarray(seed_source_slot),
        level_local_slots=jnp.asarray(level_local_slots),
        level_active=jnp.asarray(level_active),
        level_points=jnp.asarray(level_points),
        level_parent_points=jnp.asarray(level_parent_points),
        parent_local_slots=jnp.asarray(parent_local_slots),
        parent_remote_source=jnp.asarray(parent_remote_source),
        parent_remote_slot=jnp.asarray(parent_remote_slot),
        parent_is_remote=jnp.asarray(parent_is_remote),
        export_local_slots=jnp.asarray(export_local_slots),
        export_active=jnp.asarray(export_active),
        output_send_local_slots=jnp.asarray(output_send_local_slots),
        output_send_active=jnp.asarray(output_send_active),
        output_receive_offsets=jnp.asarray(output_receive_offsets),
        output_receive_active=jnp.asarray(output_receive_active),
        output_shape=output_shape,
        output_axis=output_axis,
        n_partitions=n_partitions,
        n_points=n,
        n0=n0,
        k=k,
        n_levels=n_levels,
        max_owned=max_owned,
        max_level=max_level,
        max_seed=max_seed,
        max_boundary=max_boundary,
        max_output_send=max_output_send,
    )


def _partitioned_spec(ndim, partition_axis, axis_name):
    spec = [None] * ndim
    spec[partition_axis] = axis_name
    return P(*spec)


def _local_spec(ndim, axis_name):
    return P(axis_name, *([None] * (ndim - 1)))


def _batched_dense(points, covariance, xi):
    K = compute_cov_matrix(covariance, points, points)
    L = jnp.linalg.cholesky(K)
    return jnp.einsum("ij,...j->...i", L, xi)


def generate(plan: DistributedGraph, covariance, xi: Array, *, mesh: Mesh, axis_name="space") -> Array:
    """Generate the exact GraphGP field with one owner per graph node."""

    if mesh.shape.get(axis_name) != plan.n_partitions:
        raise ValueError(
            f"mesh axis {axis_name!r} has size {mesh.shape.get(axis_name)}, expected {plan.n_partitions}"
        )
    xi = jnp.asarray(xi)
    if xi.shape[-len(plan.output_shape) :] != plan.output_shape:
        raise ValueError(f"xi must end in output shape {plan.output_shape}, got {xi.shape}")
    batch_shape = xi.shape[: -len(plan.output_shape)] if plan.output_shape else xi.shape
    batch_ndim = len(batch_shape)
    xi_spec = _partitioned_spec(batch_ndim + 2, batch_ndim, axis_name)
    field_axes = [None] * (batch_ndim + len(plan.output_shape))
    field_axes[batch_ndim + plan.output_axis] = axis_name
    field_spec = P(*field_axes)

    def route_excitations(xi_block, receive_offsets, receive_mask, owned_slots, owned_mask):
        receive_offsets = jnp.squeeze(receive_offsets, axis=0)
        receive_mask = jnp.squeeze(receive_mask, axis=0)
        owned_slots = jnp.squeeze(owned_slots, axis=0)
        owned_mask = jnp.squeeze(owned_mask, axis=0)
        local_xi = xi_block.reshape(batch_shape + (-1,))
        sends = jnp.take(local_xi, receive_offsets, axis=-1) * receive_mask
        received = lax.all_to_all(
            sends,
            axis_name,
            split_axis=batch_ndim,
            concat_axis=batch_ndim,
            tiled=False,
        )
        local_owned = jnp.zeros(
            batch_shape + (plan.max_owned,), dtype=xi_block.dtype
        )
        local_owned = local_owned.at[..., owned_slots].add(received * owned_mask)
        return jnp.expand_dims(local_owned, axis=batch_ndim)

    route = _distributed_shard_map(
        route_excitations,
        mesh=mesh,
        in_specs=(
            field_spec,
            _local_spec(plan.output_receive_offsets.ndim, axis_name),
            _local_spec(plan.output_receive_active.ndim, axis_name),
            _local_spec(plan.output_send_local_slots.ndim, axis_name),
            _local_spec(plan.output_send_active.ndim, axis_name),
        ),
        out_specs=xi_spec,
        check_rep=False,
    )
    xi_owned = route(
        xi,
        plan.output_receive_offsets,
        plan.output_receive_active,
        plan.output_send_local_slots,
        plan.output_send_active,
    )

    local_arrays = (
        plan.owned_topological,
        plan.seed_local_slots,
        plan.seed_active,
        plan.level_local_slots,
        plan.level_active,
        plan.level_points,
        plan.level_parent_points,
        plan.parent_local_slots,
        plan.parent_remote_source,
        plan.parent_remote_slot,
        plan.parent_is_remote,
        plan.export_local_slots,
        plan.export_active,
        plan.output_send_local_slots,
        plan.output_send_active,
        plan.output_receive_offsets,
        plan.output_receive_active,
    )
    local_specs = tuple(_local_spec(a.ndim, axis_name) for a in local_arrays)
    output_spec = field_spec

    def local_generate(xi_block, arrays, seed_source, seed_source_slot, cov_bins, cov_vals, seed_points):
        xi_local = jnp.squeeze(xi_block, axis=batch_ndim)
        arrays = tuple(jnp.squeeze(a, axis=0) for a in arrays)
        (
            owned_topological,
            seed_slots,
            seed_mask,
            level_slots,
            level_mask,
            level_points,
            parent_points,
            parent_local,
            remote_source,
            remote_slot,
            remote_mask,
            export_slots,
            export_mask,
            output_send_slots,
            output_send_mask,
            output_receive_offsets,
            output_receive_mask,
        ) = arrays
        covariance_local = (cov_bins, cov_vals)
        values = jnp.zeros(batch_shape + (plan.max_owned,), dtype=xi.dtype)

        seed_send = jnp.take(xi_local, seed_slots, axis=-1) * seed_mask
        seed_gathered = lax.all_gather(seed_send, axis_name, axis=batch_ndim, tiled=False)
        seed_xi = seed_gathered[..., seed_source, seed_source_slot]
        seed_values = _batched_dense(seed_points, covariance_local, seed_xi)
        current_seed = jnp.take(values, seed_slots, axis=-1)
        seed_topo = owned_topological[seed_slots]
        new_seed = seed_values[..., jnp.clip(seed_topo, 0, plan.n0 - 1)]
        seed_update = jnp.where(seed_mask, new_seed, current_seed)
        values = values.at[..., seed_slots].set(seed_update)

        def step(values, data):
            (
                node_slots,
                active,
                fine_points,
                coarse_points,
                local_parent_slots,
                remote_sources,
                remote_slots,
                is_remote,
                send_slots,
                send_active,
            ) = data
            send_values = jnp.take(values, send_slots, axis=-1) * send_active
            received = lax.all_gather(send_values, axis_name, axis=batch_ndim, tiled=False)
            local_parent_values = jnp.take(values, local_parent_slots, axis=-1)
            remote_parent_values = received[..., remote_sources, remote_slots]
            coarse_values = jnp.where(is_remote, remote_parent_values, local_parent_values)
            weights, std = jax.vmap(_conditional_weights_std, in_axes=(None, 0, 0))(
                covariance_local, coarse_points, fine_points
            )
            mean = jnp.sum(coarse_values * weights, axis=-1)
            fine_xi = jnp.take(xi_local, node_slots, axis=-1)
            refined = mean + std * fine_xi
            current = jnp.take(values, node_slots, axis=-1)
            update = jnp.where(active, refined, current)
            return values.at[..., node_slots].set(update), None

        scan_data = (
            level_slots,
            level_mask,
            level_points,
            parent_points,
            parent_local,
            remote_source,
            remote_slot,
            remote_mask,
            export_slots,
            export_mask,
        )
        values, _ = lax.scan(step, values, scan_data)
        output_send = jnp.take(values, output_send_slots, axis=-1) * output_send_mask
        output_receive = lax.all_to_all(
            output_send,
            axis_name,
            split_axis=batch_ndim,
            concat_axis=batch_ndim,
            tiled=False,
        )
        local_output = jnp.zeros(batch_shape + (plan.n_points // plan.n_partitions,), dtype=values.dtype)
        local_output = local_output.at[..., output_receive_offsets].add(output_receive * output_receive_mask)
        local_shape = list(plan.output_shape)
        local_shape[plan.output_axis] //= plan.n_partitions
        local_shape = tuple(local_shape)
        return local_output.reshape(batch_shape + local_shape)

    cov_bins, cov_vals = covariance
    mapped = _distributed_shard_map(
        local_generate,
        mesh=mesh,
        in_specs=(
            xi_spec,
            local_specs,
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=output_spec,
        check_rep=False,
    )
    output = mapped(
        xi_owned,
        local_arrays,
        plan.seed_source,
        plan.seed_source_slot,
        cov_bins,
        cov_vals,
        plan.graph.points[: plan.n0],
    )
    return output


def _closure(graph: Graph, owned_topological):
    n0 = int(graph.offsets[0])
    neighbors = _as_numpy(graph.neighbors)
    closure = set(range(n0)) | set(map(int, owned_topological))
    frontier = [i for i in owned_topological if i >= n0]
    while frontier:
        parents = set(map(int, neighbors[np.asarray(frontier) - n0].ravel()))
        new = parents - closure
        closure.update(new)
        frontier = [i for i in new if i >= n0]
    return np.asarray(sorted(closure), dtype=np.int32)


def generate_recompute(plan: DistributedGraph, covariance, xi: Array) -> Array:
    """Exact ancestor-recompute oracle using full host-side partition metadata."""

    xi = jnp.asarray(xi)
    batch_shape = xi.shape[: -len(plan.output_shape)] if plan.output_shape else xi.shape
    xi_flat = xi.reshape(batch_shape + (plan.n_points,))
    graph = plan.graph
    global_neighbors = _as_numpy(graph.neighbors)
    global_indices = np.arange(plan.n_points) if graph.indices is None else _as_numpy(graph.indices)
    result = jnp.zeros(batch_shape + (plan.n_points,), dtype=xi.dtype)
    owned_topological = _as_numpy(plan.owned_topological)
    owned_active = _as_numpy(plan.owned_active)

    for p in range(plan.n_partitions):
        owned = owned_topological[p, owned_active[p]]
        ids = _closure(graph, owned)
        inverse = {int(node): i for i, node in enumerate(ids)}
        local_neighbors = np.asarray(
            [[inverse[int(parent)] for parent in global_neighbors[node - plan.n0]] for node in ids[plan.n0 :]],
            dtype=np.int32,
        )
        depths = np.zeros(len(ids), dtype=np.int32)
        for i in range(plan.n0, len(ids)):
            depths[i] = 1 + np.max(depths[local_neighbors[i - plan.n0]])
        offsets = [plan.n0]
        for depth in range(1, int(depths.max(initial=0)) + 1):
            offsets.append(int(np.searchsorted(depths, depth + 1)))
        if offsets[-1] != len(ids):
            offsets.append(len(ids))
        local_graph = Graph(
            points=graph.points[ids],
            neighbors=jnp.asarray(local_neighbors),
            offsets=tuple(offsets),
            indices=None,
        )
        local_xi = xi_flat[..., global_indices[ids]]
        flat_batch = local_xi.reshape((-1, len(ids)))
        local_values = jax.vmap(lambda z: generate_single(local_graph, covariance, z))(flat_batch)
        local_values = local_values.reshape(batch_shape + (len(ids),))
        positions = np.asarray([inverse[int(node)] for node in owned], dtype=np.int32)
        original = global_indices[owned]
        result = result.at[..., original].set(local_values[..., positions])
    return result.reshape(batch_shape + plan.output_shape)


def partition_stats(plan: DistributedGraph) -> PartitionStats:
    owned_active = _as_numpy(plan.owned_active)
    level_active = _as_numpy(plan.level_active)
    export_active = _as_numpy(plan.export_active)
    graph = plan.graph
    indices = np.arange(plan.n_points) if graph.indices is None else _as_numpy(graph.indices)
    owner_topological = _as_numpy(plan.owners)[indices]
    neighbors = _as_numpy(graph.neighbors)
    child_owner = owner_topological[plan.n0 :]
    cross = int(np.sum(owner_topological[neighbors] != child_owner[:, None]))
    ancestor_total = 0
    for p in range(plan.n_partitions):
        ancestor_total += len(_closure(graph, np.flatnonzero(owner_topological == p)))
    return PartitionStats(
        owned_nodes=tuple(int(i) for i in owned_active.sum(axis=1)),
        cross_partition_parents=cross,
        communicated_values_per_evaluation=int(
            plan.n_partitions
            * plan.n_partitions
            * (
                plan.max_seed
                + plan.n_levels * plan.max_boundary
                + 2 * plan.max_output_send
            )
        ),
        owned_padding_fraction=float(1.0 - owned_active.mean()),
        level_padding_fraction=float(1.0 - level_active.mean()),
        boundary_padding_fraction=float(1.0 - export_active.mean()),
        ancestor_replication_factor=float(ancestor_total / plan.n_points),
    )


__all__ = [
    "DistributedGraph",
    "PartitionStats",
    "partition_graph",
    "generate",
    "generate_recompute",
    "partition_stats",
]
