# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot geometry of the shared radixshmem KV pool.

The index is one radix tree over the full-attention group's chunk hashes.
Each lane has its own slot pool: FULL for the full-attention group, SWA for
the sliding-window groups, MAMBA for the align-mode mamba groups (their
states at a chunk boundary). Models with n times more SWA or mamba layers than
full-attention layers arrive as n such groups; each slot holds one chunk of
every group of its lane::

    slot s : [ worker 0 cell | ... | worker M-1 cell ][ pad ]
    cell   : [ member group 0 | member group 1 | ... ]
    member : [ tensor 0 pages | tensor 1 pages | ... ]

Derived from ``OffloadingConfig`` alone, so every process computes the same
values and attachers can check them against the owner's.
"""

import mmap
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.config import OffloadingConfig, OffloadingGroupConfig

DEFAULT_SLOT_ALIGN = mmap.PAGESIZE

# Fingerprint fields every process must agree on.
FINGERPRINT_LAYOUT_FIELDS = (
    "num_copies",
    "slot_align",
    "tokens_per_block",
    "blocks_per_chunk",
    "tokens_per_chunk",
    "kv_cache_layout",
    "model",
    "cache_formats",
    "pools",
)

# Pool kinds in store layout order.
POOL_KINDS = ("full", "swa", "mamba")

# Spec kinds that take the FULL lane.
FULL_KINDS = frozenset({"full_attention", "mla_attention", "sink_full_attention"})

HYBRID_HINT = "Try --disable-hybrid-kv-cache-manager."

CONNECTOR_HINT = (
    "RadixShmem needs the KV cache group details that RadixShmemConnector "
    "fills in; set kv_connector to RadixShmemConnector."
)


@dataclass(frozen=True, kw_only=True)
class RadixGroupConfig(OffloadingGroupConfig):
    """A group plus what the lanes need from its KV cache spec. Filled by
    ``RadixShmemConnector``, which sees the ``KVCacheConfig``."""

    kv_cache_spec_kind: str
    cache_format: tuple[str, ...]
    # Per-worker KV bytes of one block of this group's layers; 0 if unknown.
    kv_bytes_per_block: int = 0
    # Sliding-window groups: the window and the extra tokens kept (MTP).
    sliding_window: int | None = None
    extra_retained_tokens: int = 0


def radix_groups(config: OffloadingConfig) -> tuple[RadixGroupConfig, ...]:
    groups = tuple(g for g in config.groups if isinstance(g, RadixGroupConfig))
    if len(groups) != len(config.groups):
        raise ValueError(CONNECTOR_HINT)
    return groups


@dataclass(frozen=True)
class GroupCell:
    """One KV cache group's part of a pool cell."""

    # Position in OffloadingConfig.groups.
    group_index: int
    # KVCacheConfig group id (the offload key suffix).
    group_id: int
    # Byte offset inside the worker cell.
    offset: int
    cell_bytes: int
    # SWA groups with blocks smaller than the full chunk: the slot holds the
    # last sub_blocks chunks of sub_tokens tokens before the chunk's end.
    sub_blocks: int = 1
    sub_tokens: int = 0


@dataclass(frozen=True)
class PoolGeometry:
    kind: str
    # In OffloadingConfig.groups order; FULL has exactly one.
    members: tuple[GroupCell, ...]
    # Bytes per worker cell: the members back to back.
    cell_bytes: int
    # cell_bytes x num_copies.
    slot_bytes: int
    # slot_bytes rounded up to slot_align.
    slot_stride: int
    # SWA only: the window the index requires, in index blocks.
    window_blocks: int = 0
    # Aux pools only: share of cpu_bytes_to_use.
    ratio: float = 0.0

    @property
    def group_ids(self) -> tuple[int, ...]:
        return tuple(m.group_id for m in self.members)


@dataclass(frozen=True)
class SlotGeometry:
    # Worker cells per slot: world_size, or 1 with a replicated layout.
    num_copies: int
    slot_align: int
    # Of the full-attention group.
    tokens_per_block: int
    blocks_per_chunk: int
    # Also the index block size.
    tokens_per_chunk: int
    # Node-wide budget, the server's data_bytes.
    cpu_bytes_to_use: int
    # FULL first, then aux pools in POOL_KINDS order.
    pools: tuple[PoolGeometry, ...]

    @property
    def full(self) -> PoolGeometry:
        return self.pools[0]

    @property
    def swa(self) -> PoolGeometry | None:
        return self.pool("swa")

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(p.kind for p in self.pools)

    @property
    def num_groups(self) -> int:
        return sum(len(p.members) for p in self.pools)

    def pool(self, kind: str) -> PoolGeometry | None:
        return next((p for p in self.pools if p.kind == kind), None)

    def cell_of_group(self, group_index: int) -> tuple[PoolGeometry, GroupCell]:
        for p in self.pools:
            for m in p.members:
                if m.group_index == group_index:
                    return p, m
        raise KeyError(group_index)


def _lane_of(group: RadixGroupConfig) -> str:
    kind = group.kv_cache_spec_kind
    if kind in FULL_KINDS:
        return "full"
    if kind in ("sliding_window", "sliding_window_mla"):
        return "swa"
    if kind == "mamba":
        return "mamba"
    raise ValueError(
        f"RadixShmem cannot offload KV cache group {group.group_id} of kind "
        f"{kind!r}. " + HYBRID_HINT
    )


def _assign_lanes(groups: tuple[RadixGroupConfig, ...]) -> dict[str, list[int]]:
    """Pool kind -> indexes of its groups in ``groups``."""
    by_kind: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        by_kind.setdefault(_lane_of(group), []).append(i)
    full = by_kind.get("full", [])
    if len(full) != 1:
        raise ValueError(
            "RadixShmem needs exactly one full-attention KV cache group, got "
            f"{len(full)}. " + HYBRID_HINT
        )
    mamba = {groups[i].tokens_per_block for i in by_kind.get("mamba", [])}
    if len(mamba) > 1:
        raise ValueError(
            f"RadixShmem needs one block size for all mamba groups, got {mamba}"
        )
    return by_kind


def _check_mamba(group: RadixGroupConfig, full_group: RadixGroupConfig) -> None:
    # Mamba chunks are then whole full chunks (both scale by blocks_per_chunk),
    # so every stored state sits on an index block boundary.
    if group.tokens_per_block % full_group.tokens_per_block:
        raise ValueError(
            f"RadixShmem needs the mamba block ({group.tokens_per_block} tokens) to "
            f"be a multiple of the full-attention block "
            f"({full_group.tokens_per_block})"
        )


def _swa_member(
    group: RadixGroupConfig, full_group: RadixGroupConfig, bpc: int
) -> tuple[int, int, int]:
    """``(index window, sub_blocks, sub_tokens)`` of a sliding-window group,
    after the scheduler window checks."""
    sw = group.sliding_window
    if not sw or sw <= 0:
        raise ValueError(f"RadixShmem: group {group.group_id} has no sliding_window")
    if bpc != 1:
        # A chunk may then be stored with its out-of-window blocks skipped.
        raise ValueError(
            f"RadixShmem needs blocks_per_chunk 1 with sliding-window groups, got {bpc}"
        )
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    need = SlidingWindowManager._contiguous_blocks_for_hit(
        sw + group.extra_retained_tokens, group.tokens_per_block, False
    )
    window = SlidingWindowManager._contiguous_blocks_for_hit(
        sw, group.tokens_per_block, False
    )
    if window <= 0:
        raise ValueError("RadixShmem requires an SWA window with cached history")
    chunk, sub = full_group.tokens_per_block * bpc, group.tokens_per_block * bpc
    if sub == chunk:
        return window, 1, 0
    # Smaller blocks: the window must fit in one full chunk.
    if chunk % sub or need * sub > chunk:
        raise ValueError(
            f"RadixShmem: group {group.group_id} has {sub}-token chunks and a "
            f"{window}-chunk window; it must divide and fit in the {chunk}-token "
            "full-attention chunk."
        )
    return 1, need, sub


def _packed(layout: str | None) -> bool:
    """Block-outermost layouts move each group's block as the whole packed
    block (offloading/worker.py), so every group's cell is the full stride."""
    from vllm.config.cache import _layout_from_name

    return layout is not None and _layout_from_name(layout).is_block_outermost


def server_geometry(rs: Any, geometry: SlotGeometry) -> Any:
    """The layout in the library's configuration format."""
    kwargs: dict[str, Any] = {
        "block_size": geometry.tokens_per_chunk,
        "slot_align": geometry.slot_align,
    }
    for pool in geometry.pools:
        kwargs[f"{pool.kind}_slot_bytes"] = pool.slot_bytes
        if pool.kind == "swa":
            kwargs["swa_window_blocks"] = pool.window_blocks
    return rs.Geometry(**kwargs)


def plan_slots(geometry: SlotGeometry) -> dict[str, int]:
    """The server's slot plan, without creating shared resources."""
    import radixshmem
    from radixshmem._geometry import plan

    cfg = radixshmem.ServerConfig(
        data_bytes=geometry.cpu_bytes_to_use,
        slot_align=geometry.slot_align,
        **{f"{p.kind}_ratio": p.ratio for p in geometry.pools[1:]},
    )
    planned = plan(server_geometry(radixshmem, geometry).to_dict(), cfg)
    return {kind: int(pool["num_slots"]) for kind, pool in planned["pools"].items()}


def _apply_ratios(
    pools: list[PoolGeometry], extra: Mapping[str, Any]
) -> list[PoolGeometry]:
    """Aux pools' budget share: equal slot counts by default, or
    ``<kind>_pool_ratio`` from the extra config."""
    total_stride = sum(p.slot_stride for p in pools)
    out = [pools[0]]
    for p in pools[1:]:
        key = f"{p.kind}_pool_ratio"
        raw = extra.get(key)
        ratio = p.slot_stride / total_stride if raw is None else float(raw)
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"{key} must be in (0, 1), got {raw!r}")
        out.append(replace(p, ratio=ratio))
    if sum(p.ratio for p in out) >= 1.0:
        raise ValueError("the pool ratios leave no budget for FULL slots")
    return out


def compute_geometry(
    config: OffloadingConfig, extra: Mapping[str, Any]
) -> SlotGeometry:
    """The slot geometry; FULL sizing mirrors ``CPUOffloadingSpec``.

    Raises ``ValueError`` for unsupported configurations.
    """
    parallel = config.parallel
    if parallel.pp_size != 1:
        raise ValueError(f"RadixShmem needs pp_size == 1, got {parallel.pp_size}")
    if config.canonical_layout:
        raise ValueError("RadixShmem does not support canonical_layout")
    if parallel.world_size <= 0:
        raise ValueError(f"invalid world_size {parallel.world_size}")
    if not config.groups:
        raise ValueError("RadixShmem got no KV cache groups")

    groups = radix_groups(config)
    lanes = _assign_lanes(groups)
    full_group = groups[lanes["full"][0]]
    single_group = len(groups) == 1
    blocks_per_chunk = config.cache.blocks_per_chunk
    tokens_per_block = full_group.tokens_per_block
    num_copies = 1 if config.replicated_layout else parallel.world_size

    slot_align = int(extra.get("slot_align", DEFAULT_SLOT_ALIGN))
    if slot_align < mmap.PAGESIZE or slot_align & (slot_align - 1):
        raise ValueError(
            f"slot_align must be a power of two >= {mmap.PAGESIZE}, got {slot_align}"
        )

    whole_block = single_group or _packed(config.kv_cache_layout)
    pools: list[PoolGeometry] = []
    for kind in POOL_KINDS:
        if kind not in lanes:
            continue
        members: list[GroupCell] = []
        windows: set[int] = set()
        offset = 0
        for g_index in lanes[kind]:
            group = groups[g_index]
            # The allocation stride, as the native backend sizes it.
            per_block = (
                config.worker_kv_bytes_per_block
                if whole_block
                else group.kv_bytes_per_block
            )
            if per_block <= 0:
                raise ValueError(
                    f"RadixShmem: no KV bytes per block for group {group.group_id}"
                )
            window, sub_blocks, sub_tokens = (
                _swa_member(group, full_group, blocks_per_chunk)
                if kind == "swa"
                else (0, 1, 0)
            )
            windows.add(window)
            cell = per_block * blocks_per_chunk * sub_blocks
            members.append(
                GroupCell(g_index, group.group_id, offset, cell, sub_blocks, sub_tokens)
            )
            offset += cell
        if len(windows) > 1:
            raise ValueError(
                f"RadixShmem needs the sliding-window groups to share one index "
                f"window, got {sorted(windows)} full chunks. " + HYBRID_HINT
            )
        slot_bytes = offset * num_copies
        first = groups[lanes[kind][0]]
        if kind == "mamba":
            _check_mamba(first, full_group)
            # The library checks the mamba state size against the store stride.
            slot_bytes = round_up(slot_bytes, slot_align)
        pools.append(
            PoolGeometry(
                kind=kind,
                members=tuple(members),
                cell_bytes=offset,
                slot_bytes=slot_bytes,
                slot_stride=round_up(slot_bytes, slot_align),
                window_blocks=windows.pop(),
            )
        )

    if not extra.get("cpu_bytes_to_use"):
        raise ValueError("cpu_bytes_to_use must be set in kv_connector_extra_config")
    cpu_bytes_to_use = int(extra["cpu_bytes_to_use"])
    geometry = SlotGeometry(
        num_copies=num_copies,
        slot_align=slot_align,
        tokens_per_block=tokens_per_block,
        blocks_per_chunk=blocks_per_chunk,
        tokens_per_chunk=tokens_per_block * blocks_per_chunk,
        cpu_bytes_to_use=cpu_bytes_to_use,
        pools=tuple(_apply_ratios(pools, extra)),
    )
    try:
        plan_slots(geometry)
    except ValueError as exc:
        raise ValueError(
            f"RadixShmem: cannot plan cpu_bytes_to_use={cpu_bytes_to_use}: {exc}"
        ) from exc
    return geometry


def layout_fingerprint(
    geometry: SlotGeometry, config: OffloadingConfig
) -> dict[str, Any]:
    """The layout fields the radixshmem server cannot validate."""
    return {
        "num_copies": geometry.num_copies,
        "slot_align": geometry.slot_align,
        "tokens_per_block": geometry.tokens_per_block,
        "blocks_per_chunk": geometry.blocks_per_chunk,
        "tokens_per_chunk": geometry.tokens_per_chunk,
        "kv_cache_layout": config.kv_cache_layout,
        "model": config.model.name,
        "cache_formats": [
            [g.group_id, list(g.cache_format)] for g in radix_groups(config)
        ],
        "pools": {
            p.kind: {
                "groups": [[m.group_id, m.offset, m.cell_bytes] for m in p.members],
                "cell_bytes": p.cell_bytes,
                "slot_bytes": p.slot_bytes,
                "slot_stride": p.slot_stride,
                "window_blocks": p.window_blocks,
            }
            for p in geometry.pools
        },
    }
