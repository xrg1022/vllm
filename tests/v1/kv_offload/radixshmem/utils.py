# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the radixshmem offloading tests. Two-group configs are
group 0 full attention and group 1 sliding window; an SWA key carries the
full key's block hash with group suffix 1. Index keys are the first 8 bytes
of the block hash (little-endian), so probes can observe publishes.
"""

import contextlib
import os
import uuid
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.base import (
    OffloadKey,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.radixshmem.geometry import RadixGroupConfig
from vllm.v1.kv_offload.radixshmem.manager import RadixShmemManager

radixshmem = pytest.importorskip("radixshmem")

TOKENS_PER_BLOCK = 16
SLOT_ALIGN = 4096
# A few hundred KiB keep /dev/shm small and make pool-exhausted cases cheap.
DEFAULT_NUM_SLOTS = 64
# Two-group defaults: a 32-token window is 2 chunks of 16 tokens.
SLIDING_WINDOW = 32
FULL_GROUP_ID = 0
SWA_GROUP_ID = 1


def unique_name(prefix: str = "vllm_t") -> str:
    return f"/{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def socket_path(name: str) -> str:
    return f"/dev/shm/{name.lstrip('/')}.sock"


def sentinel_path(name: str) -> str:
    return f"/dev/shm/{name.lstrip('/')}.vllm.json"


def index_shm_path(name: str) -> str:
    return f"/dev/shm/{name.lstrip('/')}"


def data_shm_path(name: str) -> str:
    return f"/dev/shm/{name.lstrip('/')}_data"


def shm_leftovers(name: str) -> list[str]:
    """Paths under /dev/shm that still exist for ``name``."""
    return [
        p
        for p in (
            index_shm_path(name),
            data_shm_path(name),
            socket_path(name),
            sentinel_path(name),
        )
        if os.path.exists(p)
    ]


def cleanup_shm(name: str) -> None:
    for p in shm_leftovers(name):
        with contextlib.suppress(OSError):
            os.unlink(p)
    with contextlib.suppress(OSError):
        os.unlink(sentinel_path(name) + ".lock")


def make_extra(
    name: str,
    *,
    cpu_bytes_to_use: int,
    endpoint: str | None = None,
    attach_timeout_s: float = 10.0,
    **overrides,
) -> dict:
    extra = {
        "index_shm_name": name,
        "cpu_bytes_to_use": cpu_bytes_to_use,
        "slot_align": SLOT_ALIGN,
        # Deterministic pool accounting: no background evictor thread.
        "background_evict_ratio": 0.0,
        "attach_timeout_s": attach_timeout_s,
        "prefault": False,
    }
    if endpoint is not None:
        extra["endpoint"] = endpoint
    extra.update(overrides)
    return extra


def make_config(
    extra: dict,
    *,
    worker_kv_bytes_per_block: int = SLOT_ALIGN,
    blocks_per_chunk: int = 1,
    world_size: int = 1,
    rank: int = 0,
    pp_size: int = 1,
    data_parallel_index: int = 0,
    data_parallel_size: int = 1,
    tokens_per_block: int = TOKENS_PER_BLOCK,
    tokens_per_hash: int | None = None,
    groups: tuple[RadixGroupConfig, ...] | None = None,
    replicated_layout: bool = False,
    canonical_layout: bool = False,
    enable_kv_cache_events: bool = False,
) -> OffloadingConfig:
    """Same shape as tests/v1/kv_offload/test_factory.py::_make_offloading_config."""
    if groups is None:
        groups = (
            RadixGroupConfig(
                tokens_per_block=tokens_per_block,
                layer_names=("layer",),
                group_id=0,
                kv_cache_spec_kind="full_attention",
                cache_format=("torch.float16", "NONE", ""),
            ),
        )
    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config=dict(extra),
        engine_id=f"test-engine-dp{data_parallel_index}",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(
            tokens_per_hash=(
                tokens_per_block if tokens_per_hash is None else tokens_per_hash
            ),
            blocks_per_chunk=blocks_per_chunk,
        ),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=max(world_size // pp_size, 1),
            pp_size=pp_size,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=data_parallel_index,
            data_parallel_size=data_parallel_size,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
        replicated_layout=replicated_layout,
        canonical_layout=canonical_layout,
        kv_cache_layout="LBNHC",
    )


def default_cpu_bytes(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    worker_kv_bytes_per_block: int = SLOT_ALIGN,
    blocks_per_chunk: int = 1,
    world_size: int = 1,
) -> int:
    """A budget that yields exactly ``num_slots`` slots of the given shape."""
    raw = worker_kv_bytes_per_block * blocks_per_chunk * world_size
    stride = -(-raw // SLOT_ALIGN) * SLOT_ALIGN
    return num_slots * stride


def two_group_cpu_bytes(
    *,
    num_slots: int = DEFAULT_NUM_SLOTS,
    full_bytes_per_block: int = SLOT_ALIGN,
    swa_bytes_per_block: int = SLOT_ALIGN // 2,
    blocks_per_chunk: int = 1,
    world_size: int = 1,
) -> int:
    """A budget that yields ``num_slots`` FULL and ``num_slots`` SWA slots with
    the default equal-count split."""
    total = 0
    for per_block in (full_bytes_per_block, swa_bytes_per_block):
        raw = per_block * blocks_per_chunk * world_size
        total += -(-raw // SLOT_ALIGN) * SLOT_ALIGN
    return num_slots * total


def two_groups(
    *,
    tokens_per_block: int = TOKENS_PER_BLOCK,
    full_bytes_per_block: int = SLOT_ALIGN,
    swa_bytes_per_block: int = SLOT_ALIGN // 2,
    sliding_window: int = SLIDING_WINDOW,
    extra_retained_tokens: int = 0,
    swa_first: bool = False,
) -> tuple[RadixGroupConfig, ...]:
    """A full-attention group (id 0) and a sliding-window group (id 1)."""
    full = RadixGroupConfig(
        tokens_per_block=tokens_per_block,
        layer_names=("full_layer",),
        group_id=FULL_GROUP_ID,
        kv_cache_spec_kind="full_attention",
        cache_format=("torch.float16", "NONE", ""),
        kv_bytes_per_block=full_bytes_per_block,
    )
    swa = RadixGroupConfig(
        tokens_per_block=tokens_per_block,
        layer_names=("swa_layer",),
        group_id=SWA_GROUP_ID,
        kv_cache_spec_kind="sliding_window",
        cache_format=("torch.float16", "NONE", ""),
        kv_bytes_per_block=swa_bytes_per_block,
        sliding_window=sliding_window,
        extra_retained_tokens=extra_retained_tokens,
    )
    return (swa, full) if swa_first else (full, swa)


def make_spec(extra: dict, **config_kwargs):
    from vllm.v1.kv_offload.radixshmem.backend import RadixShmemBackend

    return RadixShmemBackend(make_config(extra, **config_kwargs))


@contextlib.contextmanager
def open_pool(extra: dict, *, owner: bool = True, **config_kwargs):
    """The worker side of ``get_worker`` without CUDA or pinning.

    ``owner=True`` is the DP 0 / TP 0 worker: it starts the server (or, with
    ``endpoint`` set, writes or adopts the sentinel) and attaches the store.
    ``owner=False`` is any other worker: a read-only attach. Yields the
    ``RadixRegions``; closing the owner's destroys the pool.
    """
    from vllm.v1.kv_offload.radixshmem.bootstrap import RadixRegions

    spec = make_spec(extra, **config_kwargs)
    regions = RadixRegions.open_worker(
        spec.geometry, spec.settings, spec.fingerprint(), owner=owner
    )
    try:
        yield regions
    finally:
        regions.close()


@contextlib.contextmanager
def open_manager(extra: dict, **config_kwargs) -> Iterator[RadixShmemManager]:
    """Build the spec, get its manager, and shut the manager down on exit.

    The manager only attaches; create the pool with ``open_pool`` first.
    """
    manager = make_spec(extra, **config_kwargs).open_manager()
    try:
        yield manager
    finally:
        manager.shutdown()


def to_hash(i: int, salt: int = 0) -> bytes:
    """A 32-byte block hash whose first 8 bytes encode (salt, i)."""
    return ((salt << 32) | i).to_bytes(8, "little") + b"\x00" * 24


def to_key(i: int, salt: int = 0, group_idx: int = 0) -> OffloadKey:
    return make_offload_key(to_hash(i, salt), group_idx)


def to_keys(n: int, salt: int = 0) -> list[OffloadKey]:
    return [to_key(i, salt) for i in range(n)]


def swa_key(i: int, salt: int = 0) -> OffloadKey:
    """The sliding-window group's key of chunk ``i``: same hash, group 1."""
    return to_key(i, salt, SWA_GROUP_ID)


def swa_keys(n: int, salt: int = 0) -> list[OffloadKey]:
    return [swa_key(i, salt) for i in range(n)]


def to_u64(keys: Sequence[OffloadKey]) -> np.ndarray:
    return np.array(
        [int.from_bytes(get_offload_block_hash(k)[:8], "little") for k in keys],
        dtype=np.uint64,
    )


def block_hashes(keys: Sequence[OffloadKey]) -> list[BlockHash]:
    return [BlockHash(get_offload_block_hash(key)) for key in keys]


@contextlib.contextmanager
def probe(name: str, endpoint: str | None = None, timeout_s: float = 10.0):
    """An independent radixshmem client on the same server, used to observe
    the index from outside the code under test (and to act as a peer)."""
    client = radixshmem.RadixClient(name, endpoint=endpoint)
    try:
        client.wait_ready(timeout_s)
        _ = client.index  # attach now so failures surface here
        yield client
    finally:
        client.close()


def hit_len(client, keys: Sequence[OffloadKey]) -> int:
    if not keys:
        return 0
    r = client.query(
        to_u64(keys),
        mask=radixshmem.COMPONENT_MASK_FULL,
        local_only=True,
        lock=False,
        update_meta=False,
    )
    return int(r.common_hit)


def joint_hit(client, keys: Sequence[OffloadKey]) -> tuple[int, int, list[int]]:
    """``(common_hit, swa_start, swa_slots)`` of a FULL|SWA query on ``keys``."""
    if not keys:
        return 0, 0, []
    r = client.query(
        to_u64(keys),
        mask=radixshmem.COMPONENT_MASK_FULL | radixshmem.COMPONENT_MASK_SWA,
        local_only=True,
        lock=False,
        update_meta=False,
    )
    return int(r.common_hit), int(r.swa_start), [int(s) for s in r.swa_slots]


def swa_used(client) -> int:
    """SWA slots allocated or published, from the index's audit walk."""
    return int(client.health_report().swa.used)


def ref_total(client) -> int:
    """Pins still held in the shared tree (0 when every lease is released)."""
    return int(client.health_report().ref_total)


def peer_publish(client, keys: Sequence[OffloadKey], start: int = 0) -> np.ndarray:
    """Act as another DP rank: allocate and publish ``keys[start:]``."""
    n = len(keys) - start
    slots = client.allocate_slots(n)
    assert slots.size == n, "peer could not allocate"
    res = client.insert(to_u64(keys), slots, start=start, auto_recycle=True)
    assert int(res.error) == 0, res.error
    assert res.unused_slots.size == 0
    return slots


@contextlib.contextmanager
def external_server(name: str, sock_dir, *, cpu_bytes_to_use: int | None = None):
    """A standalone ``radix-server`` equivalent, started in-process, for the
    ``endpoint`` (nobody-owns) mode. Yields ``(server, endpoint)``."""
    endpoint = f"unix://{sock_dir}/radix.sock"
    srv = radixshmem.RadixServer(
        radixshmem.ServerConfig(
            name=name,
            data_bytes=default_cpu_bytes()
            if cpu_bytes_to_use is None
            else cpu_bytes_to_use,
            slot_align=SLOT_ALIGN,
            prefault=False,
            background_evict_ratio=0.0,
            endpoint=endpoint,
            rpc_workers=4,
        )
    ).start()
    try:
        yield srv, endpoint
    finally:
        srv.close()
