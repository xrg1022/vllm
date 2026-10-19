# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner / attacher bootstrap of the node-wide pool: the DP 0 / TP 0 worker
creates it (``open_pool``), schedulers and other workers attach and check the
layout sentinel ``/dev/shm/<name>.vllm.json``.
"""

import json
import os
from contextlib import closing
from dataclasses import replace

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_kv_cache_config,
    _make_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.config import (
    build_radix_offloading_config,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.radixshmem.backend import RadixShmemBackend
from vllm.v1.kv_offload.radixshmem.bootstrap import RadixRegions
from vllm.v1.kv_offload.radixshmem.manager import Chunk

from .utils import (
    DEFAULT_NUM_SLOTS,
    SLOT_ALIGN,
    TOKENS_PER_BLOCK,
    block_hashes,
    cleanup_shm,
    default_cpu_bytes,
    hit_len,
    make_config,
    make_extra,
    open_manager,
    open_pool,
    peer_publish,
    probe,
    sentinel_path,
    shm_leftovers,
    socket_path,
    to_keys,
    to_u64,
    unique_name,
)
from .utils import (
    external_server as _external_server,
)

radixshmem = pytest.importorskip("radixshmem")


def test_remote_replica_bootstraps_its_pool_and_separates_cache_domains(
    name, external_server
):
    # In remote mode a replica with a nonzero DP rank owns its node's pool.
    _, endpoint = external_server
    extra = _extra(name, endpoint=endpoint, remote=True, cache_namespace="weights-v1")
    dp3 = {"data_parallel_index": 3, "data_parallel_size": 4}
    keys = to_keys(2)
    with open_pool(extra, **dp3), open_manager(extra, **dp3) as first:
        out = first.reserve(
            "writer", block_hashes(keys), [Chunk(0, i) for i in range(len(keys))]
        )
        assert out is not None
        first.complete_store(out)
        with open_manager(extra) as same:
            assert same.index.lookup(to_u64(keys)) == 2
        with open_manager({**extra, "cache_namespace": "weights-v2"}) as other:
            assert other.index.lookup(to_u64(keys)) == 0
        first.finish_request("writer")
    groups = (
        replace(
            make_config(extra).groups[0], cache_format=("torch.bfloat16", "NONE", "")
        ),
    )
    with open_pool(extra, groups=groups), open_manager(extra, groups=groups) as other:
        assert other.index.lookup(to_u64(keys)) == 0


SENTINEL_KEYS = {
    "version",
    "owner_pid",
    "num_copies",
    "slot_align",
    "blocks_per_chunk",
    "tokens_per_block",
    "tokens_per_chunk",
    "pools",
    "pool_slots",
    "num_slots",
    "kv_cache_layout",
    "model",
    "cache_formats",
}


@pytest.fixture(autouse=True)
def _none_hash_initialized():
    # Engine-core order; the deterministic sha256 seed is identical in every process.
    init_none_hash(sha256)


@pytest.fixture
def name():
    n = unique_name("vllm_bs")
    try:
        yield n
    finally:
        cleanup_shm(n)


def _extra(name: str, **overrides) -> dict:
    return make_extra(name, cpu_bytes_to_use=default_cpu_bytes(), **overrides)


def _read_sentinel(name: str) -> dict:
    with open(sentinel_path(name)) as f:
        return json.load(f)


def test_attach_checks_resolved_cache_format(name):
    config = _make_vllm_config(extra_config=_extra(name))
    config.cache_config.cache_dtype = "auto"
    config.cache_config.kv_cache_layout = None
    config.model_config.dtype = torch.float16
    worker_config = _make_kv_cache_config()
    spec = replace(worker_config.kv_cache_groups[0].kv_cache_spec, dtype=torch.float16)
    worker_config.kv_cache_groups[0].kv_cache_spec = UniformTypeKVCacheSpecs(
        spec.block_size, {"layer": spec}
    )
    owner = RadixShmemBackend(build_radix_offloading_config(config, worker_config))
    # Worker-side config strings may resolve after the specs have been collected.
    config.cache_config.cache_dtype = "fp8"
    scheduler = RadixShmemBackend(
        build_radix_offloading_config(
            config, generate_scheduler_kv_cache_config([worker_config])
        )
    )
    assert owner.fingerprint() == scheduler.fingerprint()
    with closing(
        RadixRegions.open_worker(
            owner.geometry, owner.settings, owner.fingerprint(), owner=True
        )
    ):
        scheduler.open_manager().shutdown()
        worker_config.kv_cache_groups[0].kv_cache_spec = replace(
            spec, dtype=torch.bfloat16
        )
        other = RadixShmemBackend(build_radix_offloading_config(config, worker_config))
        assert other.geometry == owner.geometry
        with pytest.raises(RuntimeError, match="cache_formats"):
            other.open_manager()
        with pytest.raises(RuntimeError, match="cache_formats"):
            RadixRegions.open_worker(
                other.geometry, other.settings, other.fingerprint(), owner=False
            )


def test_owner_worker_creates_the_pool_and_a_scheduler_attaches(name):
    extra = make_extra(name, cpu_bytes_to_use=default_cpu_bytes(world_size=2))
    with open_pool(extra, world_size=2) as pool:
        assert pool.is_owner
        assert pool.client.local_read_only
        assert os.path.exists(socket_path(name))
        sentinel = _read_sentinel(name)
        assert set(sentinel) >= SENTINEL_KEYS, sorted(sentinel)
        assert sentinel["version"] == 2
        assert sentinel["owner_pid"] == os.getpid()
        assert sentinel["num_copies"] == 2
        assert sentinel["blocks_per_chunk"] == 1
        assert sentinel["tokens_per_block"] == TOKENS_PER_BLOCK
        assert set(sentinel["pools"]) == {"full"}
        assert sentinel["pools"]["full"]["cell_bytes"] == SLOT_ALIGN
        assert sentinel["pools"]["full"]["slot_stride"] == 2 * SLOT_ALIGN
        assert sentinel["pools"]["full"]["groups"] == [[0, 0, SLOT_ALIGN]]
        assert sentinel["pool_slots"] == {"full": DEFAULT_NUM_SLOTS}
        assert sentinel["model"] == "test-model"
        assert sentinel["num_slots"] == pool.num_slots == DEFAULT_NUM_SLOTS

        with probe(name) as client:
            assert client.mempool_total() == DEFAULT_NUM_SLOTS
            assert client.store.slot_bytes == sentinel["pools"]["full"]["slot_stride"]
            assert client.block_size() == TOKENS_PER_BLOCK

        with open_manager(extra, world_size=2) as manager:
            assert manager.index.pool_total("full") == DEFAULT_NUM_SLOTS
        # The scheduler leaving keeps the pool and the sentinel.
        assert os.path.exists(sentinel_path(name))
        assert os.path.exists(socket_path(name))
        with probe(name) as client:
            assert client.mempool_total() == DEFAULT_NUM_SLOTS

    # Close order: client, server (store + index gone), sentinel; no leftovers.
    assert shm_leftovers(name) == []


@pytest.fixture
def external_server(name, tmp_path):
    with _external_server(name, tmp_path) as (srv, endpoint):
        yield srv, endpoint


def test_external_endpoint_mode_attaches_without_owning(name, external_server):
    srv, endpoint = external_server
    assert srv.mode == "waiting"
    extra = _extra(name, endpoint=endpoint)
    keys = to_keys(4)
    with open_pool(extra, data_parallel_index=0, data_parallel_size=2) as pool:
        # The DP 0 worker configured the server and wrote the sentinel; no server.
        assert srv.mode == "ready"
        assert not pool.is_owner
        assert pool.owns_sentinel
        assert _read_sentinel(name)["owner_pid"] == os.getpid()
        with (
            open_manager(extra, data_parallel_index=0, data_parallel_size=2),
            open_manager(extra, data_parallel_index=1, data_parallel_size=2),
            open_pool(extra, owner=False, data_parallel_index=1, rank=0),
            probe(name, endpoint=endpoint) as client,
        ):
            peer_publish(client, keys)
    # Nobody owns the external server: it survives every vLLM shutdown.
    assert srv.mode == "ready"
    with probe(name, endpoint=endpoint) as client:
        assert hit_len(client, keys) == 4
    # No socket of our own was created next to the external one.
    assert not os.path.exists(socket_path(name))
    assert not os.path.exists(sentinel_path(name))
