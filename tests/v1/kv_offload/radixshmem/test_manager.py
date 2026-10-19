# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix leases and reservations against a real shared radix index."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.kv_offload.radixshmem.manager import Chunk
from vllm.v1.kv_offload.radixshmem.remote import RemotePulls
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from .utils import (
    block_hashes,
    cleanup_shm,
    default_cpu_bytes,
    hit_len,
    joint_hit,
    make_extra,
    open_manager,
    open_pool,
    peer_publish,
    probe,
    ref_total,
    swa_used,
    to_keys,
    two_group_cpu_bytes,
    two_groups,
    unique_name,
)

radixshmem = pytest.importorskip("radixshmem")


def _drain_remote(manager):
    deadline = time.monotonic() + 5
    while manager.has_pending_work():
        assert time.monotonic() < deadline, "remote lookup did not finish"
        manager.step()
        time.sleep(0.001)


@pytest.fixture
def name():
    n = unique_name("vllm_mgr")
    try:
        yield n
    finally:
        cleanup_shm(n)


@pytest.fixture
def extra(name):
    return make_extra(name, cpu_bytes_to_use=default_cpu_bytes())


@pytest.fixture
def pool(extra):
    # The owner worker's part: the server and the sentinel; managers attach.
    with open_pool(extra) as p:
        yield p


@pytest.fixture
def manager(extra, pool):
    with open_manager(extra) as m:
        yield m


@pytest.fixture
def client(name, manager):
    with probe(name) as c:
        yield c
        # Every test must leave the tree without leaked slots or pins.
        assert c.check_health() == 0, c.health_report().describe()


@pytest.fixture
def extra2(name):
    """A full-attention group (id 0) and a sliding-window group (id 1) with a
    2-chunk window. The SWA key of chunk i carries the full key's block hash
    with group suffix 1, so both lanes are positions of one tree. Calls are
    shaped like the scheduler's: keys recorded group by group, store lists
    group-major and possibly gapped inside the SWA group, window lookups
    scanned from the end."""
    return make_extra(name, cpu_bytes_to_use=two_group_cpu_bytes())


@pytest.fixture
def manager2(extra2):
    with (
        open_pool(extra2, groups=two_groups()),
        open_manager(extra2, groups=two_groups()) as m,
    ):
        yield m


@pytest.fixture
def client2(name, manager2):
    with probe(name) as c:
        yield c
        # Neither lane may leak slots or pins.
        assert c.check_health() == 0, c.health_report().describe()


def test_reservation_skips_the_head_a_peer_published(manager, client):
    keys = to_keys(4)
    hashes = block_hashes(keys)
    assert manager.lookup_and_pin("r", hashes, 4 * manager.tokens_per_chunk) is None
    peer_slots = peer_publish(client, keys[:2])
    out = manager.reserve("r", hashes, [Chunk(0, i) for i in range(4)])
    assert out is not None and out.chunks == [Chunk(0, 2), Chunk(0, 3)]
    assert client.mempool_used() == 4
    manager.complete_store(out)
    assert hit_len(client, keys) == 4
    lease = manager.lookup_and_pin("r", hashes, 4 * manager.tokens_per_chunk)
    assert lease is not None
    assert lease.slots[:2].tolist() == peer_slots.tolist()
    assert lease.slots[2:].tolist() == out.store_spec.chunk_ids.tolist()
    lease.release()
    manager.finish_request("r")


@pytest.mark.parametrize(
    ("n", "head", "early", "used_before", "joints"),
    [
        (8, ((0, 4), (2, 4)), ((4, 8), (6, 8)), (8, 4), {4: (4, 2), 8: (8, 6)}),
        (4, ((0, 4), (0, 0)), ((0, 0), (2, 4)), (4, 2), {4: (4, 2)}),
    ],
    ids=["later_range_first", "swa_only_job_first"],
)
def test_swa_publish_waits_for_full(
    manager2, client2, n, head, early, used_before, joints
):
    keys = to_keys(n)
    hashes = block_hashes(keys)
    jobs = [
        [Chunk(g, i) for g, (a, b) in enumerate(ranges) for i in range(a, b)]
        for ranges in (head, early)
    ]
    head_out, early_out = (manager2.reserve("r", hashes, job) for job in jobs)
    assert head_out is not None and early_out is not None
    manager2.complete_store(early_out)
    assert hit_len(client2, keys) == 0
    assert joint_hit(client2, keys)[0] == 0
    assert (client2.mempool_used(), swa_used(client2)) == used_before
    manager2.complete_store(head_out)
    assert hit_len(client2, keys) == n
    for end, expected in joints.items():
        assert joint_hit(client2, keys[:end])[:2] == expected
    manager2.finish_request("r")


def test_capped_prefix_pins_its_own_swa_window(manager2, client2):
    keys = to_keys(8)
    hashes = block_hashes(keys)
    output = manager2.reserve(
        "writer", hashes, [Chunk(g, i) for g in (0, 1) for i in range(8)]
    )
    assert output is not None
    manager2.complete_store(output)
    manager2.finish_request("writer")
    assert joint_hit(client2, keys)[:2] == (8, 6)
    lease = manager2.lookup_and_pin("reader", hashes, 4 * manager2.tokens_per_chunk)
    assert lease is not None and lease.hit == 4
    load = manager2.load_slots(
        lease, [Chunk(0, i) for i in range(4)] + [Chunk(1, i) for i in range(2, 4)]
    )
    assert len(load.chunk_ids) == 6 and ref_total(client2) > 0
    lease.release()
    manager2.finish_request("reader")
    assert ref_total(client2) == 0


def test_reset_rejects_reservations_and_leases_without_deferring(manager, client):
    keys, other = to_keys(4), to_keys(2, salt=1)
    hashes, other_hashes = block_hashes(keys), block_hashes(other)
    chunks = [Chunk(0, i) for i in range(4)]
    tokens = 4 * manager.tokens_per_chunk

    out = manager.reserve("r", hashes, chunks)
    assert out is not None
    peer_publish(client, other)
    assert not manager.reset()
    manager.complete_store(out)
    assert hit_len(client, keys) == 4
    lease = manager.lookup_and_pin("q", other_hashes, tokens)
    assert lease is not None and ref_total(client) > 0
    assert not manager.reset()
    lease.release()
    manager.finish_request("q")
    assert manager.reset()
    assert (hit_len(client, keys), hit_len(client, other)) == (0, 0)
    assert (client.mempool_used(), ref_total(client)) == (0, 0)

    # The pool is usable afterwards.
    out = manager.reserve("r", hashes, chunks)
    assert out is not None
    manager.complete_store(out)
    assert hit_len(client, keys) == 4
    manager.finish_request("r")


def test_library_pull_materializes_remote_bytes_before_loading(
    manager, client, monkeypatch
):
    """Exercise real pull_async/publish/pins; simulate discovery and the wire."""
    source_name = unique_name("vllm_source")
    source_extra = make_extra(source_name, cpu_bytes_to_use=default_cpu_bytes())
    keys = to_keys(3)
    try:
        with open_pool(source_extra), probe(source_name) as source:
            source_slots = peer_publish(source, keys)
            for i, slot in enumerate(source_slots):
                view = source.slot_view(int(slot))
                view[:] = bytes([i + 17]) * len(view)

            target = manager.index.client
            local_index, rpc = target.index, target._t

            class Discovery:
                def __getattr__(self, name):
                    return getattr(local_index, name)

                def rank(self):
                    return 1

                def query(self, hashes, **kwargs):
                    if kwargs.get("local_only"):
                        return local_index.query(hashes, **kwargs)
                    query = source.query(hashes, **{**kwargs, "local_only": True})
                    return SimpleNamespace(
                        status=query.status,
                        common_hit=query.common_hit,
                        full_fragments=[(0, 0, source_slots)],
                        swa_source_rank=-1,
                        swa_slots=np.empty(0, dtype=np.int32),
                        mamba=None,
                        finalize=query.finalize,
                    )

            class Wire:
                def __getattr__(self, name):
                    return getattr(rpc, name)

                def submit(self, items, timeout_ms):
                    replies = []
                    for rank, kind, remote_slots, local_slots in items:
                        assert rank == 0
                        for src, dst in zip(remote_slots, local_slots):
                            target.slot_view(dst)[:] = source.slot_view(src)
                        replies.append(
                            (
                                True,
                                len(local_slots) * source.store.slot_bytes,
                                len(local_slots),
                                "",
                            )
                        )
                    self.replies = replies
                    return 1

                def wait_any(self, tickets, timeout_ms):
                    return {"done": {1: self.replies}, "unknown": []}

            monkeypatch.setattr(target, "_index", Discovery())
            monkeypatch.setattr(target, "_t", Wire())
            manager._remote = RemotePulls(target, timeout_s=30, max_pending=2)
            hashes = block_hashes(keys)
            assert (
                manager.lookup_and_pin(
                    "library-pull", hashes, 3 * manager.tokens_per_chunk
                )
                is None
            )
            assert manager.pending("library-pull")
            assert not manager.reset()
            _drain_remote(manager)
            lease = manager.lookup_and_pin(
                "library-pull", hashes, 3 * manager.tokens_per_chunk
            )
            assert lease is not None and lease.hit == 3
            assert manager.get_load_source("library-pull", 0) is CacheHitSource.P2P
            load = manager.load_slots(lease, [Chunk(0, i) for i in range(3)])
            for i, slot in enumerate(load.chunk_ids):
                expected = bytes([i + 17]) * source.store.slot_bytes
                assert bytes(client.slot_view(int(slot))) == expected
            assert ref_total(source) == 0
            assert ref_total(client) > 0
            lease.release()
            manager.finish_request("library-pull")
            assert ref_total(client) == 0
    finally:
        cleanup_shm(source_name)
