# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two DP scheduler processes on one shared index, and owner recovery after a
SIGKILL (the new owner recreates the regions and overwrites the sentinel).
"""

import os
import signal
import time

import pytest

from vllm.utils.hashing import sha256
from vllm.utils.system_utils import get_mp_context
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_offload.radixshmem.manager import Chunk

from .utils import (
    block_hashes,
    cleanup_shm,
    default_cpu_bytes,
    hit_len,
    make_extra,
    make_spec,
    open_manager,
    open_pool,
    probe,
    sentinel_path,
    shm_leftovers,
    to_keys,
    unique_name,
)

pytest.importorskip("radixshmem")

TIMEOUT_S = 90


@pytest.fixture(autouse=True)
def _none_hash_initialized():
    init_none_hash(sha256)


@pytest.fixture
def name():
    n = unique_name("vllm_mp")
    try:
        yield n
    finally:
        cleanup_shm(n)


def _store(manager, keys, req_id):
    out = manager.reserve(
        req_id, block_hashes(keys), [Chunk(0, i) for i in range(len(keys))]
    )
    assert out is not None and len(out.chunks) == len(keys)
    manager.complete_store(out)
    manager.finish_request(req_id)
    return out.store_spec.chunk_ids.tolist()


def _attacher_proc(extra, keys_owner, keys_peer, to_child, to_parent):
    """DP rank 1: attach, read what DP 0 published, publish a prefix of its own."""
    try:
        init_none_hash(sha256)  # same deterministic seed as the parent
        with open_manager(extra, data_parallel_index=1, data_parallel_size=2) as m:
            assert to_child.get(timeout=TIMEOUT_S) == "owner_stored"
            lease = m.lookup_and_pin(
                "p1", block_hashes(keys_owner), len(keys_owner) * m.tokens_per_chunk
            )
            assert lease is not None
            hit = lease.hit
            load = m.load_slots(lease, [Chunk(0, i) for i in range(hit)])
            m.step()
            lease.release()
            m.finish_request("p1")
            to_parent.put(("hit", hit, load.chunk_ids.tolist()))

            _store(m, keys_peer, "p2")
            to_parent.put(("peer_stored", None, None))
            assert to_child.get(timeout=TIMEOUT_S) == "done"
        to_parent.put(("exit", None, None))
    except BaseException as exc:  # noqa: BLE001 - surface in the parent
        to_parent.put(("error", repr(exc), None))


@pytest.mark.timeout(240)
def test_two_scheduler_processes_share_one_index(name):
    extra = make_extra(name, cpu_bytes_to_use=default_cpu_bytes())
    keys_owner = to_keys(4, salt=1)
    keys_peer = to_keys(3, salt=2)
    ctx = get_mp_context()
    to_child, to_parent = ctx.Queue(), ctx.Queue()

    with (
        open_pool(extra, data_parallel_index=0, data_parallel_size=2),
        open_manager(extra, data_parallel_index=0, data_parallel_size=2) as owner,
    ):
        child = ctx.Process(
            target=_attacher_proc,
            args=(extra, keys_owner, keys_peer, to_child, to_parent),
        )
        child.start()
        try:
            owner_slots = _store(owner, keys_owner, "o1")
            to_child.put("owner_stored")

            kind, hit, slots = to_parent.get(timeout=TIMEOUT_S)
            assert kind == "hit", hit
            assert hit == 4
            assert slots == owner_slots

            kind, payload, _ = to_parent.get(timeout=TIMEOUT_S)
            assert kind == "peer_stored", payload
            lease = owner.lookup_and_pin(
                "o2", block_hashes(keys_peer), len(keys_peer) * owner.tokens_per_chunk
            )
            assert lease is not None and lease.hit == 3
            lease.release()
            owner.step()
            owner.finish_request("o2")

            to_child.put("done")
            kind, payload, _ = to_parent.get(timeout=TIMEOUT_S)
            assert kind == "exit", payload
            child.join(TIMEOUT_S)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.kill()
                child.join(5)

        with probe(name) as client:
            # 7 published slots; every pin the peer took was given back.
            assert client.mempool_used() == 7
            assert client.health_report().ref_total == 0
            assert client.check_health() == 0, client.health_report().describe()
    assert shm_leftovers(name) == []


def _owner_proc(extra, keys, to_parent):
    """Become the owner worker and DP 0 scheduler, publish a prefix, then
    block until killed."""
    try:
        with open_pool(extra, data_parallel_index=0):
            manager = make_spec(extra, data_parallel_index=0).open_manager()
            _store(manager, keys, "c1")
            to_parent.put(("ready", os.getpid()))
            time.sleep(600)
    except BaseException as exc:  # noqa: BLE001
        to_parent.put(("error", repr(exc)))


@pytest.mark.timeout(240)
def test_new_owner_recovers_after_the_old_one_is_killed(name):
    extra = make_extra(name, cpu_bytes_to_use=default_cpu_bytes())
    keys = to_keys(2, salt=7)
    ctx = get_mp_context()
    to_parent = ctx.Queue()
    child = ctx.Process(target=_owner_proc, args=(extra, keys, to_parent))
    child.start()
    try:
        kind, payload = to_parent.get(timeout=TIMEOUT_S)
        assert kind == "ready", payload
        assert payload == child.pid
        os.kill(child.pid, signal.SIGKILL)
        child.join(TIMEOUT_S)
    finally:
        if child.is_alive():
            child.kill()
            child.join(5)

    # The dead owner's sentinel still points at its pid.
    assert os.path.exists(sentinel_path(name))

    with (
        open_pool(extra, data_parallel_index=0),
        open_manager(extra, data_parallel_index=0) as owner,
    ):
        # Recreated, not reused: the killed owner's entries are gone.
        assert (
            owner.lookup_and_pin(
                "n1", block_hashes(keys), len(keys) * owner.tokens_per_chunk
            )
            is None
        )
        owner.finish_request("n1")
        with probe(name) as client:
            assert client.mempool_used() == 0
            assert hit_len(client, keys) == 0
        import json

        with open(sentinel_path(name)) as f:
            assert json.load(f)["owner_pid"] == os.getpid()
    assert shm_leftovers(name) == []
