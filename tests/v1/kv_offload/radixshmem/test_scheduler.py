# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The independent RadixShmem connector driven by ``RequestRunner``
(``RequestRunner``; spec glue in ``runner_utils``): one full-attention group,
full attention plus sliding-window groups, and full attention plus align-mode
mamba groups.
"""

import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import RequestRunner
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.scheduler import (
    RadixShmemConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem_connector import (
    build_radix_offloading_config,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    maybe_convert_block_hash,
)
from vllm.v1.core.sched.output import KVConnectorBlockState
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.radixshmem.remote import RemotePulls
from vllm.v1.outputs import KVConnectorOutput

from .runner_utils import (
    RadixShmemRunnerSpec,
    runner_extra,
    scheduler_spec,
    stored_keys,
)
from .utils import (
    SLOT_ALIGN,
    cleanup_shm,
    external_server,
    hit_len,
    joint_hit,
    make_extra,
    probe,
    ref_total,
    swa_used,
    unique_name,
)

radixshmem = pytest.importorskip("radixshmem")

BLOCK_SIZE = 4
# Full-attention-only runners use 3-block chunks; hybrid runners need 1.
BLOCKS_PER_CHUNK = 3
TOKENS_PER_CHUNK = BLOCK_SIZE * BLOCKS_PER_CHUNK
# A window of 8 tokens is 2 chunks of 4; the scheduler keeps 2 blocks per tail.
SLIDING_WINDOW = 8
NUM_GPU_BLOCKS = 100
NUM_SLOTS = 64
# GPU blocks are 32 B; every pool slot is one 4 KiB stride.
CPU_BYTES = NUM_SLOTS * 2 * SLOT_ALIGN
FULL, SWA = 0, 1


def _groups(
    num_swa_groups: int = 1,
    full_block: int = BLOCK_SIZE,
    swa_window: int = SLIDING_WINDOW,
) -> list[KVCacheGroupSpec]:
    """A full-attention group and ``num_swa_groups`` sliding-window groups
    with one window, the way the hybrid manager splits an n:1 model."""
    return [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=full_block, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        *(
            KVCacheGroupSpec(
                [f"layer{i}"],
                SlidingWindowSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=swa_window,
                ),
            )
            for i in range(1, num_swa_groups + 1)
        ),
    ]


@pytest.fixture
def name():
    n = unique_name("vllm_sched")
    try:
        yield n
    finally:
        cleanup_shm(n)


@pytest.fixture
def runners(name):
    made: list[RequestRunner] = []

    def factory(
        *,
        endpoint: str | None = None,
        async_scheduling: bool = False,
        swa_groups: int = 0,
        full_block: int = BLOCK_SIZE,
        worker_count: int = 1,
        swa_window: int = SLIDING_WINDOW,
        eagle_group: int | None = None,
    ) -> RequestRunner:
        extra = runner_extra(
            make_extra(name, cpu_bytes_to_use=CPU_BYTES, endpoint=endpoint)
        )
        groups = None
        if swa_groups or eagle_group is not None:
            groups = _groups(swa_groups, full_block, swa_window)
            if eagle_group is not None:
                groups[eagle_group] = replace(groups[eagle_group], is_eagle_group=True)
        runner = RequestRunner(
            block_size=BLOCK_SIZE,
            num_gpu_blocks=NUM_GPU_BLOCKS,
            blocks_per_chunk=BLOCKS_PER_CHUNK if groups is None else 1,
            async_scheduling=async_scheduling,
            kv_cache_groups=groups,
            extra_config_overrides=extra,
            worker_count=worker_count,
        )
        made.append(runner)
        return runner

    try:
        yield factory
    finally:
        RadixShmemRunnerSpec.teardown(made)


def test_store_publishes_and_a_later_request_loads_from_the_index(
    name, runners, monkeypatch
):
    from vllm.distributed.kv_transfer.kv_connector.v1 import offloading_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        scheduler,
        worker,
    )

    def native_init_forbidden(*args, **kwargs):
        raise AssertionError("RadixShmem must construct its own connector roles")

    for cls in (
        offloading_connector.OffloadingConnector,
        scheduler.OffloadingConnectorScheduler,
        worker.OffloadingConnectorWorker,
    ):
        monkeypatch.setattr(cls, "__init__", native_init_forbidden)
    runner = runners()
    published: list = []
    monkeypatch.setattr(
        runner.scheduler.kv_event_publisher,
        "publish",
        lambda batch: published.extend(batch.events),
    )
    prompt = [0] * (TOKENS_PER_CHUNK * 2)
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[0, 0], expected_stored=(0, 1, 2, 3, 4, 5))
    keys = stored_keys(runner)
    assert len(keys) == 2
    with probe(name) as client:
        assert hit_len(client, keys) == 2
        assert client.mempool_used() == 2
    # One native-format store event per chunk, chained through the request.
    events = [e for e in published if getattr(e, "medium", None) == MEDIUM_CPU]
    hashes = runner.scheduler.requests[str(runner.req_id)].block_hashes
    assert len(events) == 2
    for i, event in enumerate(events):
        chunk_hashes = hashes[i * BLOCKS_PER_CHUNK : (i + 1) * BLOCKS_PER_CHUNK]
        assert event.block_hashes == [maybe_convert_block_hash(h) for h in chunk_hashes]
        assert event.parent_block_hash == (
            maybe_convert_block_hash(hashes[i * BLOCKS_PER_CHUNK - 1]) if i else None
        )
        assert event.token_ids == prompt[:TOKENS_PER_CHUNK]
        assert (event.block_size, event.group_idx) == (BLOCK_SIZE, 0)
        assert event.kv_cache_spec_kind == "full_attention"
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # Same prompt with GPU cache cleared: every block is loaded from the shared index.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(0, 1, 2, 3, 4, 5))
    spec = scheduler_spec(runner)
    assert spec.prepared_loads == [keys]
    with probe(name) as client:
        # Loaded chunks were not stored twice and no pin is left behind.
        assert client.mempool_used() == 2
        assert ref_total(client) == 0


def test_eagle_group_hit_drops_the_trailing_block(name, runners):
    # The drafter's last block is volatile: a hit of 3 stored blocks loads 2.
    runner = runners(eagle_group=0)
    prompt = [0] * (BLOCK_SIZE * 3)
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[0, EOS_TOKEN_ID], expected_stored=(0, 1, 2))
    keys = stored_keys(runner)
    with probe(name) as client:
        assert hit_len(client, keys) == 3

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(0, 1))
    assert scheduler_spec(runner).prepared_loads[-1] == keys[:2]
    with probe(name) as client:
        assert client.mempool_used() == 3
        assert ref_total(client) == 0


def test_eagle_flag_on_a_sliding_window_group_is_ignored(name, runners):
    # An SWA slot holds the window tail before a FULL boundary and cannot hold
    # the EAGLE peek block past it, so the group hits like a non-EAGLE one.
    runner = runners(swa_groups=1, eagle_group=SWA)
    prompt = [0] * (BLOCK_SIZE * 3 + 1)
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[0, EOS_TOKEN_ID], expected_stored=(0, 1, 2))

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=prompt)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=((FULL, 0), (FULL, 1), (FULL, 2), (SWA, 1), (SWA, 2)),
    )
    with probe(name) as client:
        assert ref_total(client) == 0


def test_failed_transfers_publish_nothing_and_release_the_request(
    name, runners, monkeypatch
):
    runner = runners()
    mock_worker = runner.worker_connector.connector_worker.worker
    real_submit = {"store": mock_worker.submit_store, "load": mock_worker.submit_load}
    failing = {"store": True, "load": False}

    def submit(kind):
        def _submit(*args, **kwargs):
            if failing[kind]:
                raise RuntimeError("transfer error")
            return real_submit[kind](*args, **kwargs)

        return _submit

    monkeypatch.setattr(mock_worker, "submit_store", submit("store"))
    monkeypatch.setattr(mock_worker, "submit_load", submit("load"))

    # A failed store publishes nothing and returns its slots.
    prompt = [0] * (TOKENS_PER_CHUNK * 2)
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[0, EOS_TOKEN_ID])
    keys = stored_keys(runner)
    assert len(keys) == 2
    with probe(name) as client:
        assert hit_len(client, keys) == 0
        assert client.mempool_used() == 0
        assert ref_total(client) == 0
    assert not runner.connector_scheduler._jobs

    # The next request stores the same prefix.
    failing["store"] = False
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[0, EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5))
    with probe(name) as client:
        assert hit_len(client, keys) == 2

    # A failed load is reported for recomputation; the index and pins are intact.
    runner.scheduler.reset_prefix_cache()
    failing["load"] = True
    runner.new_request(token_ids=prompt)
    req_id = str(runner.req_id)
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    assert scheduler_spec(runner).prepared_loads[-1] == keys
    results = runner.worker_connector.get_transfer_results(set())
    assert results.failed_recving == {req_id}
    with probe(name) as client:
        assert hit_len(client, keys) == 2
        assert client.mempool_used() == 2
        assert ref_total(client) == 0
    assert not runner.connector_scheduler._jobs


def test_shared_slot_waits_for_both_tp_workers(name, runners):
    runner = runners(worker_count=2)
    runner.new_request(token_ids=[0] * TOKENS_PER_CHUNK)
    runner.run(decoded_tokens=[0], complete_transfers=False)
    jobs = runner.connector_scheduler._jobs
    assert len(jobs) == 1
    job_id = next(iter(jobs))
    keys = stored_keys(runner)
    source_ids = jobs[job_id].gpu_block_ids
    pool = runner.scheduler.kv_cache_manager.block_pool
    refs = {bid: pool.blocks[bid].ref_cnt for bid in source_ids}

    with probe(name) as client:
        assert hit_len(client, keys) == 0
        for remaining in (1, 0):
            runner.connector_scheduler.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=RadixShmemWorkerMetadata(
                        completed_jobs={job_id: 1}
                    )
                )
            )
            assert hit_len(client, keys) == (0 if remaining else 1)
            assert all(
                pool.blocks[bid].ref_cnt == count - (remaining == 0)
                for bid, count in refs.items()
            )
        assert not jobs
        assert ref_total(client) == 0


def test_second_runner_on_the_same_index_gets_an_external_hit(name, runners, tmp_path):
    with external_server(name, tmp_path, cpu_bytes_to_use=CPU_BYTES) as (
        _,
        endpoint,
    ):
        a = runners(endpoint=endpoint)
        b = runners(endpoint=endpoint)
        prompt = [0] * (TOKENS_PER_CHUNK * 2)
        a.new_request(token_ids=prompt)
        a.run(decoded_tokens=[0, 0], expected_stored=(0, 1, 2, 3, 4, 5))
        a.run(decoded_tokens=[EOS_TOKEN_ID])
        keys = stored_keys(a)

        b.new_request(token_ids=prompt)
        b._run(decoded_tokens=[EOS_TOKEN_ID], complete_transfers=True)
        loaded = {
            gpu_block
            for transfer in b.completed_loads
            for gpu_block in transfer.gpu_blocks
        }
        assert loaded == set(b._to_gpu_blocks((0, 1, 2, 3, 4, 5)))
        assert scheduler_spec(b).prepared_loads == [keys]
        # B did not store what A had already published.
        assert b.completed_stores == []
        with probe(name, endpoint=endpoint) as client:
            assert client.mempool_used() == 2
            assert ref_total(client) == 0


def test_scheduler_waits_for_pull_then_loads_without_prefilling(
    name, runners, monkeypatch
):
    writer = runners()
    prompt = [0] * (TOKENS_PER_CHUNK * 2)
    writer.new_request(token_ids=prompt)
    writer.run(decoded_tokens=[0, 0], expected_stored=(0, 1, 2, 3, 4, 5))
    writer.run(decoded_tokens=[EOS_TOKEN_ID])
    keys = stored_keys(writer)

    reader = runners()
    manager = scheduler_spec(reader).real_manager
    assert manager is not None
    client = manager.index.client
    pull_async = client.pull_async
    release = Event()

    def delayed_pull(hashes, **kwargs):
        assert release.wait(5)
        return pull_async(hashes, **kwargs)

    # Simulate discovery being needed, then use the library's real local result.
    monkeypatch.setattr(manager.index, "lookup_and_pin", lambda hashes: None)
    monkeypatch.setattr(client, "pull_async", delayed_pull)
    manager._remote = RemotePulls(client, timeout_s=30, max_pending=2)
    reader.new_request(token_ids=prompt)
    deadline = time.monotonic() + 5
    saw_pending = False

    def advance_network():
        nonlocal saw_pending
        assert time.monotonic() < deadline, "scheduler did not resume after pull"
        if manager.has_pending_work():
            saw_pending = True
            assert not reader.scheduler.running
            release.set()

    try:
        reader._run([EOS_TOKEN_ID], True, post_step_fn=advance_network)
    finally:
        release.set()
    assert saw_pending
    loaded = {
        block for transfer in reader.completed_loads for block in transfer.gpu_blocks
    }
    assert loaded == set(reader._to_gpu_blocks((0, 1, 2, 3, 4, 5)))
    assert scheduler_spec(reader).prepared_loads == [keys]
    assert reader.completed_stores == []
    with probe(name) as probe_client:
        assert ref_total(probe_client) == 0


def test_preemption_flushes_stores_and_resumes_from_the_index(name, runners):
    runner = runners()
    free_block_queue = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    num_free_blocks = free_block_queue.num_free_blocks

    # 3 prompt chunks, stores stay in flight: blocks [0, 1, 2], [3, 4, 5], [6, 7, 8]
    runner.new_request(token_ids=[0] * TOKENS_PER_CHUNK * 3)
    runner.run(decoded_tokens=[0], complete_transfers=False)
    with probe(name) as client:
        assert client.mempool_used() == 3
        assert hit_len(client, stored_keys(runner)) == 0

    # KV cache runs out: preemption flushes the in-flight stores, which publishes them.
    free_block_queue.num_free_blocks = 0
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_flushed=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )
    keys = stored_keys(runner)
    assert len(keys) == 3
    with probe(name) as client:
        assert hit_len(client, keys) == 3

    # Space is back: the three chunks reload from the index; decode is not stored.
    free_block_queue.num_free_blocks = num_free_blocks
    runner.scheduler.reset_prefix_cache()
    runner.run(
        decoded_tokens=[0] * BLOCK_SIZE,
        expected_loaded=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    with probe(name) as client:
        assert hit_len(client, keys) == 3
        assert client.mempool_used() == 3
        assert ref_total(client) == 0
    assert not runner.connector_scheduler._jobs


def test_reset_cache_clears_the_pool_once_in_flight_stores_finish(name, runners):
    runner = runners()
    prompt = [0] * (TOKENS_PER_CHUNK * 2)

    # Two chunks stored, their transfers still running.
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[0], complete_transfers=False)
    keys = stored_keys(runner)
    assert len(keys) == 2
    with probe(name) as client:
        assert client.mempool_used() == 2

    # Reset while the request runs (vLLM cannot reset the GPU cache then): the
    # in-flight stores are flushed on the next step, then the pool is cleared.
    assert runner.scheduler.reset_prefix_cache(reset_connector=True) is False
    runner.run(
        decoded_tokens=[0],
        complete_transfers=False,
        expected_flushed=(0, 1, 2, 3, 4, 5),
        expected_stored=(0, 1, 2, 3, 4, 5),
    )
    assert not runner.connector_scheduler._jobs
    with probe(name) as client:
        assert client.generation() == 1
        assert hit_len(client, keys) == 0
        assert client.mempool_used() == 0
        assert ref_total(client) == 0

    # The same prompt misses now and is stored again.
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.new_request(token_ids=prompt)
    runner.run(decoded_tokens=[0, EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5))
    with probe(name) as client:
        assert hit_len(client, keys) == 2

    # Nothing in flight: the pool is cleared at once.
    assert runner.scheduler.reset_prefix_cache(reset_connector=True)
    with probe(name) as client:
        assert client.generation() == 2
        assert hit_len(client, keys) == 0
        assert client.mempool_used() == 0


@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("swa_window", [8, 9])
def test_dense_swa_stores_and_a_window_hit_from_the_shared_index(
    name, runners, async_scheduling: bool, swa_window: int
):
    # Mirrors test_two_groups_full_and_sliding_window with the real manager.
    runner = runners(
        swa_groups=1, async_scheduling=async_scheduling, swa_window=swa_window
    )
    runner.new_request(token_ids=[0] * (BLOCK_SIZE * 6 + 2))
    runner.run(decoded_tokens=[0])
    # Both groups store every prompt chunk.
    runner.run(decoded_tokens=[0], expected_stored=(0, 1, 2, 3, 4, 5))
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    full_keys = stored_keys(runner, FULL)
    swa_keys = stored_keys(runner, SWA)
    assert len(full_keys) == len(swa_keys) == 6
    with probe(name) as client:
        assert hit_len(client, full_keys) == 6
        assert joint_hit(client, full_keys)[:2] == (6, 4)
        assert joint_hit(client, full_keys[:3])[:2] == (3, 1)
        assert swa_used(client) == 6

    # Prefix cache cleared: 3 chunks hit; the SWA lane serves the last 2.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * (BLOCK_SIZE * 3 + 1))
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    )
    spec = scheduler_spec(runner)
    assert spec.prepared_loads[-1] == full_keys[:3] + swa_keys[1:3]

    # 3 chunks on the GPU, one more loaded: the window ends at the hit.
    runner.new_request(token_ids=[0] * (BLOCK_SIZE * 4 + 1))
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=((0, 3), (1, 3)))
    assert spec.prepared_loads[-1] == [full_keys[3], swa_keys[3]]
    with probe(name) as client:
        assert ref_total(client) == 0
        assert client.mempool_used() == 6
        assert swa_used(client) == 6


def test_two_swa_groups_share_one_swa_slot_per_chunk(name, runners):
    # One full group and two sliding-window groups with one window.
    runner = runners(swa_groups=2)
    spec = scheduler_spec(runner)
    assert spec.geometry.swa is not None
    assert spec.geometry.swa.group_ids == (1, 2)
    assert spec.geometry.swa.cell_bytes == 2 * spec.geometry.full.cell_bytes
    groups = (FULL, 1, 2)
    runner.new_request(token_ids=[0] * (BLOCK_SIZE * 6 + 2))
    runner.run(decoded_tokens=[0])
    runner.run(
        decoded_tokens=[0],
        expected_stored=tuple((g, i) for g in groups for i in range(6)),
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    keys = {g: stored_keys(runner, g) for g in groups}
    assert [len(k) for k in keys.values()] == [6, 6, 6]
    with probe(name) as client:
        assert joint_hit(client, keys[FULL])[:2] == (6, 4)
        # One SWA slot per chunk for both groups.
        assert swa_used(client) == 6

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * (BLOCK_SIZE * 3 + 1))
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(
            (FULL, 0),
            (FULL, 1),
            (FULL, 2),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        ),
    )
    assert spec.prepared_loads[-1] == keys[FULL][:3] + keys[1][1:3] + keys[2][1:3]
    with probe(name) as client:
        assert ref_total(client) == 0
        assert swa_used(client) == 6


@pytest.mark.parametrize("num_tokens", [48, 56], ids=["aligned", "unaligned"])
def test_swa_blocks_smaller_than_the_full_chunk(name, runners, num_tokens):
    # Smaller SWA blocks: 16-token full chunks and 4-token SWA blocks with an
    # 8-token window; each SWA slot holds the 2 sub-chunks before its end.
    # An unaligned prompt stores the same: its tail has no FULL chunk.
    runner = runners(swa_groups=1, full_block=16)
    spec = scheduler_spec(runner)
    assert spec.geometry.swa.window_blocks == 1
    assert spec.geometry.swa.members[0].sub_blocks == 2
    runner.new_request(token_ids=[0] * num_tokens)
    runner.run(decoded_tokens=[0])
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((FULL, 0), (FULL, 1), (FULL, 2))
        + tuple((SWA, i) for i in (2, 3, 6, 7, 10, 11)),
    )
    full_keys = stored_keys(runner, FULL)
    with probe(name) as client:
        assert joint_hit(client, full_keys)[:2] == (3, 2)
        assert swa_used(client) == 3

    # The prefix cache cleared: 3 chunks hit; the last chunk's 2 sub-chunks load.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * (num_tokens + 1))
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=((FULL, 0), (FULL, 1), (FULL, 2), (SWA, 10), (SWA, 11)),
    )
    with probe(name) as client:
        assert ref_total(client) == 0


BLOCK, MAMBA_BLOCK = 4, 16


@pytest.fixture(
    params=[(1, 1), (2, 1)],
    ids=["one_mamba_group", "two_mamba_groups"],
)
def sched(request):
    num_mamba, bpc = request.param
    name = unique_name("vllm_mamba")
    extra = runner_extra(make_extra(name, cpu_bytes_to_use=64 * 2 * SLOT_ALIGN))
    vllm_config = _make_vllm_config(extra_config=extra)
    vllm_config.cache_config.block_size = BLOCK
    vllm_config.cache_config.prefix_match_unit = BLOCK
    vllm_config.cache_config.kv_cache_layout = None
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        *(
            KVCacheGroupSpec(
                [f"mamba{j}"],
                MambaSpec(
                    block_size=MAMBA_BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            )
            for j in range(num_mamba)
        ),
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=64, kv_cache_tensors=[], kv_cache_groups=groups
    )
    spec = RadixShmemRunnerSpec(
        build_radix_offloading_config(vllm_config, kv_cache_config)
    )
    scheduler = RadixShmemConnectorScheduler(spec, vllm_config, kv_cache_config)
    scheduler.bind_gpu_block_pool(MagicMock())
    try:
        yield scheduler, name, num_mamba, bpc
    finally:
        scheduler.shutdown()
        RadixShmemRunnerSpec.teardown([])
        cleanup_shm(name)


def _request(scheduler, req_id: str, n: int):
    request = MagicMock(
        request_id=req_id,
        kv_transfer_params=None,
        num_prompt_tokens=n,
        num_tokens=n,
        num_computed_tokens=0,
        block_hashes=[BlockHash(f"h{i}".encode()) for i in range(n // BLOCK)],
        all_token_ids=list(range(n)),
        lora_request=None,
        status=None,
    )
    request.is_finished.return_value = False
    scheduler.on_new_request(request)
    status = scheduler._req_status[req_id]
    status.num_locally_computed_tokens = 0
    return request, status


def _complete_all(scheduler) -> None:
    job_ids = list(scheduler._jobs)
    if job_ids:
        scheduler.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=RadixShmemWorkerMetadata(
                    completed_jobs=dict.fromkeys(job_ids, 1)
                )
            )
        )


def test_mamba_states_store_and_serve_capped_hits(sched):
    scheduler, name, n, bpc = sched
    mamba = range(1, n + 1)
    # A mamba chunk is bpc mamba blocks, a whole number of full chunks.
    chunk = MAMBA_BLOCK * bpc
    prompt = 2 * chunk
    req, status = _request(scheduler, "A", prompt)
    assert scheduler._lookup(status) == 0
    status.group_states[0].block_ids[:] = list(range(1, prompt // BLOCK + 1))
    num_mamba_blocks = prompt // MAMBA_BLOCK
    for g in mamba:
        status.group_states[g].block_ids[:] = [
            100 * g + j for j in range(1, num_mamba_blocks + 2)
        ]
    out = SimpleNamespace(
        num_scheduled_tokens={"A": prompt},
        finished_req_ids=set(),
        kv_connector_block_state=KVConnectorBlockState(
            req_ids=set(),
            resolve_block_ids={}.__getitem__,
            # A state at every mamba block end; only chunk ends are stored.
            boundary_state_offloads={
                "A": [
                    (g, 100 * g + b, MAMBA_BLOCK * b)
                    for b in range(1, num_mamba_blocks + 1)
                    for g in mamba
                ]
            },
        ),
    )
    # Boundary jobs are built first, so they complete before the FULL job.
    scheduler._build_boundary_store_jobs(out)
    scheduler._build_store_jobs(out)
    _complete_all(scheduler)
    req.is_finished.return_value = True
    scheduler.request_finished(req)

    # One more token: the cap stays at the prompt, served by the deepest state.
    _, b = _request(scheduler, "B", prompt + BLOCK)
    assert scheduler._lookup(b) == prompt
    b.lease.release()
    b.lease = None
    scheduler.manager.finish_request(b.req.request_id)

    # The same prompt: the cap drops one chunk, below the pinned state.
    c_req, c = _request(scheduler, "C", prompt)
    assert scheduler._lookup(c) == chunk
    null = KVCacheBlock(0)
    null.is_null = True
    scheduler.update_state_after_alloc(
        c_req,
        KVCacheBlocks(
            ([KVCacheBlock(31 + i) for i in range(chunk // BLOCK)],)
            + tuple([null] * (bpc - 1) + [KVCacheBlock(40 + g)] for g in mamba)
        ),
        num_external_tokens=chunk,
    )
    # The FULL slots of [0, chunk) and the mamba slot at chunk, as the index
    # holds them.
    with probe(name) as client:
        r = client.query(
            scheduler.manager.prefix_hashes(req.block_hashes)[: chunk // (BLOCK * bpc)],
            mask=radixshmem.COMPONENT_MASK_FULL | radixshmem.COMPONENT_MASK_MAMBA,
            local_only=True,
            lock=False,
            update_meta=False,
        )
    [load] = scheduler._current_batch_load_jobs.values()
    expected = r.full_fragments[0][2].tolist() + [int(r.mamba[1])] * n
    assert load.src_spec.real.chunk_ids.tolist() == expected
    # Each mamba group loads its one state block, into the chunk's last block.
    gpu = load.dst_spec
    assert list(gpu.group_sizes[1:]) == [1] * n
    assert list(gpu.block_indices[1:]) == [bpc - 1] * n
    _complete_all(scheduler)
    with probe(name) as client:
        assert ref_total(client) == 0
        # One mamba slot per stored chunk end, whatever the number of groups.
        assert client.health_report().mamba.used == 2
        assert client.check_health() == 0
