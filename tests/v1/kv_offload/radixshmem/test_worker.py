# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU <-> SlotStore transfers through ``get_worker``, with the bytes checked
in the store: TP ranks write disjoint cells of one slot, a hybrid model's
groups land in their pool's cell part, and several sliding-window groups share
one SWA slot at their offsets. All ranks run in this process; rank 0 owns the
pool, the others attach.
"""

import contextlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from .utils import (
    cleanup_shm,
    make_extra,
    make_spec,
    probe,
    shm_leftovers,
    two_groups,
    unique_name,
)

radixshmem = pytest.importorskip("radixshmem")
pytestmark = pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="needs CUDA"
)

NUM_GPU_BLOCKS = 16
PAGE = 2048
WORLD_SIZE = 2
NUM_SLOTS = 16
GUARD = 0xEE
FULL, SWA = radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA


def _pattern(tensor_idx: int, block: int, rank: int) -> torch.Tensor:
    base = torch.arange(PAGE, dtype=torch.int64)
    return ((base * 7 + 13 * block + 29 * tensor_idx + 61 * rank) % 127).to(torch.int8)


def _fill(tensors, blocks, rank):
    for t_idx, t in enumerate(tensors):
        for b in blocks:
            t[b].copy_(_pattern(t_idx, b, rank).cuda())


def _gpu(*groups) -> GPULoadStoreSpec:
    """GPU block ids per group, in group order."""
    return GPULoadStoreSpec(
        [b for blocks in groups for b in blocks],
        group_sizes=tuple(len(blocks) for blocks in groups),
        block_indices=(0,) * len(groups),
    )


def _wait(worker, job_ids: set[int], timeout_s: float = 10.0) -> dict:
    """Successful results of ``job_ids`` by job id (get_finished drains all)."""
    results: dict = {}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and job_ids - results.keys():
        for result in worker.get_finished():
            assert result.success, result
            results[result.job_id] = result
        time.sleep(0.01)
    assert job_ids <= results.keys(), f"jobs {job_ids - results.keys()} did not finish"
    return results


def _as_u8(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.uint8).cpu().numpy()


class _StoreBytes:
    """Per-rank cells of the pools as views of the store, at the offsets it
    reports. A cell holds the chunk's pages tensor by tensor."""

    def __init__(self, store, cell: dict, blocks_per_chunk: int):
        self.raw = np.frombuffer(store.data_view(), dtype=np.uint8)
        self.pools = {kind: store.pool(kind) for kind in cell}
        self.cell_bytes = cell
        self.blocks_per_chunk = blocks_per_chunk

    def cell(self, kind, slot: int, rank: int) -> np.ndarray:
        pool, size = self.pools[kind], self.cell_bytes[kind]
        off = int(pool.base) + slot * int(pool.slot_bytes) + rank * size
        return self.raw[off : off + size]

    def page(self, kind, slot: int, rank: int, tensor_idx: int, sub: int):
        off = (tensor_idx * self.blocks_per_chunk + sub) * PAGE
        return self.cell(kind, slot, rank)[off : off + PAGE]


@contextlib.contextmanager
def _store_bytes(name: str, cell: dict, blocks_per_chunk: int = 1):
    with probe(name) as client:
        mem = _StoreBytes(client.store, cell, blocks_per_chunk)
        try:
            yield mem
        finally:
            # The mapping cannot be closed while a numpy view of it is alive.
            mem.raw = None


@contextlib.contextmanager
def _rig(cpu_bytes: int, refs: list[list[int]], world_size=WORLD_SIZE, **cfg):
    """``world_size`` workers over tensors of ``PAGE``-byte blocks, group ``g``
    referencing tensors ``refs[g]``, then the scheduler; the store is filled
    with ``GUARD``."""
    name = unique_name("vllm_gpu")
    extra = make_extra(name, cpu_bytes_to_use=cpu_bytes)
    cfg["world_size"] = world_size
    num_tensors = max(max(r) for r in refs) + 1
    torch.accelerator.set_device_index(0)
    workers, tensors, manager = [], [], None
    try:
        # Engine-core order: workers register their KV caches, then the scheduler.
        for rank in range(world_size):
            rank_tensors = [
                torch.zeros((NUM_GPU_BLOCKS, PAGE), dtype=torch.int8, device="cuda")
                for _ in range(num_tensors)
            ]
            caches = CanonicalKVCaches(
                tensors=[
                    CanonicalKVCacheTensor(tensor=t, page_size_bytes=PAGE)
                    for t in rank_tensors
                ],
                group_data_refs=[
                    [CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=PAGE) for i in r]
                    for r in refs
                ],
            )
            tensors.append(rank_tensors)
            workers.append(make_spec(extra, rank=rank, **cfg).get_worker(caches))
        manager = make_spec(extra, rank=0, **cfg).open_manager()
        with probe(name) as client:
            raw = np.frombuffer(client.store.data_view(), dtype=np.uint8)
            raw[:] = GUARD
            del raw
        yield SimpleNamespace(name=name, workers=workers, tensors=tensors)
    finally:
        if manager is not None:
            manager.shutdown()
        # The owner (rank 0) goes last and destroys the pool.
        for worker in reversed(workers):
            worker.shutdown()
        leftovers = shm_leftovers(name)
        cleanup_shm(name)
        assert leftovers == []


# One full-attention group over two tensors, two blocks per chunk.
DENSE_BPC = 2
DENSE_CELL = 2 * PAGE * DENSE_BPC


def test_two_ranks_write_disjoint_cells_of_one_slot(default_vllm_config):
    with _rig(
        NUM_SLOTS * WORLD_SIZE * DENSE_CELL,
        refs=[[0, 1]],
        worker_kv_bytes_per_block=2 * PAGE,
        blocks_per_chunk=DENSE_BPC,
    ) as rig:
        slot = 3
        blocks = [0, 1]
        for rank in range(WORLD_SIZE):
            _fill(rig.tensors[rank], blocks, rank=rank)
        golden = [[t.clone() for t in rank_tensors] for rank_tensors in rig.tensors]

        for rank, worker in enumerate(rig.workers):
            assert worker.submit_store(rank + 1, _gpu(blocks), CPULoadStoreSpec([slot]))
        for rank, worker in enumerate(rig.workers):
            _wait(worker, {rank + 1})

        with _store_bytes(rig.name, {FULL: DENSE_CELL}, DENSE_BPC) as mem:
            assert int(mem.pools[FULL].slot_bytes) == WORLD_SIZE * DENSE_CELL
            for rank in range(WORLD_SIZE):
                for t_idx in range(2):
                    for sub, block in enumerate(blocks):
                        assert np.array_equal(
                            mem.page(FULL, slot, rank, t_idx, sub),
                            _as_u8(golden[rank][t_idx][block]),
                        ), (rank, t_idx, sub)
            for other in range(NUM_SLOTS):
                if other != slot:
                    assert all(
                        np.all(mem.cell(FULL, other, r) == GUARD)
                        for r in range(WORLD_SIZE)
                    ), other

        dst = [4, 5]
        for rank, worker in enumerate(rig.workers):
            assert worker.submit_load(10 + rank, CPULoadStoreSpec([slot]), _gpu(dst))
        for rank, worker in enumerate(rig.workers):
            _wait(worker, {10 + rank})
        torch.accelerator.synchronize()
        for rank in range(WORLD_SIZE):
            for t_idx, t in enumerate(rig.tensors[rank]):
                assert torch.equal(t[dst].cpu(), golden[rank][t_idx][blocks].cpu())
                # Blocks that were not a destination stay zero.
                assert torch.all(t[[2, 3, 14, 15]] == 0)


# Group 0 (full): tensor 0. Group 1 (SWA): tensors 0 and 1. One cell per rank.
CELL = {FULL: PAGE, SWA: 2 * PAGE}


def test_round_trip_through_both_pools(default_vllm_config):
    groups = two_groups(
        full_bytes_per_block=PAGE, swa_bytes_per_block=2 * PAGE, sliding_window=64
    )
    with _rig(
        NUM_SLOTS * WORLD_SIZE * sum(CELL.values()),
        refs=[[0], [0, 1]],
        groups=groups,
        blocks_per_chunk=1,
    ) as rig:
        with _store_bytes(rig.name, CELL) as mem:
            for kind, pool in mem.pools.items():
                assert (pool.num_slots, pool.slot_bytes) == (
                    NUM_SLOTS,
                    WORLD_SIZE * CELL[kind],
                )
            assert int(mem.pools[FULL].base) == 0
            assert int(mem.pools[SWA].base) == NUM_SLOTS * WORLD_SIZE * CELL[FULL]

        # Both ranks store the same chunks into the same slots of both pools.
        full_blocks, swa_blocks, swa_only = [2], [6], [4]
        full_slot, swa_slot, swa_only_slot = 5, 9, 3
        for rank, tensors in enumerate(rig.tensors):
            _fill(tensors, full_blocks + swa_blocks + swa_only, rank)
        golden = [[t.clone() for t in tensors] for tensors in rig.tensors]
        for worker in rig.workers:
            assert worker.submit_store(
                1,
                _gpu(full_blocks, swa_blocks),
                CPULoadStoreSpec([full_slot, swa_slot]),
            )
        # Rank 0 also stores a chunk of the SWA group alone: zero full blocks.
        assert rig.workers[0].submit_store(
            2, _gpu([], swa_only), CPULoadStoreSpec([swa_only_slot])
        )
        jobs = [{1, 2}] + [{1}] * (WORLD_SIZE - 1)
        done = [_wait(worker, ids) for worker, ids in zip(rig.workers, jobs)]
        # Group 0 moves tensor 0 only; group 1 moves tensors 0 and 1.
        assert [d[1].transfer_size for d in done] == [(1 + 2) * PAGE] * WORLD_SIZE
        assert done[0][2].transfer_size == 2 * PAGE

        # (pool, slot, rank) -> (GPU blocks, tensors) written there; all else GUARD.
        written = {(FULL, full_slot, r): (full_blocks, [0]) for r in range(WORLD_SIZE)}
        written |= {(SWA, swa_slot, r): (swa_blocks, [0, 1]) for r in range(WORLD_SIZE)}
        written[SWA, swa_only_slot, 0] = (swa_only, [0, 1])
        with _store_bytes(rig.name, CELL) as mem:
            for kind in CELL:
                for slot in range(NUM_SLOTS):
                    for rank in range(WORLD_SIZE):
                        where = (kind, slot, rank)
                        if where not in written:
                            assert np.all(mem.cell(*where) == GUARD), where
                            continue
                        blocks, tensor_indices = written[where]
                        for t_idx in tensor_indices:
                            for sub, block in enumerate(blocks):
                                assert np.array_equal(
                                    mem.page(*where, t_idx, sub),
                                    _as_u8(golden[rank][t_idx][block]),
                                ), (*where, t_idx, sub)

        full_dst, swa_dst, swa_only_dst = [10], [12], [8]
        for worker in rig.workers:
            assert worker.submit_load(
                3, CPULoadStoreSpec([full_slot, swa_slot]), _gpu(full_dst, swa_dst)
            )
        assert rig.workers[0].submit_load(
            4, CPULoadStoreSpec([swa_only_slot]), _gpu([], swa_only_dst)
        )
        for rank, worker in enumerate(rig.workers):
            _wait(worker, {3, 4} if rank == 0 else {3})
        torch.accelerator.synchronize()
        for rank, (t0, t1) in enumerate(rig.tensors):
            g0, g1 = golden[rank]
            assert torch.equal(t0[full_dst], g0[full_blocks])
            assert torch.equal(t0[swa_dst], g0[swa_blocks])
            assert torch.equal(t1[swa_dst], g1[swa_blocks])
            # The full group does not reference tensor 1: its destination stays 0.
            assert torch.all(t1[full_dst] == 0)
            assert torch.all(t0[[0, 1, 14, 15]] == 0)
            assert torch.all(t1[[0, 1, 14, 15]] == 0)
        t0, t1 = rig.tensors[0]
        assert torch.equal(t0[swa_only_dst], golden[0][0][swa_only])
        assert torch.equal(t1[swa_only_dst], golden[0][1][swa_only])


# Every canonical tensor holds one layer of each group. Full group 0 and SWA
# group 1 reference tensors 0 and 1; SWA group 2 references tensor 0 only and
# has 4-token blocks, smaller than the full chunk, so its slot part holds 2
# sub-chunks. Groups 1 and 2 share each SWA slot: group 1's part at offset 0,
# group 2's after it; group 2's CPU ids are slot * 2 + sub-chunk.
SWA_PARTS = {1: [0, 1], 2: [0]}


def test_swa_groups_share_a_slot_at_their_offsets(default_vllm_config):
    from dataclasses import replace

    full, swa = two_groups(
        full_bytes_per_block=2 * PAGE, swa_bytes_per_block=2 * PAGE, sliding_window=16
    )
    swa2 = replace(
        swa,
        group_id=2,
        layer_names=("swa2",),
        kv_bytes_per_block=PAGE,
        tokens_per_block=4,
        sliding_window=8,
    )
    with _rig(
        NUM_SLOTS * 2 * 4096,
        refs=[[0, 1], [0, 1], [0]],
        world_size=1,
        groups=(full, swa, swa2),
        blocks_per_chunk=1,
    ) as rig:
        worker, tensors = rig.workers[0], rig.tensors[0]
        # Full block 2, SWA group 1 block 5, group 2 sub-chunks in blocks 9, 10.
        blocks = {0: [2], 1: [5], 2: [9, 10]}
        _fill(tensors, [b for bs in blocks.values() for b in bs], rank=0)
        golden = [t.clone() for t in tensors]
        full_slot, swa_slot = 3, 7
        ids = [full_slot, swa_slot, swa_slot * 2, swa_slot * 2 + 1]
        assert worker.submit_store(1, _gpu(*blocks.values()), CPULoadStoreSpec(ids))
        done = _wait(worker, {1})
        assert done[1].transfer_size == (2 + 2 + 2) * PAGE

        with _store_bytes(rig.name, {FULL: 2 * PAGE, SWA: 4 * PAGE}) as mem:
            for t_idx in (0, 1):
                np.testing.assert_array_equal(
                    mem.page(FULL, full_slot, 0, t_idx, 0), _as_u8(golden[t_idx][2])
                )
            # SWA slot: group 1's tensors 0 and 1, then group 2's 2 sub-chunks.
            expected = [(0, 5), (1, 5), (0, 9), (0, 10)]
            for i, (t_idx, block) in enumerate(expected):
                np.testing.assert_array_equal(
                    mem.page(SWA, swa_slot, 0, i, 0), _as_u8(golden[t_idx][block])
                )
            # Nothing past the four SWA pages of the slot was written.
            pool = mem.pools[SWA]
            start = int(pool.base) + swa_slot * int(pool.slot_bytes)
            end = start + int(pool.slot_bytes)
            assert (mem.raw[start + 4 * PAGE : end] == GUARD).all()

        # Load the three chunks back into fresh GPU blocks.
        for t in tensors:
            t.zero_()
        back = {0: [11], 1: [12], 2: [13, 14]}
        assert worker.submit_load(2, CPULoadStoreSpec(ids), _gpu(*back.values()))
        _wait(worker, {2})
        for gid, t_idxs in {0: [0, 1], **SWA_PARTS}.items():
            for t_idx in t_idxs:
                for dst, src in zip(back[gid], blocks[gid]):
                    torch.testing.assert_close(tensors[t_idx][dst], golden[t_idx][src])
        # Tensor 1 of group 2's blocks is not part of any transfer.
        assert (tensors[1][13:15] == 0).all()
