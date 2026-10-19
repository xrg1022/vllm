# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RadixShmem request scheduling, sparse-state boundaries and transfer lifetimes."""

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import MEDIUM_CPU, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.coordinator import (
    plan_load,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemMetadata,
    RadixShmemWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    get_kv_cache_spec_kind,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LoadStoreSpec
from vllm.v1.kv_offload.radixshmem.backend import RadixShmemBackend
from vllm.v1.kv_offload.radixshmem.manager import (
    Chunk,
    LookupLease,
    StoreReservation,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    pending_count: int
    reservation: StoreReservation | None = None
    lease: LookupLease | None = None
    gpu_block_ids: list[int] = field(default_factory=list)
    failed: bool = False


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    is_full: bool
    kv_cache_spec: KVCacheSpec
    manager_cls: type[SingleTypeKVCacheManager]
    requires_cow_source: bool = False
    # EAGLE/MTP drafter layers in the FULL group: the trailing block of a hit
    # is dropped. Other lanes ignore the flag: an SWA slot holds the window
    # tail before a FULL boundary and cannot hold the EAGLE peek block past it.
    # Drafter KV only affects draft acceptance, never the sampled tokens.
    is_eagle_group: bool = False


class RadixSchedulerConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    num_workers: int
    alignment_tokens: int

    @classmethod
    def from_spec(
        cls,
        spec: RadixShmemBackend,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "RadixSchedulerConfig":
        groups = []
        for group in spec.config.groups:
            cache_group = kv_cache_config.kv_cache_groups[group.group_id]
            kv_spec = cache_group.kv_cache_spec
            manager_cls = KVCacheSpecRegistry.get_manager_class(kv_spec)
            assert manager_cls is not None
            tokens_per_chunk = group.tokens_per_block * spec.blocks_per_chunk
            groups.append(
                GroupOffloadConfig(
                    group_idx=group.group_id,
                    tokens_per_block=group.tokens_per_block,
                    tokens_per_chunk=tokens_per_chunk,
                    hashes_per_chunk=tokens_per_chunk // spec.tokens_per_hash,
                    is_full=isinstance(kv_spec, FullAttentionSpec),
                    kv_cache_spec=kv_spec,
                    manager_cls=manager_cls,
                    requires_cow_source=isinstance(kv_spec, MambaSpec),
                    is_eagle_group=(
                        cache_group.is_eagle_group
                        and isinstance(kv_spec, FullAttentionSpec)
                    ),
                )
            )
        # Geometry requires exactly one FULL group; all other lanes share its tree.
        alignment_tokens = next(
            group.tokens_per_chunk for group in groups if group.is_full
        )
        return cls(
            tuple(groups),
            spec.blocks_per_chunk,
            vllm_config.parallel_config.world_size,
            alignment_tokens,
        )


@dataclass
class RequestGroupState:
    block_ids: list[int] = field(default_factory=list)
    next_stored_chunk_idx: int = 0


@dataclass(slots=True)
class RadixRequestState:
    config: RadixSchedulerConfig
    req: Request
    lease: LookupLease | None = None
    load_masks: tuple[list[bool], ...] = ()
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    num_locally_computed_tokens: int = 0
    transfer_jobs: set[int] = field(default_factory=set)
    finished_signaled: bool = False
    # A load failed: vLLM recomputes, and the hit is not offered again.
    load_failed: bool = False

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )

    def storable_chunks(
        self,
        group_config: "GroupOffloadConfig",
        group_state: RequestGroupState,
        num_offloadable_tokens: int,
    ) -> int:
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        num_allocated_chunks = (
            len(group_state.block_ids) // self.config.blocks_per_chunk
        )
        num_keyed_chunks = len(self.req.block_hashes) // group_config.hashes_per_chunk
        return min(num_chunks, num_allocated_chunks, num_keyed_chunks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )


class RadixShmemConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: RadixShmemBackend,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.config = RadixSchedulerConfig.from_spec(spec, vllm_config, kv_cache_config)
        self._groups = {g.group_idx: g for g in self.config.kv_group_configs}
        self.manager = spec.open_manager()

        self._has_sparse_groups = any(
            not group.is_full for group in self.config.kv_group_configs
        )
        self._req_status: dict[ReqId, RadixRequestState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        self._gpu_block_pool: BlockPool | None = None
        self._pending_store_jobs: dict[int, TransferJob] = {}

        self._job_counter: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        self._events: list[KVCacheEvent] = []

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def bind_gpu_block_pool(self, pool: BlockPool) -> None:
        self._gpu_block_pool = pool

    def _new_job(
        self,
        state: RadixRequestState,
        src: LoadStoreSpec,
        dst: LoadStoreSpec,
        *,
        reservation: StoreReservation | None = None,
        lease: LookupLease | None = None,
    ) -> tuple[int, TransferJob]:
        job_id = self._generate_job_id()
        blocks: list[int] = []
        if reservation is not None:
            assert isinstance(src, GPULoadStoreSpec)
            assert self._gpu_block_pool is not None
            blocks = list(dict.fromkeys(int(bid) for bid in src.block_ids))
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[bid] for bid in blocks]
            )
        state.transfer_jobs.add(job_id)
        self._jobs[job_id] = TransferJobStatus(
            state.req.request_id,
            self.config.num_workers,
            reservation,
            lease,
            blocks,
        )
        return job_id, TransferJob(state.req.request_id, src, dst)

    def _collect_events(self) -> None:
        for req_id, chunks in self.manager.take_events():
            req = self._req_status[req_id].req
            self._events.extend(self._stored_event(req, chunk) for chunk in chunks)

    def _stored_event(self, req: Request, chunk: Chunk) -> BlockStored:
        """The native store event of one chunk: its block hashes, the hash
        before it and its tokens."""
        group = self._groups[chunk.group_id]
        bpc = self.config.blocks_per_chunk
        hashes_per_block = group.hashes_per_chunk // bpc
        first_block = chunk.index * bpc
        hashes = [
            req.block_hashes[(block + 1) * hashes_per_block - 1]
            for block in range(first_block, first_block + bpc)
        ]
        parent = (
            req.block_hashes[first_block * hashes_per_block - 1]
            if first_block
            else None
        )
        tok_start = chunk.index * group.tokens_per_chunk
        spec = group.kv_cache_spec
        lora = req.lora_request
        return BlockStored(
            block_hashes=[maybe_convert_block_hash(h) for h in hashes],
            parent_block_hash=(
                maybe_convert_block_hash(parent) if parent is not None else None
            ),
            token_ids=list(
                req.all_token_ids[tok_start : tok_start + group.tokens_per_chunk]
            ),
            block_size=group.tokens_per_block,
            lora_id=lora.adapter_id if lora is not None else None,
            medium=MEDIUM_CPU,
            lora_name=lora.name if lora is not None else None,
            group_idx=chunk.group_id,
            kv_cache_spec_kind=get_kv_cache_spec_kind(spec).value,
            kv_cache_spec_sliding_window=(
                spec.sliding_window if isinstance(spec, SlidingWindowSpec) else None
            ),
        )

    def _calc_num_offloadable_tokens(
        self, req_status: RadixRequestState, num_computed_tokens: int
    ) -> int:
        # Only prompt tokens are offloaded.
        return min(num_computed_tokens, req_status.req.num_prompt_tokens)

    def _lookup(
        self,
        state: RadixRequestState,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        local = state.num_locally_computed_tokens
        cap = state.req.num_tokens - self._has_sparse_groups
        if max_num_new_tokens is not None:
            cap = min(cap, local + max_num_new_tokens)
        if state.lease is not None:
            state.lease.release()
            state.lease = None
        if cap <= local or state.load_failed:
            return 0
        lease = self.manager.lookup_and_pin(
            state.req.request_id, state.req.block_hashes, cap
        )
        if lease is None:
            return None if self.manager.pending(state.req.request_id) else 0
        hit, masks = plan_load(
            state.req.block_hashes,
            self.config.kv_group_configs,
            self.manager,
            lease,
            lease.hit * self.manager.tokens_per_chunk,
        )
        if hit <= local:
            lease.release()
            return 0
        state.lease, state.load_masks = lease, masks
        return hit - local

    def on_new_request(self, request: Request) -> None:
        self._req_status[request.request_id] = RadixRequestState(self.config, request)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        max_num_new_tokens: int | None = None,
    ) -> tuple[int | None, bool]:
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.num_locally_computed_tokens = num_computed_tokens
        if request.skip_reading_prefix_cache:
            return 0, False
        num_hit_tokens = self._lookup(req_status, max_num_new_tokens)
        return num_hit_tokens, bool(num_hit_tokens)

    def get_external_cache_hit_sources(
        self,
        request: Request,
        num_external_tokens: int,
    ) -> dict[CacheHitSource, int]:
        start = self._req_status[request.request_id].num_locally_computed_tokens
        end = start + num_external_tokens
        size = self.manager.tokens_per_chunk
        sources: dict[CacheHitSource, int] = {}
        for i in range(start // size, cdiv(end, size)):
            source = self.manager.get_load_source(request.request_id, i)
            sources[source] = (
                sources.get(source, 0) + min(end, (i + 1) * size) - max(start, i * size)
            )
        return sources

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        state = self._req_status[request.request_id]
        lease, state.lease = state.lease, None
        if num_external_tokens == 0:
            if lease is not None:
                lease.release()
            return
        assert lease is not None
        end = state.num_locally_computed_tokens + num_external_tokens
        chunks: list[Chunk] = []
        ids: list[int] = []
        sizes: list[int] = []
        offsets: list[int] = []
        for group, group_state, mask in zip(
            self.config.kv_group_configs, state.group_states, state.load_masks
        ):
            table = blocks.blocks[group.group_idx]
            first = cdiv(state.num_locally_computed_tokens, group.tokens_per_block)
            count = cdiv(end, group.tokens_per_block)
            fresh = [
                i
                for i in range(first, count)
                if mask[i] and not table[i].is_null and table[i].block_hash is None
            ]
            indices = list(
                dict.fromkeys(i // self.config.blocks_per_chunk for i in fresh)
            )
            chunks.extend(Chunk(group.group_idx, i) for i in indices)
            ids.extend(table[i].block_id for i in fresh)
            sizes.append(len(fresh))
            offsets.append(fresh[0] if fresh else 0)
            group_state.next_stored_chunk_idx = cdiv(end, group.tokens_per_chunk)
        src = self.manager.load_slots(lease, chunks)
        job_id, job = self._new_job(
            state, src, GPULoadStoreSpec(ids, sizes, offsets), lease=lease
        )
        self._current_batch_load_jobs[job_id] = job

    def _update_req_states(self, output: SchedulerOutput) -> None:
        block_state = output.kv_connector_block_state
        for req_id, _, _ in yield_req_data(output):
            state = self._req_status[req_id]
            assert block_state is not None
            table = block_state.get_block_ids(req_id)
            assert table is not None
            for group, group_state in zip(
                self.config.kv_group_configs, state.group_states
            ):
                group_state.block_ids[:] = table[group.group_idx]

    def _build_boundary_store_jobs(
        self, output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        block_state = output.kv_connector_block_state
        handoffs = block_state.boundary_state_offloads if block_state else None
        jobs = {}
        for req_id, entries in (handoffs or {}).items():
            state = self._req_status.get(req_id)
            if state is None:
                continue
            cap = self._calc_num_offloadable_tokens(state, state.req.num_tokens)
            boundaries: dict[int, dict[int, int]] = {}
            for gid, block, boundary in entries:
                boundaries.setdefault(boundary, {})[gid] = block
            for boundary, blocks in boundaries.items():
                chunks, ids, sizes, offsets = [], [], [], []
                for group in self.config.kv_group_configs:
                    block = blocks.get(group.group_idx, 0)
                    store = bool(
                        block
                        and boundary <= cap
                        and group.requires_cow_source
                        and boundary % group.tokens_per_chunk == 0
                    )
                    if store:
                        chunks.append(
                            Chunk(
                                group.group_idx, boundary // group.tokens_per_chunk - 1
                            )
                        )
                        ids.append(block)
                    sizes.append(int(store))
                    offsets.append(
                        boundary // group.tokens_per_block - 1 if store else 0
                    )
                reservation = self.manager.reserve(
                    req_id, state.req.block_hashes, chunks
                )
                if reservation is None or not reservation.chunks:
                    continue
                assert reservation.chunks == chunks
                job_id, job = self._new_job(
                    state,
                    GPULoadStoreSpec(ids, sizes, offsets),
                    reservation.store_spec,
                    reservation=reservation,
                )
                jobs[job_id] = job
        return jobs

    def _reachable_store_block_mask(
        self, group_config: GroupOffloadConfig, start_chunk_idx: int, end_chunk_idx: int
    ) -> list[bool] | None:
        kv_cache_spec = group_config.kv_cache_spec
        if isinstance(kv_cache_spec, SlidingWindowSpec) and (
            kv_cache_spec.extra_retained_tokens
        ):
            kv_cache_spec = replace(
                kv_cache_spec,
                sliding_window=(
                    kv_cache_spec.sliding_window + kv_cache_spec.extra_retained_tokens
                ),
                extra_retained_tokens=0,
            )
        # Dense: the tail before every aligned boundary, a superset of what the
        # GPU cache keeps under prefix_cache_retention_interval. A tail past
        # the last boundary has no FULL chunk to hang off in the index.
        return group_config.manager_cls.reachable_block_mask(
            start_block=start_chunk_idx * self.config.blocks_per_chunk,
            end_block=end_chunk_idx * self.config.blocks_per_chunk,
            alignment_tokens=self.config.alignment_tokens,
            kv_cache_spec=kv_cache_spec,
            use_eagle=group_config.is_eagle_group,
        )

    def _store_candidates(
        self,
        state: RadixRequestState,
        group: GroupOffloadConfig,
        group_state: RequestGroupState,
        limit: int,
    ) -> dict[Chunk, tuple[int, list[int]]]:
        if group.requires_cow_source:
            return {}
        start = group_state.next_stored_chunk_idx
        end = state.storable_chunks(group, group_state, limit)
        if end <= start:
            return {}
        mask = self._reachable_store_block_mask(group, start, end)
        bpc = self.config.blocks_per_chunk
        candidates = {}
        for i in range(start, end):
            local = (i - start) * bpc
            if not group_state.block_ids[(i + 1) * bpc - 1]:
                continue
            if mask is not None and not any(mask[local : local + bpc]):
                continue
            blocks = [
                bid for bid in group_state.block_ids[i * bpc : (i + 1) * bpc] if bid
            ]
            candidates[Chunk(group.group_idx, i)] = (i, blocks)
        return candidates

    def _store_request(
        self, state: RadixRequestState, computed: int
    ) -> dict[int, TransferJob]:
        req_id = state.req.request_id
        limit = self._calc_num_offloadable_tokens(state, computed)
        candidates = [
            self._store_candidates(state, group, group_state, limit)
            for group, group_state in zip(
                self.config.kv_group_configs, state.group_states
            )
        ]
        keys = [key for group_keys in candidates for key in group_keys]
        if not keys:
            state.advance_stored_idx(limit)
            return {}
        store = self.manager.reserve(req_id, state.req.block_hashes, keys)
        if store is None:
            return {}
        selected = set(store.chunks)
        state.advance_stored_idx(limit)
        if not selected:
            return {}
        ids: list[int] = []
        sizes, offsets = [], []
        for group_state, group_keys in zip(state.group_states, candidates):
            blocks: list[int] = []
            offset = 0
            for key, (i, source) in group_keys.items():
                if key not in selected:
                    continue
                if not blocks:
                    offset = i * self.config.blocks_per_chunk
                    # Sparse chunks may start with null GPU blocks.
                    gpu_blocks = group_state.block_ids
                    while gpu_blocks[offset] == 0:
                        offset += 1
                blocks.extend(source)
            ids.extend(blocks)
            sizes.append(len(blocks))
            offsets.append(offset)
        job_id, job = self._new_job(
            state,
            GPULoadStoreSpec(ids, sizes, offsets),
            store.store_spec,
            reservation=store,
        )
        return {job_id: job}

    def _build_store_jobs(self, output: SchedulerOutput) -> dict[int, TransferJob]:
        jobs = {}
        for req_id, scheduled in output.num_scheduled_tokens.items():
            state = self._req_status[req_id]
            if not state.req.is_finished():
                jobs.update(
                    self._store_request(
                        state, state.req.num_computed_tokens + scheduled
                    )
                )
        return jobs

    def build_connector_meta(self, output: SchedulerOutput) -> KVConnectorMetadata:
        self._update_req_states(output)
        for pending_state in self._req_status.values():
            if pending_state.lease is not None:
                pending_state.lease.release()
                pending_state.lease = None
        self.manager.step()
        self._collect_events()
        for req_id in output.preempted_req_ids or ():
            state = self._req_status.get(req_id)
            if state is not None:
                self._current_batch_jobs_to_flush.update(state.transfer_jobs)
        stores, self._pending_store_jobs = self._pending_store_jobs, {}
        stores.update(self._build_boundary_store_jobs(output))
        stores.update(self._build_store_jobs(output))
        meta = RadixShmemMetadata(
            self._current_batch_load_jobs, stores, self._current_batch_jobs_to_flush
        )
        for req_id in output.finished_req_ids or ():
            state = self._req_status.get(req_id)
            if state is None:
                continue
            state.finished_signaled = True
            self.manager.finish_request(req_id)
            if not state.transfer_jobs:
                del self._req_status[req_id]
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        return meta

    def has_pending_push_work(self) -> bool:
        return bool(self._jobs) or self.manager.has_pending_work()

    def reset_cache(self) -> bool:
        """Caller must first quiesce every DP rank and cross-node transfer.

        Only this scheduler's outstanding work is checked here.
        """
        return not self._jobs and self.manager.reset()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        meta = connector_output.kv_connector_worker_meta
        if meta is None:
            return
        assert isinstance(meta, RadixShmemWorkerMetadata)
        for job_id in meta.failed_jobs:
            self._jobs[job_id].failed = True
        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            job_status = self._jobs[job_id]
            # A shared slot is readable only after every TP worker wrote its cell.
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            if job_status.reservation is not None:
                self.manager.complete_store(
                    job_status.reservation, not job_status.failed
                )
                self._collect_events()
            if job_status.lease is not None:
                job_status.lease.release()
            if job_status.gpu_block_ids:
                assert self._gpu_block_pool is not None
                self._gpu_block_pool.free_blocks(
                    self._gpu_block_pool.blocks[bid]
                    for bid in reversed(job_status.gpu_block_ids)
                )
            del self._jobs[job_id]
            state = self._req_status[job_status.req_id]
            if job_status.failed and job_status.lease is not None:
                state.load_failed = True
            state.transfer_jobs.remove(job_id)
            if state.finished_signaled and not state.transfer_jobs:
                del self._req_status[job_status.req_id]

    def request_finished(
        self,
        request: Request,
        block_ids: tuple[list[int], ...] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        state = self._req_status.get(request.request_id)
        if state is None:
            self.manager.finish_request(request.request_id)
            return False, None
        if block_ids is not None:
            for group, group_state in zip(
                self.config.kv_group_configs, state.group_states
            ):
                group_state.block_ids[:] = block_ids[group.group_idx]
        # Pin final store sources before the engine frees the request's blocks.
        computed = (
            request.num_computed_tokens
            if request.status is RequestStatus.FINISHED_ABORTED
            else max(request.num_prompt_tokens, request.num_tokens - 1)
        )
        self._pending_store_jobs.update(self._store_request(state, computed))
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        events, self._events = self._events, []
        return events

    def shutdown(self) -> None:
        # EngineCore stops the worker processes before shutting down the scheduler.
        for job in self._jobs.values():
            if job.lease is not None:
                job.lease.release()
            if job.gpu_block_ids:
                assert self._gpu_block_pool is not None
                self._gpu_block_pool.free_blocks(
                    self._gpu_block_pool.blocks[bid]
                    for bid in reversed(job.gpu_block_ids)
                )
        self._jobs.clear()
        self._pending_store_jobs.clear()
        for state in self._req_status.values():
            if state.lease is not None:
                state.lease.release()
        self._req_status.clear()
        self.manager.shutdown()
