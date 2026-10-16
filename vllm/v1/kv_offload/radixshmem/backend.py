# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct the RadixShmem pools, native manager and GPU transfer handlers."""

import hashlib
import json
from typing import Any

import numpy as np
import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
    compute_sub_block_ptrs,
)
from vllm.v1.kv_offload.radixshmem.bootstrap import (
    RadixRegions,
    RadixShmemSettings,
)
from vllm.v1.kv_offload.radixshmem.geometry import (
    SlotGeometry,
    compute_geometry,
    layout_fingerprint,
)
from vllm.v1.kv_offload.radixshmem.manager import RadixIndex, RadixShmemManager
from vllm.v1.kv_offload.radixshmem.region import SlotStoreRegion
from vllm.v1.kv_offload.radixshmem.remote import RemotePulls

logger = init_logger(__name__)


class PooledTransferHandler(SingleDirectionOffloadingHandler):
    """The native handler with one CPU view per (group, tensor): groups
    share GPU tensors but not CPU pools, so the CPU side depends on both."""

    def __init__(
        self,
        cpu_views: list[list[torch.Tensor | None]],
        sub_blocks: list[int],
        **kwargs: Any,
    ) -> None:
        self._cpu_views = cpu_views
        # Per group: chunks per slot part; CPU ids are slot * k + sub-chunk.
        self._sub_blocks = sub_blocks
        super().__init__(**kwargs)

    def _fill_direct_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        views = self._cpu_views[g_idx]
        num_bytes = 0
        for data_ref in self.layer_refs_per_group[g_idx]:
            t_idx = data_ref.tensor_idx
            cpu = views[t_idx]
            assert cpu is not None, (g_idx, t_idx)
            if self.gpu_to_cpu:
                src_tensor, dst_tensor = self.src_tensors[t_idx], cpu
            else:
                src_tensor, dst_tensor = cpu, self.dst_tensors[t_idx]
            end_idx = op_idx + group_size
            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                all_src[op_idx:end_idx],
                src_tensor,
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                all_dst[op_idx:end_idx],
                dst_tensor,
                skip_count=dst_skip_count,
            )
            k = self._sub_blocks[g_idx]
            if k > 1:
                # CPU ids are slot * k + sub-chunk.
                ids, out = (
                    (group_dst, all_dst) if self.gpu_to_cpu else (group_src, all_src)
                )
                slot, sub = np.divmod(ids.astype(np.uint64)[:group_size], k)
                out[op_idx:end_idx] = (
                    cpu.data_ptr() + slot * cpu.stride(0) + sub * (cpu.shape[1] // k)
                )
            all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
            num_bytes += group_size * data_ref.page_size_bytes
            op_idx = end_idx
        return op_idx, num_bytes


class RadixShmemOffloadingWorker(OffloadingWorker):
    """Transfers between the GPU KV caches and this worker's pool cells.

    Built in ``get_worker`` before CUDA graph capture, like the native
    backend. Shutdown drains both directions, unregisters the pools and
    closes the regions (the owner also stops the server).
    """

    def __init__(
        self,
        store_handler: SingleDirectionOffloadingHandler,
        load_handler: SingleDirectionOffloadingHandler,
        region: SlotStoreRegion,
        regions: RadixRegions,
    ):
        self._store_handler = store_handler
        self._load_handler = load_handler
        self._region: SlotStoreRegion | None = region
        self._regions = regions

    @override
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    @override
    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    @override
    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    @override
    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    @override
    def shutdown(self) -> None:
        failed = False
        for name, handler in (
            ("store", self._store_handler),
            ("load", self._load_handler),
        ):
            try:
                handler.shutdown()
            except Exception:
                logger.exception("Failed to shut down the %s transfer handler", name)
                failed = True
        try:
            if self._region is not None:
                if failed:
                    # A transfer may still run; do not unregister under it.
                    torch.accelerator.synchronize()
                self._region.cleanup()
                self._region = None
        finally:
            self._regions.close()


def plan_cpu_views(
    kv_caches: CanonicalKVCaches, geometry: SlotGeometry, region: SlotStoreRegion
) -> tuple[list[list[torch.Tensor | None]], list[int], list[torch.Tensor]]:
    """Validate each group's byte count while building its CPU tensor views."""
    if len(kv_caches.group_data_refs) != geometry.num_groups:
        raise ValueError(
            f"RadixShmem: the worker has {len(kv_caches.group_data_refs)} KV cache "
            f"groups, the geometry {geometry.num_groups}"
        )
    bpc = geometry.blocks_per_chunk
    views: list[list[torch.Tensor | None]] = []
    sub_blocks = []
    cpu_tensors: dict[int, torch.Tensor] = {}
    for g_index, refs in enumerate(kv_caches.group_data_refs):
        pool, member = geometry.cell_of_group(g_index)
        sub_blocks.append(member.sub_blocks)
        per_tensor: list[torch.Tensor | None] = [None] * len(kv_caches.tensors)
        offset = member.offset
        for ref in refs:
            page = kv_caches.tensors[ref.tensor_idx].page_size_bytes
            width = page * bpc * member.sub_blocks
            view = per_tensor[ref.tensor_idx] = region.view(pool.kind, offset, width)
            # The native handler shape-checks one sub-chunk of any group's view.
            cpu_tensors.setdefault(ref.tensor_idx, view[:, : page * bpc])
            offset += width
        used = offset - member.offset
        if used != member.cell_bytes:
            raise ValueError(
                f"RadixShmem: KV cache group {member.group_id} covers {used} bytes "
                f"per chunk, its {pool.kind} cell part {member.cell_bytes}. Packed "
                "single-tensor layouts cannot be split between pools."
            )
        views.append(per_tensor)
    if len(cpu_tensors) != len(kv_caches.tensors):
        raise ValueError("RadixShmem: a canonical tensor belongs to no KV cache group")
    return views, sub_blocks, [cpu_tensors[i] for i in range(len(kv_caches.tensors))]


def build_transfer_worker(
    kv_caches: CanonicalKVCaches,
    regions: RadixRegions,
    geometry: SlotGeometry,
    *,
    rank: int,
) -> RadixShmemOffloadingWorker:
    """The transfer worker over this rank's cells; host-registers every pool
    (seconds for a large pool)."""
    region = SlotStoreRegion(
        store=regions.store,
        geometry=geometry,
        pool_slots=regions.pool_slots,
        rank=rank,
    )
    try:
        gpu_tensors = [
            t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
            for t in kv_caches.tensors
        ]
        cpu_views, sub, cpu_tensors = plan_cpu_views(kv_caches, geometry, region)
        pinned = region.pin() if PIN_MEMORY else False
        handlers = [
            PooledTransferHandler(
                cpu_views,
                sub,
                gpu_tensors=gpu_tensors,
                cpu_tensors=cpu_tensors,
                blocks_per_chunk=geometry.blocks_per_chunk,
                layer_refs_per_group=kv_caches.group_data_refs,
                gpu_to_cpu=gpu_to_cpu,
                host_memory_is_pinned=pinned,
            )
            for gpu_to_cpu in (True, False)
        ]
    except BaseException:
        region.cleanup()
        raise
    return RadixShmemOffloadingWorker(handlers[0], handlers[1], region, regions)


class RadixShmemBackend:
    """CPU offloading into one shared-memory pool per node.

    ``cpu_bytes_to_use`` is node-wide. Supports one full-attention group plus
    sliding-window groups sharing one index window (sub-block packed when
    their blocks are smaller than the chunk) and align-mode mamba groups;
    PP 1; no canonical layout. The DP 0 / TP 0 worker creates the pool in
    ``get_worker``; schedulers and other workers attach. With ``remote``,
    prefixes found on other nodes are pulled into the local pool first.
    """

    def __init__(self, config: OffloadingConfig):
        self.config = config
        self.extra_config = config.extra_config
        self.tokens_per_hash = config.cache.tokens_per_hash
        self.blocks_per_chunk = config.cache.blocks_per_chunk
        self.settings = RadixShmemSettings.from_extra_config(self.extra_config)
        self.geometry = compute_geometry(config, self.extra_config)
        self.replicated_layout = config.replicated_layout
        self._manager: RadixShmemManager | None = None
        self._worker: RadixShmemOffloadingWorker | None = None

    @property
    def is_dp0(self) -> bool:
        """The first DP rank on the node."""
        parallel = self.config.parallel
        local = parallel.data_parallel_rank_local
        if local is None:
            local = parallel.data_parallel_index
        return local == 0

    @property
    def is_owner_worker(self) -> bool:
        """DP 0 / TP 0 creates the pool (with an external ``endpoint`` it only
        writes or adopts the sentinel); with ``remote`` every replica's TP 0
        does, serialized by a file lock."""
        return (self.settings.remote or self.is_dp0) and self.config.parallel.rank == 0

    def fingerprint(self) -> dict[str, Any]:
        return layout_fingerprint(self.geometry, self.config)

    def open_manager(self) -> RadixShmemManager:
        if self._manager is None:
            regions = RadixRegions.open_scheduler(
                self.geometry,
                self.settings,
                self.fingerprint(),
            )
            try:
                hash_salt = 0
                remote = None
                if self.settings.remote:
                    # Keys from another node match only for the same model,
                    # layout and namespace.
                    domain = {
                        **layout_fingerprint(self.geometry, self.config),
                        "cache_namespace": self.settings.cache_namespace,
                        "parallel_layout": [
                            self.config.parallel.tp_size,
                            self.config.parallel.pcp_size,
                            self.config.parallel.dcp_size,
                        ],
                        "hash_format": "vllm-radixshmem-u64-v1",
                    }
                    digest = hashlib.sha256(
                        json.dumps(domain, sort_keys=True).encode()
                    ).digest()
                    hash_salt = int.from_bytes(digest[:8], "little")
                    remote = RemotePulls(
                        regions.client,
                        timeout_s=self.settings.remote_timeout_s,
                        max_pending=self.settings.remote_max_pending,
                    )
                index = RadixIndex(
                    regions.client, kinds=self.geometry.kinds, hash_salt=hash_salt
                )
            except BaseException:
                regions.close()
                raise
            self._manager = RadixShmemManager(
                index,
                regions,
                tokens_per_chunk=self.geometry.tokens_per_chunk,
                tokens_per_hash=self.tokens_per_hash,
                group_tokens={
                    g.group_id: g.tokens_per_block * self.blocks_per_chunk
                    for g in self.config.groups
                },
                lanes={
                    m.group_id: p.kind for p in self.geometry.pools for m in p.members
                },
                sub_chunks={
                    m.group_id: (m.sub_tokens, m.sub_blocks)
                    for p in self.geometry.pools
                    for m in p.members
                    if m.sub_tokens
                },
                enable_events=self.config.enable_kv_cache_events,
                remote=remote,
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "RadixShmem offloading is only supported on CUDA-alike and XPU GPUs"
                )
            regions = RadixRegions.open_worker(
                self.geometry,
                self.settings,
                self.fingerprint(),
                owner=self.is_owner_worker,
            )
            try:
                self._worker = build_transfer_worker(
                    kv_caches,
                    regions,
                    self.geometry,
                    rank=0 if self.replicated_layout else self.config.parallel.rank,
                )
            except BaseException:
                regions.close()
                raise
        return self._worker
