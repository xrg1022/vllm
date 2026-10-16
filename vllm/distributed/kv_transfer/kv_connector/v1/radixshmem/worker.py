# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RadixShmem KV registration, deferred stores and transfer completion."""

from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _TransferMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemMetadata,
    RadixShmemWorkerMetadata,
    ReqId,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.radixshmem.backend import RadixShmemBackend

logger = init_logger(__name__)

# (bytes, time, size) metric names of each transfer direction.
_LOAD_METRICS = (
    _TransferMetricName.LOAD_BYTES,
    _TransferMetricName.LOAD_TIME,
    _TransferMetricName.LOAD_SIZE,
)
_STORE_METRICS = (
    _TransferMetricName.STORE_BYTES,
    _TransferMetricName.STORE_TIME,
    _TransferMetricName.STORE_SIZE,
)


class _KVCacheViews(OffloadingConnectorWorker):
    """Reuses ``OffloadingConnectorWorker``'s KV cache layout handling only.

    Its ``register_kv_caches`` derives the per-group byte views from every
    attention backend layout and passes them to ``_init_worker``; none of that
    is RadixShmem specific, so the hook is overridden instead of copying it.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        on_views: Callable[[CanonicalKVCaches], None],
    ):
        # Not calling super().__init__: register_kv_caches reads only these.
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self._on_views = on_views

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self._on_views(kv_caches)


class RadixShmemConnectorWorker:
    """Implementation of Worker side methods."""

    def __init__(
        self,
        spec: RadixShmemBackend,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None
        self._is_store_writer = (
            not self.spec.replicated_layout or self.spec.config.parallel.rank == 0
        )

        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = RadixShmemWorkerMetadata()
        self._stats = OffloadingConnectorStats()
        # Requests whose load finished (or failed) since the last report.
        self._finished_recving: set[ReqId] = set()
        self._failed_recving: set[ReqId] = set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _KVCacheViews(
            self.vllm_config, self.kv_cache_config, self._init_worker
        ).register_kv_caches(kv_caches)

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)

    def _complete(self, job_id: int, *, success: bool) -> None:
        """Report a job to the scheduler. vLLM recomputes a failed load."""
        self._connector_worker_meta.mark_completed(job_id, success)
        req_id = self._load_jobs.pop(job_id, None)
        if req_id is not None:
            self._finished_recving.add(req_id)
            if not success:
                self._failed_recving.add(req_id)

    def _submit(
        self, job_id: int, src: LoadStoreSpec, dst: LoadStoreSpec, *, load: bool
    ) -> None:
        assert self.worker is not None
        try:
            if load:
                assert isinstance(dst, GPULoadStoreSpec)
                ok = self.worker.submit_load(job_id, src, dst)
            else:
                assert isinstance(src, GPULoadStoreSpec)
                ok = self.worker.submit_store(job_id, src, dst)
        except Exception:
            logger.exception(
                "RadixShmem: %s job %d failed to start",
                "load" if load else "store",
                job_id,
            )
            ok = False
        if not ok:
            self._complete(job_id, success=False)

    def _submit_deferred_stores(self) -> None:
        jobs, self._unsubmitted_store_jobs = self._unsubmitted_store_jobs, []
        for job_id, src_spec, dst_spec in jobs:
            self._submit(job_id, src_spec, dst_spec, load=False)

    def handle_preemptions(self, kv_connector_metadata: RadixShmemMetadata):
        assert self.worker is not None

        # Submit deferred stores before fencing GPU block reuse.
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        self._submit_deferred_stores()
        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: RadixShmemMetadata):
        assert self.worker is not None
        self._submit_deferred_stores()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            self._submit(job_id, entry.src_spec, entry.dst_spec, load=True)

    def prepare_store_kv(self, metadata: RadixShmemMetadata):
        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                self._connector_worker_meta.mark_completed(job_id)
                continue
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            # Start stores on the next step, after sampling-related copies.
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Failed loads are also in ``finished_recving``; see
        ``get_transfer_results`` for the failures."""
        assert self.worker is not None
        for result in self.worker.get_finished():
            job_id = result.job_id
            is_load = job_id in self._load_jobs
            size, time = result.transfer_size, result.transfer_time
            if result.success and size is not None and time is not None:
                bytes_name, time_name, size_name = (
                    _LOAD_METRICS if is_load else _STORE_METRICS
                )
                self._stats.increase_counter(bytes_name, size)
                self._stats.increase_counter(time_name, time)
                self._stats.observe_histogram(size_name, size)
            if not result.success:
                logger.warning(
                    "RadixShmem: %s job %d failed",
                    "load" if is_load else "store",
                    job_id,
                )
            self._complete(job_id, success=result.success)
        finished, self._finished_recving = self._finished_recving, set()
        return set(), finished

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        _, finished_recving = self.get_finished(finished_req_ids)
        failed, self._failed_recving = self._failed_recving, set()
        return KVConnectorTransferResults(
            finished_recving=finished_recving, failed_recving=failed
        )

    def build_connector_worker_meta(self) -> RadixShmemWorkerMetadata | None:
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = RadixShmemWorkerMetadata()
        return meta

    def get_kv_connector_stats(self) -> OffloadingConnectorStats | None:
        if self._stats.is_empty():
            return None
        stats, self._stats = self._stats, OffloadingConnectorStats()
        return stats

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._finished_recving.clear()
        self._failed_recving.clear()
        self._connector_worker_meta = RadixShmemWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
