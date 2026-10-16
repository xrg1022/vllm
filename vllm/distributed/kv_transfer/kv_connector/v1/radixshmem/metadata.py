# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RadixShmem transfer jobs and TP completion metadata."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.base import LoadStoreSpec

ReqId = str


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec."""

    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec


@dataclass
class RadixShmemMetadata(KVConnectorMetadata):
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None


@dataclass
class RadixShmemWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs."""

    completed_jobs: dict[int, int] = field(default_factory=dict)
    # Jobs that failed on at least one worker: their stores are not published.
    failed_jobs: set[int] = field(default_factory=set)

    def mark_completed(self, job_id: int, success: bool = True) -> None:
        self.completed_jobs[job_id] = 1
        if not success:
            self.failed_jobs.add(job_id)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, RadixShmemWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        return RadixShmemWorkerMetadata(
            completed_jobs=merged, failed_jobs=self.failed_jobs | other.failed_jobs
        )
