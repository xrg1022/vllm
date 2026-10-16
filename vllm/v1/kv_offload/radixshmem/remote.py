# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Submit distributed prefix pulls without blocking the scheduler."""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _Pull:
    req_id: str
    future: Future
    deadline: float


class RemotePulls:
    """Submit pulls off the scheduler thread; RadixShmem owns the transfers.

    Canceled pulls must finish and release pins before the client closes.
    """

    def __init__(self, client: Any, *, timeout_s: float, max_pending: int):
        self.client = client
        self.timeout_s = timeout_s
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="radixshmem-pull"
        )
        self._pulls: dict[str, _Pull] = {}
        self._retired: list[Future] = []

    def submit(self, req_id: str, hashes: np.ndarray, mask: int) -> bool:
        if len(self._pulls) + len(self._retired) >= self.max_pending:
            return False
        future = self._executor.submit(
            self.client.pull_async,
            hashes.copy(),
            mask=mask,
            lock=True,
            block=False,
            timeout_ms=max(1, int(self.timeout_s * 1000)),
        )
        self._pulls[req_id] = _Pull(req_id, future, time.monotonic() + self.timeout_s)
        return True

    def pending(self, req_id: str) -> bool:
        return req_id in self._pulls

    @property
    def has_pending_work(self) -> bool:
        return bool(self._pulls or self._retired)

    def cancel(self, req_id: str) -> None:
        pull = self._pulls.pop(req_id, None)
        if pull is not None and not pull.future.cancel():
            self._retired.append(pull.future)

    def clear(self) -> None:
        for req_id in list(self._pulls):
            self.cancel(req_id)

    def poll(self) -> list[tuple[str, Any, Any]]:
        completed = []
        for req_id, pull in list(self._pulls.items()):
            try:
                if pull.future.done():
                    job = pull.future.result()
                    if job.done():
                        result = job.wait(0)
                        del self._pulls[req_id]
                        completed.append((pull.req_id, job, result))
                        continue
            except Exception:
                del self._pulls[req_id]
                logger.exception("RadixShmem: remote pull failed for %s", req_id)
                continue
            if time.monotonic() >= pull.deadline:
                self.cancel(req_id)
        remaining = []
        for future in self._retired:
            try:
                if future.done() and future.result().done():
                    future.result().wait(0).finalize()
                else:
                    remaining.append(future)
            except Exception:
                logger.exception("RadixShmem: abandoned remote pull failed")
        self._retired = remaining
        return completed

    def close(self) -> None:
        self.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for future in self._retired:
            try:
                if not future.cancelled():
                    future.result().wait().finalize()
            except Exception:
                logger.exception("RadixShmem: remote pull failed during shutdown")
        self._retired.clear()
