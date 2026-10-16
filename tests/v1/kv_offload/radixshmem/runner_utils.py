# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The real ``RadixShmemBackend`` manager under ``RequestRunner``:
``RadixShmemRunnerSpec`` wraps it in ``MagicMock(wraps=...)``, returns a mock
worker, and re-wraps load/store specs as ``MockLoadStoreSpec`` carrying the
keys. The pool the owner worker would create is opened here per index name.
"""

from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockLoadStoreSpec,
    MockOffloadingSpec,
    MockOffloadingWorker,
    RequestRunner,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingWorker,
    get_offload_group_idx,
    make_offload_key,
)

radixshmem = pytest.importorskip("radixshmem")

from vllm.v1.kv_offload.radixshmem.backend import (  # noqa: E402
    RadixShmemBackend,
)
from vllm.v1.kv_offload.radixshmem.bootstrap import RadixRegions  # noqa: E402
from vllm.v1.kv_offload.radixshmem.manager import RadixShmemManager  # noqa: E402


class KeyedSpec(MockLoadStoreSpec):
    """What the runner parses, plus the real spec the manager produced."""

    def __init__(self, keys, real):
        super().__init__(keys)
        self.real = real


class RadixShmemRunnerSpec(RadixShmemBackend, MockOffloadingSpec):
    instances: list["RadixShmemRunnerSpec"] = []
    # index_shm_name -> the owner worker's regions (the pool), opened on demand.
    pools: dict[str, RadixRegions] = {}

    def _ensure_pool(self) -> None:
        name = self.settings.index_shm_name
        if name not in RadixShmemRunnerSpec.pools:
            RadixShmemRunnerSpec.pools[name] = RadixRegions.open_worker(
                self.geometry, self.settings, self.fingerprint(), owner=True
            )

    def __init__(self, config):
        super().__init__(config)
        if not hasattr(self, "handler"):
            self.handler = MockOffloadingWorker()
        self.real_manager: RadixShmemManager | None = None
        self._wrapped_manager: MagicMock | None = None
        self.prepared_stores: list[list] = []
        self.prepared_loads: list[list] = []
        RadixShmemRunnerSpec.instances.append(self)

    def open_manager(self) -> RadixShmemManager:
        if self._wrapped_manager is None:
            self._ensure_pool()
            real = RadixShmemBackend.open_manager(self)
            self.real_manager = real
            mock = MagicMock(spec=type(real), wraps=real)
            for name in (
                "tokens_per_chunk",
                "tokens_per_hash",
                "group_tokens",
                "lanes",
            ):
                setattr(mock, name, getattr(real, name))

            lookup_hashes = {}

            def keys_for(hashes, chunks):
                return [
                    make_offload_key(
                        hashes[
                            (chunk.index + 1)
                            * real.group_tokens[chunk.group_id]
                            // real.tokens_per_hash
                            - 1
                        ],
                        chunk.group_id,
                    )
                    for chunk in chunks
                ]

            def reserve(req_id, hashes, chunks):
                chunks = list(chunks)
                self.prepared_stores.append(keys_for(hashes, chunks))
                out = real.reserve(req_id, hashes, chunks)
                if out is not None:
                    out.store_spec = KeyedSpec(
                        keys_for(hashes, out.chunks), out.store_spec
                    )
                return out

            def lookup_and_pin(req_id, hashes, max_tokens):
                lease = real.lookup_and_pin(req_id, hashes, max_tokens)
                if lease is not None:
                    lookup_hashes[id(lease)] = hashes
                return lease

            def load_slots(lease, chunks):
                chunks = list(chunks)
                keys = keys_for(lookup_hashes.pop(id(lease)), chunks)
                self.prepared_loads.append(keys)
                return KeyedSpec(keys, real.load_slots(lease, chunks))

            mock.reserve.side_effect = reserve
            mock.lookup_and_pin.side_effect = lookup_and_pin
            mock.load_slots.side_effect = load_slots
            self._wrapped_manager = mock
            self.manager = mock
        return self._wrapped_manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        self._ensure_pool()
        return self.handler

    @classmethod
    def teardown(cls, runners: list[RequestRunner]) -> None:
        """Shut the runners' managers down, then destroy the pools."""
        for runner in reversed(runners):
            runner.worker_connector.shutdown()
            runner.connector_scheduler.shutdown()
        cls.instances.clear()
        pools, cls.pools = cls.pools, {}
        for pool in pools.values():
            pool.close()


def runner_extra(extra: dict) -> dict:
    """``extra_config_overrides`` selecting the runner spec."""
    return {
        **extra,
        "spec_name": RadixShmemRunnerSpec.__name__,
        "spec_module_path": RadixShmemRunnerSpec.__module__,
    }


def scheduler_spec(runner: RequestRunner) -> RadixShmemRunnerSpec:
    for spec in RadixShmemRunnerSpec.instances:
        if spec._wrapped_manager is runner.manager:
            return spec
    raise AssertionError("scheduler-side spec not found")


def stored_keys(runner: RequestRunner, group_idx: int | None = None) -> list:
    """Chunk identities offered to reserve so far, in order, once each; optionally
    those of one KV cache group."""
    seen: dict = {}
    for keys in scheduler_spec(runner).prepared_stores:
        for key in keys:
            if group_idx is None or get_offload_group_idx(key) == group_idx:
                seen.setdefault(key, None)
    return list(seen)
