# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture(autouse=True)
def _set_spawn_method(monkeypatch):
    # Spawn context for multi-process tests; the env var also silences vLLM's override.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture(autouse=True)
def _radix_connector(monkeypatch):
    from tests.v1.kv_connector.unit.offloading_connector import utils as runner
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1 import radixshmem_connector
    from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
        RadixShmemMetadata,
        RadixShmemWorkerMetadata,
    )

    from .runner_utils import RadixShmemRunnerSpec

    # Reuse the test driver with the RadixShmem protocol and mock transfers.
    monkeypatch.setattr(radixshmem_connector, "RadixShmemBackend", RadixShmemRunnerSpec)
    monkeypatch.setattr(
        runner, "OffloadingConnector", radixshmem_connector.RadixShmemConnector
    )
    monkeypatch.setattr(runner, "OffloadingConnectorMetadata", RadixShmemMetadata)
    monkeypatch.setattr(runner, "OffloadingWorkerMetadata", RadixShmemWorkerMetadata)
    monkeypatch.setitem(
        KVConnectorFactory._registry,
        "OffloadingConnector",
        lambda: radixshmem_connector.RadixShmemConnector,
    )
