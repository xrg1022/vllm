# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This worker's cells of the attached radixshmem SlotStore as strided views.

Slot ``s`` of a pool is row ``s`` at ``base + s * slot_stride``; this worker's
cell starts ``rank * cell_bytes`` into the row. The native transfer handler
addresses these views like ``SharedOffloadRegion`` rows.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.gpu_worker import MAX_HOST_REGISTER_CHUNK_BYTES
from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister
from vllm.v1.kv_offload.radixshmem.bootstrap import _import_radixshmem, component_of
from vllm.v1.kv_offload.radixshmem.geometry import SlotGeometry

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PoolRange:
    # Offset of slot 0 in data_view.
    base: int
    num_slots: int
    slot_stride: int
    cell_bytes: int

    @property
    def num_bytes(self) -> int:
        return self.num_slots * self.slot_stride


class SlotStoreRegion:
    def __init__(
        self,
        store: Any,
        geometry: SlotGeometry,
        pool_slots: dict[str, int],
        rank: int,
    ) -> None:
        rs = _import_radixshmem()
        self._store = store
        # The memoryview keeps the store mapping alive for the tensors.
        self._data_view = store.data_view()
        base = torch.frombuffer(self._data_view, dtype=torch.int8)
        self.total_size_bytes = base.numel()
        self.rank = rank
        self._pools: dict[str, _PoolRange] = {}
        for pool in geometry.pools:
            sp = store.pool(component_of(rs, pool.kind))
            if (rank + 1) * pool.cell_bytes > pool.slot_stride:
                raise ValueError(
                    f"rank {rank} cell [{rank * pool.cell_bytes}, "
                    f"{(rank + 1) * pool.cell_bytes}) does not fit in a "
                    f"{pool.kind} slot of {pool.slot_stride} bytes"
                )
            rng = _PoolRange(
                base=int(sp.base),
                num_slots=pool_slots[pool.kind],
                slot_stride=pool.slot_stride,
                cell_bytes=pool.cell_bytes,
            )
            self._pools[pool.kind] = rng
        self._base: torch.Tensor | None = base
        self.is_pinned: bool = False
        self.pinned_addresses: list[int] = []

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("SlotStoreRegion has been released.")
        return self._base

    def view(self, kind: str, cell_offset: int, width: int) -> torch.Tensor:
        """``(num_slots, width)`` int8 rows, ``cell_offset`` into this cell."""
        pool = self._pools[kind]
        if cell_offset + width > pool.cell_bytes:
            raise ValueError(
                f"view ends at {cell_offset + width} past the {kind} cell "
                f"({pool.cell_bytes} bytes)"
            )
        view = torch.as_strided(
            self.base_tensor,
            size=(pool.num_slots, width),
            stride=(pool.slot_stride, 1),
            storage_offset=pool.base + self.rank * pool.cell_bytes + cell_offset,
        )
        return view

    def pin(self) -> bool:
        """Host-register every pool in slot-aligned chunks; on any failure
        unregister all and return False (transfers stay pageable)."""
        base_ptr = self.base_tensor.data_ptr()
        addresses: list[int] = []
        for kind, pool in self._pools.items():
            rows = max(MAX_HOST_REGISTER_CHUNK_BYTES // pool.slot_stride, 1)
            chunk = rows * pool.slot_stride
            for offset in range(0, pool.num_bytes, chunk):
                address = base_ptr + pool.base + offset
                size = min(chunk, pool.num_bytes - offset)
                if host_register(address, size):
                    addresses.append(address)
                    continue
                logger.warning(
                    "host_register failed for the %s pool at %.2f of %.2f GB; "
                    "the radixshmem pool stays pageable",
                    kind,
                    offset / 1e9,
                    pool.num_bytes / 1e9,
                )
                for registered in reversed(addresses):
                    host_unregister(registered)
                return False
        self.pinned_addresses.extend(addresses)
        self.is_pinned = True
        logger.debug(
            "Host-registered radixshmem pools rank=%d %.2f GB in %d chunk(s)",
            self.rank,
            sum(p.num_bytes for p in self._pools.values()) / 1e9,
            len(addresses),
        )
        return True

    def cleanup(self) -> None:
        """Unregister pinned ranges and drop the mapping references."""
        if self.is_pinned:
            for address in reversed(self.pinned_addresses):
                host_unregister(address)
            self.pinned_addresses.clear()
            self.is_pinned = False
        self._base = None
        self._data_view = None
        self._store = None
