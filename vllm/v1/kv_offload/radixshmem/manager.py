# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RadixShmem prefix leases and slot reservations for one DP scheduler."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.base import LoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.radixshmem.bootstrap import (
    RadixRegions,
    component_of,
    mask_of,
)
from vllm.v1.kv_offload.radixshmem.remote import RemotePulls
from vllm.v1.metrics.cache_hit_source import CacheHitSource

logger = init_logger(__name__)


def to_u64(keys: Iterable[BlockHash]) -> np.ndarray:
    """First 8 bytes of each key's block hash, little-endian. vLLM block
    hashes chain the parent, so truncation keeps prefix semantics."""
    parts = [k[:8].ljust(8, b"\0") for k in keys]
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return np.frombuffer(b"".join(parts), dtype=np.uint64).copy()


@dataclass(frozen=True)
class PublishResult:
    # Blocks registered in the tree.
    published: int
    # Publish refused: an error, or the FULL path below the segment is gone.
    rejected: bool
    # The slots the tree did not take (deduplicated or refused).
    unused: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


class LookupLease:
    """Pinned FULL ``slots[0:hit]``, SWA ``swa_slots`` of chunks
    ``[swa_start, hit)`` and the mamba state at ``hit`` (``mamba_slot``),
    valid until ``release()`` or a reset of the pool."""

    __slots__ = (
        "hit",
        "slots",
        "swa_start",
        "swa_slots",
        "mamba_slot",
        "generation",
        "_finalize",
        "_owner",
        "_released",
    )

    def __init__(
        self,
        hit: int,
        slots: np.ndarray,
        swa_start: int,
        swa_slots: np.ndarray,
        finalize,
        owner: "RadixIndex",
        generation: int,
        mamba_slot: int = -1,
    ):
        self.hit = hit
        self.slots = slots
        self.swa_start = swa_start
        self.swa_slots = swa_slots
        self.mamba_slot = mamba_slot
        self.generation = generation
        self._finalize = finalize
        self._owner = owner
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Drop the pin (idempotent); a no-op once the pool was reset."""
        if self._released:
            return
        self._released = True
        try:
            if self._owner.generation() == self.generation:
                self._finalize()
        finally:
            self._owner._forget(self)

    def covers(self, lane: str, a: int, b: int) -> bool:
        """Chunks ``[a, b)`` of ``lane`` are pinned."""
        if lane == "full":
            return a >= 0 and b <= self.hit
        if lane == "swa":
            return self.swa_start <= a and b <= self.hit
        if lane == "mamba":
            return self.mamba_slot >= 0 and (a, b) == (self.hit - 1, self.hit)
        return False

    def __repr__(self) -> str:
        state = "released" if self._released else "held"
        return (
            f"LookupLease({self.hit} blocks, swa [{self.swa_start}, {self.hit}), "
            f"{state})"
        )


class RadixIndex:
    """One scheduler's index operations over an attached ``RadixClient``."""

    def __init__(
        self, client: Any, kinds: Iterable[str] = ("full",), *, hash_salt: int = 0
    ):
        import radixshmem

        self.client = client
        self._hash_salt = np.uint64(hash_salt)
        self.kinds = tuple(kinds)
        if not self.kinds or self.kinds[0] != "full":
            raise ValueError(f"the FULL lane must come first, got {self.kinds}")
        self._full_mask = mask_of(radixshmem, "full")
        self.mask = 0
        for kind in self.kinds:
            self.mask |= mask_of(radixshmem, kind)
        self._component = {kind: component_of(radixshmem, kind) for kind in self.kinds}
        self._leases: set[LookupLease] = set()

    @property
    def has_swa(self) -> bool:
        return "swa" in self.kinds

    def encode_hashes(self, hashes: np.ndarray) -> np.ndarray:
        return hashes ^ self._hash_salt if self._hash_salt else hashes

    def generation(self) -> int:
        """The pool's reset count; see ``RadixClient.generation``."""
        return int(self.client.generation())

    def reset(self) -> None:
        """Clear the whole pool in place; earlier leases and reservations go stale."""
        self.client.reset()

    @property
    def has_leases(self) -> bool:
        return bool(self._leases)

    def lookup(self, hashes: np.ndarray) -> int:
        """FULL blocks of the root-anchored ``hashes`` in the pool."""
        if hashes.size == 0:
            return 0
        r = self.client.query(
            self.encode_hashes(hashes),
            mask=self._full_mask,
            local_only=True,
            lock=False,
            update_meta=False,
        )
        return int(r.common_hit)

    def lookup_and_pin(self, hashes: np.ndarray) -> LookupLease | None:
        """Match every lane and pin the joint hit; None (nothing pinned) on a
        zero-block hit."""
        if hashes.size == 0:
            return None
        r = self.client.query(
            self.encode_hashes(hashes),
            mask=self.mask,
            local_only=True,
            lock=True,
            update_meta=True,
        )
        hit = int(r.common_hit)
        if int(r.status) != 0 or hit == 0:
            if int(r.status) != 0:
                logger.warning_once("RadixShmem query failed: %s", r.status)
            r.finalize()
            return None
        # local_only: exactly one fragment, already cropped to common_hit.
        return self._lease(
            hit,
            r.full_fragments[0][2],
            int(r.swa_start),
            r.swa_slots,
            int(r.mamba[1]) if r.mamba is not None else -1,
            r.finalize,
        )

    def lease_from_pull(self, result: Any) -> LookupLease | None:
        """The lease of a completed ``pull_async`` (slots are local)."""
        hit = int(result.common_hit)
        if hit == 0:
            result.finalize()
            return None
        return self._lease(
            hit,
            result.full_slots,
            int(result.swa_start),
            result.swa_slots,
            int(result.mamba_slot),
            result.finalize,
        )

    def _lease(
        self,
        hit: int,
        full_slots: Any,
        swa_start: int,
        swa_slots: Any,
        mamba_slot: int,
        finalize,
    ) -> LookupLease:
        slots = np.ascontiguousarray(full_slots, dtype=np.int32)
        assert slots.size == hit, (slots.size, hit)
        if self.has_swa:
            swa_slots = np.ascontiguousarray(swa_slots, dtype=np.int32)
            assert swa_start + swa_slots.size == hit, (swa_start, swa_slots.size, hit)
        else:
            swa_slots = np.empty(0, dtype=np.int32)
            swa_start = hit
        if "mamba" in self.kinds:
            # With MAMBA in the mask common_hit is a checkpointed node END.
            assert mamba_slot >= 0, hit
        else:
            mamba_slot = -1
        gen = self.generation()
        lease = LookupLease(
            hit, slots, swa_start, swa_slots, finalize, self, gen, mamba_slot
        )
        self._leases.add(lease)
        return lease

    def _forget(self, lease: LookupLease) -> None:
        self._leases.discard(lease)

    def pool_total(self, kind: str) -> int:
        if kind == "full":
            return int(self.client.mempool_total())
        return int(getattr(self.client, f"{kind}_mempool_total")())

    def allocate(self, n: int, kind: str = "full") -> np.ndarray | None:
        """``n`` slots of pool ``kind`` (evicting unpinned LRU entries), or
        None when the pool cannot supply all of them."""
        if n <= 0 or n > self.pool_total(kind):
            return None
        slots = self.client.allocate_slots(n, self._component[kind])
        if int(slots.size) < n:
            if slots.size:
                self.client.recycle_slots(slots, self._component[kind])
            return None
        return np.ascontiguousarray(slots, dtype=np.int32)

    def recycle(self, slots: np.ndarray | None, kind: str = "full") -> None:
        if slots is None or slots.size == 0:
            return
        self.client.recycle_slots(
            np.ascontiguousarray(slots, dtype=np.int32), self._component[kind]
        )

    def publish(
        self, hashes: np.ndarray, slots: np.ndarray, start: int, kind: str = "full"
    ) -> PublishResult:
        """Publish FULL at ``start``, or aux slots at the prefix tail.

        The index deduplicates peers' stores and recycles unused slots.
        """
        n = int(slots.size)
        if n == 0:
            return PublishResult(0, False)
        res = self.client.insert(
            self.encode_hashes(np.ascontiguousarray(hashes, dtype=np.uint64)),
            np.ascontiguousarray(slots, dtype=np.int32),
            start=start if kind == "full" else 0,
            auto_recycle=True,
            component=self._component[kind],
        )
        unused_slots = np.asarray(res.unused_slots, dtype=np.int32)
        unused = int(unused_slots.size)
        rejected = int(res.error) != 0
        if rejected and (kind == "full" or res.error.name != "FULL_PATH_MISSING"):
            logger.warning_once(
                "RadixShmem %s insert failed (%s); raise max_nodes / data_pool_ratio",
                kind.upper(),
                res.error,
            )
        # An all-unused FULL insert either met a peer's prefix or lost its parent.
        if not rejected and kind == "full" and unused == n:
            rejected = self.lookup(hashes) < len(hashes)
        return PublishResult(0 if rejected else n - unused, rejected, unused_slots)

    def release_all(self) -> None:
        """Release every outstanding pin."""
        if self._leases:
            logger.warning(
                "RadixShmem: releasing %d lookup lease(s) still open",
                len(self._leases),
            )
        for lease in list(self._leases):
            try:
                lease.release()
            except Exception:
                logger.exception("RadixShmem: failed to release a lookup lease")
        self._leases.clear()

    def close(self) -> None:
        self.release_all()
        self.client = None


@dataclass(frozen=True, order=True)
class Chunk:
    """A group's logical chunk index, independent of the prefix hash encoding."""

    group_id: int
    index: int


@dataclass
class _PendingStore:
    lane: str
    hashes: np.ndarray
    slots: np.ndarray
    start: int
    end: int
    chunks: list[Chunk]
    # Pool reset count at reservation; a reset frees the slots.
    generation: int
    # None while the transfer runs; False when it failed.
    done: bool | None = None


@dataclass
class StoreReservation:
    req_id: str
    chunks: list[Chunk]
    store_spec: LoadStoreSpec
    entries: list[_PendingStore] = field(default_factory=list)


class Publication(NamedTuple):
    req_id: str
    chunks: list[Chunk]


@dataclass
class _ReqStores:
    full: list[_PendingStore] = field(default_factory=list)
    aux: list[_PendingStore] = field(default_factory=list)
    full_known: int = 0


class RadixShmemManager:
    """Owns slot reservations; the scheduler owns returned lookup leases.

    A reservation is completed once, after all TP workers finish. Aux pools
    publish only after the corresponding FULL path is ready.
    """

    def __init__(
        self,
        index: RadixIndex,
        regions: RadixRegions | None = None,
        *,
        tokens_per_chunk: int,
        tokens_per_hash: int,
        group_tokens: Mapping[int, int],
        lanes: Mapping[int, str],
        sub_chunks: Mapping[int, tuple[int, int]] | None = None,
        enable_events: bool = False,
        remote: RemotePulls | None = None,
    ):
        self.index = index
        self._regions = regions
        self.tokens_per_chunk = tokens_per_chunk
        self.tokens_per_hash = tokens_per_hash
        self.group_tokens = dict(group_tokens)
        self.lanes = dict(lanes)
        self._sub = dict(sub_chunks or {})
        self._lane_groups = {
            lane: [g for g, kind in lanes.items() if kind == lane]
            for lane in index.kinds
        }
        self._full_group = self._lane_groups["full"][0]
        self._remote = remote
        self._remote_ready: dict[str, LookupLease] = {}
        self._remote_checked: set[str] = set()
        self._remote_sources: dict[str, tuple[int, int]] = {}
        self._inflight_stores: dict[str, _ReqStores] = {}
        self._enable_events = enable_events
        self._events: list[Publication] = []
        self._reset_pending = False

    def prefix_hashes(self, hashes: Sequence[BlockHash]) -> np.ndarray:
        stride = self.tokens_per_chunk // self.tokens_per_hash
        return to_u64(hashes[stride - 1 :: stride])

    def _position(self, chunk: Chunk) -> tuple[int, int]:
        end = (chunk.index + 1) * self.group_tokens[chunk.group_id]
        index = (end - 1) // self.tokens_per_chunk
        if chunk.group_id not in self._sub:
            assert end % self.tokens_per_chunk == 0
            return index, 0
        size, count = self._sub[chunk.group_id]
        back = ((index + 1) * self.tokens_per_chunk - end) // size
        return index, count - 1 - back

    def lookup_and_pin(
        self,
        req_id: str,
        hashes: Sequence[BlockHash],
        max_tokens: int,
    ) -> LookupLease | None:
        if self._reset_pending:
            return None
        prefix = self.prefix_hashes(hashes)[: max_tokens // self.tokens_per_chunk]
        lease = self._remote_ready.pop(req_id, None)
        if lease is not None and lease.hit > len(prefix):
            lease.release()
            lease = None
        if lease is None:
            lease = self.index.lookup_and_pin(prefix)
        hit = lease.hit if lease else 0
        if self._remote is not None and hit < len(prefix):
            if req_id not in self._remote_checked:
                self._remote_checked.add(req_id)
                self._remote.submit(
                    req_id, self.index.encode_hashes(prefix), self.index.mask
                )
            if self.pending(req_id):
                if lease is not None:
                    lease.release()
                return None
        return lease

    def pending(self, req_id: str) -> bool:
        return self._remote is not None and self._remote.pending(req_id)

    def get_load_source(self, req_id: str, chunk_index: int) -> CacheHitSource:
        start, end = self._remote_sources.get(req_id, (0, 0))
        return CacheHitSource.P2P if start <= chunk_index < end else CacheHitSource.HOST

    def contains(self, lease: LookupLease, chunk: Chunk) -> bool:
        index, sub = self._position(chunk)
        return (
            sub >= 0
            and not lease.released
            and lease.covers(self.lanes[chunk.group_id], index, index + 1)
        )

    def load_slots(self, lease: LookupLease, chunks: Sequence[Chunk]) -> LoadStoreSpec:
        ids = []
        for chunk in chunks:
            assert self.contains(lease, chunk)
            lane = self.lanes[chunk.group_id]
            index, sub = self._position(chunk)
            if lane == "full":
                slot = lease.slots[index]
            elif lane == "swa":
                slot = lease.swa_slots[index - lease.swa_start]
            else:
                slot = lease.mamba_slot
            count = self._sub.get(chunk.group_id, (0, 1))[1]
            ids.append(int(slot) * count + sub)
        return CPULoadStoreSpec(ids)

    def reserve(
        self,
        req_id: str,
        hashes: Sequence[BlockHash],
        chunks: Sequence[Chunk],
    ) -> StoreReservation | None:
        if self._reset_pending:
            return None
        prefix = self.prefix_hashes(hashes)
        generation = self.index.generation()
        stores = self._inflight_stores.setdefault(req_id, _ReqStores())
        by_lane: dict[str, dict[int, dict[tuple[int, int], Chunk]]] = {}
        for chunk in chunks:
            index, sub = self._position(chunk)
            if sub >= 0 and index < len(prefix):
                by_lane.setdefault(self.lanes[chunk.group_id], {}).setdefault(
                    index, {}
                )[chunk.group_id, sub] = chunk
        entries: list[_PendingStore] = []
        slot_of: dict[Chunk, int] = {}
        for lane, by_index in by_lane.items():
            members = [
                (g, j)
                for g in self._lane_groups[lane]
                for j in range(self._sub.get(g, (0, 1))[1])
            ]
            complete = sorted(
                i for i, parts in by_index.items() if len(parts) == len(members)
            )
            runs = _index_runs(complete)
            if lane == "full" and runs:
                start, end = runs[0]
                hit = self.index.lookup(prefix[:end])
                stores.full_known = max(stores.full_known, hit)
                runs = [(max(start, hit), end)] if hit < end else []
            for start, end in runs:
                slots = self.index.allocate(end - start, lane)
                if slots is None:
                    if lane == "full":
                        for entry in entries:
                            self.index.recycle(entry.slots, entry.lane)
                        if not stores.full and not stores.aux:
                            self._inflight_stores.pop(req_id, None)
                        return None
                    continue
                selected = [
                    by_index[i][member] for member in members for i in range(start, end)
                ]
                entry = _PendingStore(
                    lane, prefix[:end].copy(), slots, start, end, selected, generation
                )
                entries.append(entry)
                for chunk in selected:
                    index, sub = self._position(chunk)
                    count = self._sub.get(chunk.group_id, (0, 1))[1]
                    slot_of[chunk] = int(slots[index - start]) * count + sub
        selected = [chunk for chunk in chunks if chunk in slot_of]
        for entry in entries:
            (stores.full if entry.lane == "full" else stores.aux).append(entry)
        if not stores.full and not stores.aux:
            self._inflight_stores.pop(req_id, None)
        return StoreReservation(
            req_id, selected, CPULoadStoreSpec([slot_of[c] for c in selected]), entries
        )

    def complete_store(
        self, reservation: StoreReservation, success: bool = True
    ) -> None:
        for entry in reservation.entries:
            assert entry.done is None
            entry.done = success
        self._publish_ready(reservation.req_id)

    def _publish_ready(self, req_id: str) -> None:
        stores = self._inflight_stores.get(req_id)
        if stores is None:
            return
        while stores.full and stores.full[0].done is not None:
            entry = stores.full.pop(0)
            result = self._publish(entry, req_id)
            if result is not None and not result.rejected:
                stores.full_known = max(stores.full_known, entry.end)
        remaining = []
        for entry in stores.aux:
            if entry.done is None or (
                entry.done
                and stores.full_known < entry.end
                and any(full.start < entry.end for full in stores.full)
            ):
                remaining.append(entry)
            else:
                self._publish(entry, req_id)
        stores.aux = remaining
        if not stores.full and not stores.aux:
            del self._inflight_stores[req_id]

    def _publish(self, entry: _PendingStore, req_id: str) -> PublishResult | None:
        if self.index.generation() != entry.generation:
            return None  # a reset since the reservation freed the slots
        if not entry.done:
            # The transfer failed: the slots hold no usable data.
            self.index.recycle(entry.slots, entry.lane)
            return None
        result = self.index.publish(entry.hashes, entry.slots, entry.start, entry.lane)
        if self._enable_events and result.published:
            unused = set(result.unused.tolist())
            chunks = [
                chunk
                for chunk in entry.chunks
                if int(entry.slots[self._position(chunk)[0] - entry.start])
                not in unused
            ]
            self._events.append(Publication(req_id, chunks))
        return result

    def finish_request(self, req_id: str) -> None:
        if self._remote is not None:
            self._remote.cancel(req_id)
        lease = self._remote_ready.pop(req_id, None)
        if lease is not None:
            lease.release()
        self._remote_checked.discard(req_id)
        self._remote_sources.pop(req_id, None)

    def request_reset(self) -> None:
        """Serve no hit and reserve nothing until ``maybe_reset`` clears the pool."""
        self._reset_pending = True
        if self._remote is not None:
            self._remote.clear()
        self._remote_checked.clear()
        self._remote_sources.clear()
        for lease in self._remote_ready.values():
            lease.release()
        self._remote_ready.clear()

    def maybe_reset(self) -> bool:
        """Clear the pool once nothing of this scheduler is in flight."""
        if (
            not self._reset_pending
            or self._inflight_stores
            or self.index.has_leases
            or self.has_pending_work()
        ):
            return False
        self.index.reset()
        self._reset_pending = False
        logger.info("RadixShmem: pool reset (generation %d)", self.index.generation())
        return True

    def step(self) -> None:
        for unused_lease in self._remote_ready.values():
            unused_lease.release()
        self._remote_ready.clear()
        if self._remote is not None:
            for req_id, job, result in self._remote.poll():
                lease = self.index.lease_from_pull(result)
                if lease is not None:
                    self._remote_ready[req_id] = lease
                if result.remote_blocks:
                    start, end = int(job.local_hit), int(result.common_hit)
                    self._remote_sources[req_id] = start, end
                    if self._enable_events:
                        self._events.append(
                            Publication(
                                req_id,
                                [Chunk(self._full_group, i) for i in range(start, end)],
                            )
                        )

    def take_events(self) -> list[Publication]:
        events, self._events = self._events, []
        return events

    def has_pending_work(self) -> bool:
        return self._remote is not None and self._remote.has_pending_work

    def shutdown(self) -> None:
        if self._remote is not None:
            self._remote.close()
        for lease in self._remote_ready.values():
            lease.release()
        self._remote_ready.clear()
        for stores in self._inflight_stores.values():
            for entry in stores.full + stores.aux:
                self.index.recycle(entry.slots, entry.lane)
        self._inflight_stores.clear()
        try:
            self.index.close()
        finally:
            if self._regions is not None:
                self._regions.close()
                self._regions = None


def _index_runs(idxs: list[int]) -> list[tuple[int, int]]:
    """``[a, b)`` runs of sorted, distinct ``idxs``."""
    runs: list[tuple[int, int]] = []
    for i in idxs:
        if runs and runs[-1][1] == i:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    return runs
