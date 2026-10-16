# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply vLLM cache-manager hit rules to a pinned RadixShmem prefix."""

from math import lcm
from typing import TYPE_CHECKING, cast

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock, eagle_proof_margin
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.radixshmem.manager import Chunk, LookupLease, RadixShmemManager

if TYPE_CHECKING:
    from .scheduler import GroupOffloadConfig


class _PinnedBlockPool:
    def __init__(
        self, hashes: list[BlockHash], manager: RadixShmemManager, lease: LookupLease
    ):
        self.hash_block_size = manager.tokens_per_hash
        self.null_block = KVCacheBlock(0)
        self._present = KVCacheBlock(1)
        self._positions = {
            h: (i + 1) * self.hash_block_size for i, h in enumerate(hashes)
        }
        self.manager, self.lease = manager, lease

    def get_cached_block(
        self, block_hash: BlockHash, group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        end = self._positions.get(block_hash, 0)
        if not end:
            return None
        for gid in group_ids:
            size = self.manager.group_tokens[gid]
            if self.manager.lanes[gid] == "mamba" and end % size:
                return None
            chunk = Chunk(gid, (end - 1) // size)
            if not self.manager.contains(self.lease, chunk):
                return None
        return [self._present] * len(group_ids)


def plan_load(
    hashes: list[BlockHash],
    groups: tuple["GroupOffloadConfig", ...],
    manager: RadixShmemManager,
    lease: LookupLease,
    max_tokens: int,
) -> tuple[int, tuple[list[bool], ...]]:
    pool = _PinnedBlockPool(hashes, manager, lease)
    alignment = lcm(*(g.tokens_per_chunk for g in groups))
    hit = max_tokens
    masks: list[list[bool]] = [[] for _ in groups]
    # EAGLE groups drop the trailing block once; a later tightening by another
    # group makes them match again from the new bound.
    verified: set[int] = set()
    while hit:
        previous = hit
        for i, group in enumerate(groups):
            eagle = group.is_eagle_group and i not in verified
            bound = hit
            if eagle and not isinstance(group.kv_cache_spec, MambaSpec):
                # Match one block past the hit so the drop lands on the hit.
                bound = min(
                    max_tokens,
                    hit
                    + eagle_proof_margin(
                        group.tokens_per_block, manager.tokens_per_hash, False
                    ),
                )
            blocks, found = group.manager_cls.find_longest_cache_hit(
                block_hashes=hashes,
                max_length=bound,
                kv_cache_group_ids=[group.group_idx],
                block_pool=cast(BlockPool, pool),
                kv_cache_spec=group.kv_cache_spec,
                drop_eagle_block=eagle,
                alignment_tokens=alignment,
            )
            masks[i] = [block is not pool.null_block for block in blocks[0]]
            if eagle:
                verified.add(i)
            elif found < hit:
                verified.clear()
            hit = min(hit, found)
        if hit == previous:
            break
    return hit, tuple(masks)
