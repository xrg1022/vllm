# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extend vLLM cache configuration with the RadixShmem pool geometry."""

import os
from dataclasses import fields, replace

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids as get_offloading_group_ids,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
    iter_layer_specs,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.radixshmem.geometry import RadixGroupConfig

logger = init_logger(__name__)


def _sliding_window(group: KVCacheGroupSpec) -> tuple[int | None, int]:
    """``(sliding_window, extra_retained_tokens)``; packed groups are read per
    layer, as the scheduler does."""
    specs = iter_layer_specs(group.kv_cache_spec)
    first = next(iter(specs), None)
    if isinstance(first, SlidingWindowSpec) and all(
        isinstance(s, SlidingWindowSpec) for s in specs
    ):
        return first.sliding_window, first.extra_retained_tokens
    return None, 0


def _kv_bytes_per_block(group: KVCacheGroupSpec) -> int:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.page_size_bytes
    return spec.page_size_bytes * len(group.layer_names)


def _check_supported(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
    assert vllm_config.kv_transfer_config is not None
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "blocks_per_chunk" in extra:
        raise ValueError(
            "RadixShmem takes the offloaded chunk size as 'block_size' (tokens), "
            "not 'blocks_per_chunk'"
        )
    if extra.get("remote") and vllm_config.parallel_config.nnodes_within_dp != 1:
        raise ValueError(
            "RadixShmem remote sharing requires each DP replica's TP workers on "
            "one node"
        )
    if vllm_config.parallel_config.decode_context_parallel_size > 1:
        raise ValueError("RadixShmem does not support decode context parallelism")
    if (
        vllm_config.cache_config.prefix_caching_hash_algo.startswith("xxhash")
        and os.getenv("PYTHONHASHSEED") is None
    ):
        logger.warning(
            "RadixShmem: PYTHONHASHSEED is not set, so with %s vLLM seeds "
            "NONE_HASH randomly in every process and DP ranks share no prefix. "
            "Set the same PYTHONHASHSEED in every vLLM process.",
            vllm_config.cache_config.prefix_caching_hash_algo,
        )


def build_radix_offloading_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> OffloadingConfig:
    """The native config with ``RadixGroupConfig`` groups."""
    _check_supported(vllm_config, kv_cache_config)
    config = build_offloading_config(vllm_config, kv_cache_config)
    groups = []
    for g in config.groups:
        kv_group = kv_cache_config.kv_cache_groups[g.group_id]
        sliding_window, extra_retained = _sliding_window(kv_group)
        groups.append(
            RadixGroupConfig(
                **{f.name: getattr(g, f.name) for f in fields(g)},
                kv_cache_spec_kind=get_kv_cache_spec_kind(kv_group.kv_cache_spec).value,
                kv_bytes_per_block=_kv_bytes_per_block(kv_group),
                sliding_window=sliding_window,
                extra_retained_tokens=extra_retained,
            )
        )
    return replace(config, groups=tuple(groups))
