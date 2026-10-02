# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte accounting for TPU arrays that cannot alias hybrid cache storage."""

import dataclasses

from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_page_size
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


@dataclasses.dataclass(frozen=True)
class HybridCacheBudget:
    uniform_page_size_bytes: int
    attention_bytes_per_block: int
    mamba_bytes_per_slot: int


def hybrid_cache_budget(specs: dict[str, KVCacheSpec], physical_pages: dict[str, int]) -> HybridCacheBudget:
    """Compute uniform scheduler pages and actual independently allocated bytes.

    vLLM charges the largest group's layer count times the uniform page size
    per block. TPU allocates every layer separately, so that charge must cover
    the sum of all layers' physical pages, including heterogeneous recurrent
    shapes.
    """
    largest_page = max(physical_pages.values())
    provisional = {
        name: dataclasses.replace(spec, page_size_padded=largest_page)
        for name, spec in specs.items()
    }
    groups = _get_kv_cache_groups_uniform_page_size(provisional)
    group_size = max(len(group.layer_names) for group in groups)
    total = sum(physical_pages.values())
    uniform_page = max(largest_page, (total + group_size - 1) // group_size)
    mamba_bytes = sum(physical_pages[name] for name, spec in specs.items() if isinstance(spec, MambaSpec))
    return HybridCacheBudget(uniform_page, total - mamba_bytes, mamba_bytes)
