# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionBackend
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_page_size,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tpu_inference.layers.common.grug_short_conv import grug_short_conv_local
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.vllm.backends.flash_attn import PallasAttentionBackend
from tpu_inference.runner.hybrid_cache import hybrid_cache_budget
from tpu_inference.runner.kv_cache_manager import KVCacheManager


def _hero_specs(layers, hidden, local_k, global_k, kv_heads=2):
    specs = {}
    for layer in range(layers):
        specs[f"layer.{layer}.attn"] = FullAttentionSpec(
            block_size=16, num_kv_heads=kv_heads, head_size=128, dtype=torch.bfloat16)
        widths = {"k": global_k if (layer + 1) % 4 == 0 else local_k,
                  "attn": hidden, "mlp": hidden}
        for site, width in widths.items():
            specs[f"layer.{layer}.sconv_{site}"] = MambaSpec(
                block_size=16, shapes=((3, width),), dtypes=(torch.bfloat16,),
                mamba_type=MambaAttentionBackendEnum.SHORT_CONV)
    return specs


def test_production_hero_budget_counts_each_recurrent_shape():
    specs = _hero_specs(48, 6144, 1536, 768, kv_heads=12)
    pages = {name: spec.page_size_bytes for name, spec in specs.items()}
    budget = hybrid_cache_budget(specs, pages)
    assert budget.mamba_bytes_per_slot == 3 * 2 * (36 * 1536 + 12 * 768 + 96 * 6144)
    assert budget.attention_bytes_per_block == 48 * 2 * 16 * 12 * 128 * 2
    padded = {name: dataclasses.replace(spec, page_size_padded=budget.uniform_page_size_bytes)
              for name, spec in specs.items()}
    groups = _get_kv_cache_groups_uniform_page_size(padded)
    scheduler_bytes = max(len(group.layer_names) for group in groups) * budget.uniform_page_size_bytes
    assert scheduler_bytes >= sum(pages.values())
    # The previous first-K-state approximation undercharges the actual states.
    assert 144 * pages["layer.0.sconv_k"] < budget.mamba_bytes_per_slot


@pytest.mark.parametrize("compact", [False, True])
def test_heterogeneous_padded_groups_allocate_independent_bounded_arrays(tmp_path, monkeypatch, compact):
    Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=128,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                architectures=["Qwen2ForCausalLM"]).save_pretrained(tmp_path)
    config = EngineArgs(model=str(tmp_path), skip_tokenizer_init=True,
                        max_model_len=32, max_num_batched_tokens=32,
                        max_num_seqs=2, enable_prefix_caching=False,
                        enforce_eager=True).create_engine_config()
    config.cache_config.block_size = 16
    config.cache_config.num_gpu_blocks_override = None if compact else 8
    config.cache_config.gpu_memory_utilization = 1.0
    available = 2 * 2**20
    monkeypatch.setattr("tpu_inference.runner.kv_cache_manager.utils.hbm_usage_bytes",
                        lambda _: [(0, available)])
    mesh = jax.make_mesh((1,) * len(MESH_AXIS_NAMES), MESH_AXIS_NAMES,
                         axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_AXIS_NAMES),
                         devices=jax.local_devices()[:1])
    runner = SimpleNamespace(
        model_config=config.model_config, vllm_config=config, mesh=mesh,
        cache_config=config.cache_config, kv_cache_dtype=torch.bfloat16,
        kv_caches=[], layer_name_to_kvcache_index={}, max_num_reqs=2,
        max_model_len=32, max_num_tokens=32, dp_size=1,
        persistent_batch_manager=SimpleNamespace(),
    )
    manager = KVCacheManager(runner)
    specs = _hero_specs(11, 128, 64, 32)
    physical_pages = {name: spec.page_size_bytes for name, spec in specs.items()}
    manager.update_mamba_page_size_padded(specs)
    groups = _get_kv_cache_groups_uniform_page_size(specs)
    assert len({len(group.layer_names) for group in groups}) > 1
    # Only telemetry is replaced; grouping, allocation and indexing are real.
    monkeypatch.setattr("tpu_inference.runner.kv_cache_manager.utils.hbm_usage_gb", lambda _: 0)
    with set_current_vllm_config(config):
        # Match EngineCore: resolve backend-supported layouts before allocation.
        layouts = get_supported_kv_cache_layouts([PallasAttentionBackend, ShortConvAttentionBackend])
        resolve_kv_cache_layout(config, [[layout.name for layout in layouts]], specs.values())
        cache_config = get_kv_cache_config_from_groups(config, groups, 16 * 2**20)
        manager.initialize_kv_cache(cache_config)
    allocated = 0
    checked_shapes = set()
    for name, spec in specs.items():
        cache = runner.kv_caches[runner.layer_name_to_kvcache_index[name]]
        arrays = cache if isinstance(spec, MambaSpec) else (cache,)
        for array in arrays:
            allocated += array.nbytes
        if isinstance(spec, MambaSpec):
            assert cache[0].shape == (3, *spec.shapes[0])
            if spec.shapes[0] not in checked_shapes:
                width = spec.shapes[0][-1]
                state = jnp.full_like(cache[0], 9)
                output, updated = jax.jit(grug_short_conv_local)(
                    jnp.ones((2, width), jnp.bfloat16), state,
                    jnp.ones((4, width), jnp.bfloat16),
                    jnp.asarray([0, 2], jnp.int32), jnp.asarray([2], jnp.int32),
                    jnp.asarray([0, 0, 1], jnp.int32), jnp.asarray([2], jnp.int32))
                np.testing.assert_array_equal(np.asarray(output), np.broadcast_to([[1], [2]], (2, width)))
                expected_state = np.full(cache[0].shape, 9)
                expected_state[2] = np.broadcast_to([[0], [1], [1]], (3, width))
                np.testing.assert_array_equal(np.asarray(updated), expected_state)
                checked_shapes.add(spec.shapes[0])
        else:
            assert cache.shape[0] == cache_config.num_blocks
    expected = sum(page * (3 if isinstance(specs[name], MambaSpec) else cache_config.num_blocks)
                   for name, page in physical_pages.items())
    assert allocated == expected
    charged = max(len(group.layer_names) for group in groups) * next(iter(specs.values())).page_size_bytes * cache_config.num_blocks
    if compact:
        assert allocated <= available
        assert available - allocated < sum(page for name, page in physical_pages.items()
                                            if not isinstance(specs[name], MambaSpec))
    else:
        assert allocated <= charged
    assert len(set(runner.layer_name_to_kvcache_index.values())) == len(specs)
