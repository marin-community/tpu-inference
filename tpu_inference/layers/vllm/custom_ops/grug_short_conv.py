# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Torchax bridge for the Grug short-convolution custom op."""

import torch
import torchax
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.grugmoe import GrugMoeShortConv

from tpu_inference.layers.common.grug_short_conv import grug_short_conv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import truncate_sharded_tensor
from tpu_inference.models.vllm.vllm_model_wrapper_context import get_vllm_model_wrapper_context
from tpu_inference.utils import get_mesh_shape_product


def grug_short_conv_tpu(inputs: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    """Update runner-owned convolution history and write the custom op's output."""
    context = get_vllm_model_wrapper_context()
    config = context.vllm_config
    assert config is not None
    if (config.cache_config.enable_prefix_caching or config.cache_config.mamba_cache_mode != "none"
            or config.speculative_config is not None):
        raise NotImplementedError("Grug TPU short convolution requires no prefix caching or speculative decoding")
    if get_mesh_shape_product(context.mesh, ShardingAxisName.ATTN_HEAD) != 1:
        raise NotImplementedError("Grug TPU short convolution currently requires TP1/EP1")
    if config.parallel_config.prefill_context_parallel_size != 1:
        raise NotImplementedError("Grug TPU short convolution requires prefill context parallelism of one")
    forward = get_forward_context()
    metadata = forward.attn_metadata
    if isinstance(metadata, dict):
        metadata = metadata[layer_name]
    layer: GrugMoeShortConv = forward.no_compile_layers[layer_name]
    index = context.layer_name_to_kvcache_index[layer_name]
    (state,) = context.kv_caches[index]
    if state.shape[1:] != (layer.kernel_size - 1, layer.dim):
        raise ValueError("Grug TPU short convolution requires state layout [slot, lag, channel]")
    dp = get_mesh_shape_product(context.mesh, ShardingAxisName.ATTN_DATA)
    requests_per_shard = metadata.padded_num_reqs // dp
    starts = truncate_sharded_tensor(metadata.query_start_loc, requests_per_shard + 1, dp)
    indices = truncate_sharded_tensor(metadata.mamba_state_indices, requests_per_shard, dp)
    lengths = truncate_sharded_tensor(metadata.seq_lens, requests_per_shard, dp)
    result, state = grug_short_conv(
        jax_view(inputs), state, jax_view(layer.weight), starts, indices,
        metadata.request_distribution, lengths, context.mesh,
    )
    context.kv_caches[index] = (state,)
    output.copy_(torch_view(result))


def register_grug_short_conv() -> None:
    """Register the mutating Torchax op after the Marin Grug model defines it."""
    torchax.default_env().override_op_definition(torch.ops.vllm.grug_moe_short_conv.default, grug_short_conv_tpu)
