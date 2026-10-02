# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from torchax.interop import jax_view, torch_view
from transformers import Qwen2Config
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.grugmoe import GrugMoeShortConv

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.vllm.custom_ops.grug_short_conv import register_grug_short_conv
from tpu_inference.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context


def _reference(sequence, weight):
    result = np.zeros_like(sequence, dtype=np.float32)
    for position in range(len(sequence)):
        for lag in range(min(position + 1, len(weight))):
            result[position] += sequence[position - lag] * weight[lag]
    return result


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("kernel", [2, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.disable_jax_cache
def test_grug_short_conv_preserves_request_histories(tmp_path, dp, kernel, dtype):
    """Exercise the real Torchax custom op across chunking, reordering and slot reuse."""
    if jax.local_device_count() < dp:
        pytest.skip("Two-device case requires a two-device CPU or TPU allocation")
    mesh = jax.make_mesh((dp, 1, 1, 1, 1, 1, 1), MESH_AXIS_NAMES,
                         axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_AXIS_NAMES),
                         devices=jax.local_devices()[:dp])
    # The TPU platform requires model metadata even for an isolated layer op.
    Qwen2Config(vocab_size=128, hidden_size=128, intermediate_size=128,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                architectures=["Qwen2ForCausalLM"]).save_pretrained(tmp_path)
    config = EngineArgs(model=str(tmp_path), skip_tokenizer_init=True,
                        max_model_len=32, max_num_batched_tokens=32,
                        max_num_seqs=4, enable_prefix_caching=False,
                        enforce_eager=True).create_engine_config()
    torch_dtype = torch.bfloat16 if dtype == jnp.bfloat16 else torch.float32
    with set_current_vllm_config(config):
        layer = GrugMoeShortConv(4, kernel, torch_dtype, config.cache_config, "test.sconv")
    weight = ((np.arange(kernel * 4).reshape(kernel, 4) % 7) - 3).astype(np.float32) / 8
    register_grug_short_conv()
    with torchax.default_env():
        layer.weight = torch.nn.Parameter(torch_view(jnp.asarray(weight, dtype)))

    @jax.jit
    def run(state, inputs, metadata):
        caches = [(state,)]
        with torchax.default_env(), set_vllm_model_wrapper_context(
            kv_caches=caches, mesh=mesh, layer_name_to_kvcache_index={"test.sconv": 0}, vllm_config=config
        ), set_forward_context(metadata, config, num_tokens=inputs.shape[0]):
            output = jax_view(layer(torch_view(inputs)))
        return caches[0][0], output

    sequences = [((np.arange(24).reshape(6, 4) + shift) % 9 - 4).astype(np.float32) / 8 for shift in (0, 3, 6)]
    # (sequence, start, count, total length, physical state slot). Slot zero is padding.
    phases = [
        ([(0, 0, 3, 3, 1), (1, 0, 2, 2, 2)], 0),
        ([(1, 2, 1, 3, 2), (0, 3, 2, 5, 1)], 1),
        ([(0, 5, 1, 6, 1), (1, 3, 1, 4, 2)], 2),
        ([(2, 0, 1, 1, 1), (1, 4, 1, 5, 2)], 2),
        ([], 0),
    ]
    state = jnp.full((dp * 4, kernel - 1, 4), 17, dtype)
    expected_state = np.asarray(state).copy()
    for requests, decodes in phases:
        inputs = np.full((dp * 8, 4), 23, np.float32)
        expected = np.zeros_like(inputs)
        starts, indices, lengths = [], [], []
        for shard in range(dp):
            cursor = 0
            shard_starts = [0]
            for sequence, start, count, total, slot in requests:
                values = sequences[sequence] * (1 if shard == 0 else -1)
                inputs[shard * 8 + cursor:shard * 8 + cursor + count] = values[start:start + count]
                expected[shard * 8 + cursor:shard * 8 + cursor + count] = _reference(values, weight)[start:start + count]
                history = np.pad(values[:total], ((kernel - 1, 0), (0, 0)))[-(kernel - 1):]
                expected_state[shard * 4 + slot] = history
                cursor += count
                shard_starts.append(cursor)
            starts.extend(shard_starts + [cursor] * (5 - len(shard_starts)))
            indices.extend([request[4] for request in requests] + [0] * (4 - len(requests)))
            lengths.extend([request[3] for request in requests] + [0] * (4 - len(requests)))
        metadata = AttentionMetadata(
            input_positions=jnp.zeros(dp * 8, jnp.int32),
            seq_lens=jnp.asarray(lengths, jnp.int32),
            query_start_loc=jnp.asarray(starts, jnp.int32),
            request_distribution=jnp.asarray([decodes, decodes, len(requests)] * dp, jnp.int32),
            mamba_state_indices=jnp.asarray(indices, jnp.int32),
            padded_num_reqs=dp * 4,
        )
        with jax.set_mesh(mesh):
            state, actual = run(state, jnp.asarray(inputs, dtype), metadata)
        # Dyadic inputs/taps keep these sums exactly representable in both dtypes.
        np.testing.assert_array_equal(np.asarray(actual).astype(np.float32), expected)
        np.testing.assert_array_equal(np.asarray(state).astype(np.float32), expected_state.astype(np.float32))


def test_short_conv_registration_does_not_acquire_accelerator():
    # The test parent may already own TPU; model metadata inspection must work
    # in a child process without creating another Torchax environment.
    subprocess.run([sys.executable, "-c", """
from jax._src import xla_bridge
from tpu_inference.layers.vllm.custom_ops.grug_short_conv import register_grug_short_conv
register_grug_short_conv()
assert not xla_bridge.backends_are_initialized()
"""], check=True, capture_output=True, text=True)
