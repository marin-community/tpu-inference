# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import pytest
import torch
import torchax
from torchax.ops.mappings import j2t
from transformers import Qwen2Config
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import ReplicatedLinear

from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("numpy_shared", [False, True])
@pytest.mark.disable_jax_cache
def test_replicated_linear_releases_host_storage(tmp_path, dtype, numpy_shared):
    """FP32 router and NumPy-backed weights survive host cleanup and retain values."""
    Qwen2Config(vocab_size=128,
                hidden_size=128,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                architectures=["Qwen2ForCausalLM"]).save_pretrained(tmp_path)
    config = EngineArgs(model=str(tmp_path),
                        skip_tokenizer_init=True,
                        max_model_len=32,
                        max_num_batched_tokens=32,
                        max_num_seqs=2).create_engine_config()
    config.model_config.dtype = dtype
    mesh = test_utils.get_spmd_mesh(1)
    torch.manual_seed(42)
    inputs = torch.randn(8, 128, dtype=dtype) / 10
    weight = torch.randn(16, 128, dtype=dtype) / 20
    bias = torch.randn(16, dtype=dtype) / 20
    expected_weight = weight.T.clone()
    expected_bias = bias.clone()
    # The TPU linear path rounds matmul to its output dtype before adding bias.
    expected = (torch.nn.functional.linear(inputs, weight) + bias).float()
    with set_current_vllm_config(config):
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist'}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        linear = ReplicatedLinear(
            input_size=128,
            output_size=16,
            bias=True,
            return_bias=False,
            params_dtype=dtype,
            quant_config=get_tpu_quantization_config(config, mesh))
    linear.weight.data = weight
    linear.bias.data = bias
    if numpy_shared:
        # Creating a view makes storage non-resizable even after the view dies.
        # The byte view also covers BF16, which NumPy cannot represent directly.
        linear.weight.detach().view(torch.uint8).numpy()
        linear.bias.detach().view(torch.uint8).numpy()
    # Isolate storage transfer from TPU's default reduced-precision FP32 matmul.
    with torchax.default_env(), jax.set_mesh(mesh), \
            jax.default_matmul_precision("highest"):
        linear.quant_method.process_weights_after_loading(linear)
        actual_weight = j2t(linear.weight.to(torch.float32)).to(dtype)
        actual_bias = j2t(linear.bias.to(torch.float32)).to(dtype)
        actual = j2t(linear(inputs.to("jax")).to(torch.float32))
    torch.testing.assert_close(actual_weight, expected_weight, atol=0, rtol=0)
    torch.testing.assert_close(actual_bias, expected_bias, atol=0, rtol=0)
    torch.testing.assert_close(actual.to(dtype), expected.to(dtype))
