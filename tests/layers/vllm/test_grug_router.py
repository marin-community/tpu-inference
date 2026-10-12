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
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.grugmoe import GrugMoeRouter

from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.interface.moe import FusedMoEFactory
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config


@pytest.mark.disable_jax_cache
def test_grug_custom_routes_reach_tpu_experts(tmp_path):
    """Biased selection and normalized sigmoid weights survive monolithic MoE."""
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
    config.model_config.dtype = torch.bfloat16
    mesh = test_utils.get_spmd_mesh(1)
    torch.manual_seed(42)
    inputs = torch.randn(8, 128, dtype=torch.bfloat16) / 10
    logits = torch.linspace(-1, 1, 8).repeat(8, 1)
    bias = torch.tensor([4., 3., 0., 0., 0., 0., 0., 0.])
    w1 = torch.randn(8, 256, 128, dtype=torch.bfloat16) / 20
    w2 = torch.randn(8, 128, 128, dtype=torch.bfloat16) / 20
    # Independent dense reference: biased top-k, unbiased sigmoid weights,
    # normalized to 2.5. Ordinary softmax would select experts 7 and 6.
    selected = torch.argsort(logits + bias, descending=True)[:, :2]
    weights = torch.sigmoid(torch.gather(logits, 1, selected))
    weights = (2.5 * weights / (weights.sum(-1, keepdim=True) + 1e-9)).to(
        inputs.dtype)
    projected = torch.einsum("td,tkfd->tkf", inputs, w1[selected])
    gate, up = projected.chunk(2, dim=-1)
    expert_outputs = torch.einsum("tkf,tkdf->tkd",
                                  torch.nn.functional.silu(gate) * up,
                                  w2[selected])
    expected = (expert_outputs * weights[..., None]).sum(1).float()
    assert torch.equal(selected, torch.tensor([[0, 1]]).expand(8, -1))

    with set_current_vllm_config(config):
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist'}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        moe = FusedMoEFactory(num_experts=8,
                              top_k=2,
                              hidden_size=128,
                              intermediate_size=128,
                              renormalize=False,
                              tp_size=1,
                              dp_size=1,
                              router=GrugMoeRouter(2, 8, bias),
                              quant_config=get_tpu_quantization_config(
                                  config, mesh))
    moe.routed_experts.w13_weight.data = w1
    moe.routed_experts.w2_weight.data = w2
    with torchax.default_env(), set_forward_context(
            None, config), jax.set_mesh(mesh):
        moe.router.bias = bias.to("jax")
        moe.routed_experts.quant_method.process_weights_after_loading(
            moe.routed_experts)
        actual = j2t(moe(inputs.to("jax"), logits.to("jax")).to(torch.float32))
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
