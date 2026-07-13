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

import json
from types import SimpleNamespace

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh
from safetensors.numpy import save_file

from tpu_inference.models.jax import grugmoe


class _SpyNorm:

    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x + self.offset


def _minimal_model(*, is_first_rank: bool) -> grugmoe.GrugMoeModel:
    model = object.__new__(grugmoe.GrugMoeModel)
    object.__setattr__(model, "is_first_rank", is_first_rank)
    object.__setattr__(model, "is_last_rank", False)
    object.__setattr__(model, "embed_norm", _SpyNorm(1.0))
    object.__setattr__(model, "embed_gated_norm", _SpyNorm(10.0))
    object.__setattr__(model, "config", SimpleNamespace(sliding_window=16))
    object.__setattr__(model, "layers", [])
    return model


def test_grugmoe_skips_embed_norm_for_received_pp_hidden_states():
    model = _minimal_model(is_first_rank=False)
    inputs_embeds = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    attention_metadata = SimpleNamespace(
        input_positions=jnp.arange(inputs_embeds.shape[0]))

    _kv_caches, hidden_states, routed_experts = model(
        [],
        input_ids=None,
        attention_metadata=attention_metadata,
        inputs_embeds=inputs_embeds,
    )

    assert jnp.array_equal(hidden_states, inputs_embeds)
    assert model.embed_norm.calls == 0
    assert model.embed_gated_norm.calls == 0
    assert routed_experts is None


def test_grugmoe_applies_embed_norm_on_first_pp_rank():
    model = _minimal_model(is_first_rank=True)
    inputs_embeds = jnp.zeros((3, 4), dtype=jnp.float32)
    attention_metadata = SimpleNamespace(
        input_positions=jnp.arange(inputs_embeds.shape[0]))

    _kv_caches, hidden_states, _routed_experts = model(
        [],
        input_ids=None,
        attention_metadata=attention_metadata,
        inputs_embeds=inputs_embeds,
    )

    assert jnp.array_equal(hidden_states, jnp.full_like(inputs_embeds, 11.0))
    assert model.embed_norm.calls == 1
    assert model.embed_gated_norm.calls == 1


def test_grugmoe_route_selects_configured_top_k_experts():
    mlp = object.__new__(grugmoe.GrugMoeMLP)
    object.__setattr__(mlp, "cfg", SimpleNamespace(num_experts_per_token=2))
    object.__setattr__(
        mlp,
        "router",
        SimpleNamespace(value=jnp.asarray([
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -1.0],
        ])),
    )
    object.__setattr__(
        mlp,
        "router_bias",
        SimpleNamespace(value=jnp.asarray([0.0, 2.0, -10.0])),
    )

    selected, combine_weights = mlp.route(jnp.asarray([[1.0, 2.0]]))

    expected_weights = jax.nn.sigmoid(jnp.asarray([[2.0, 1.0]]))
    assert selected.tolist() == [[1, 0]]
    assert jnp.allclose(combine_weights, expected_weights)


def test_grugmoe_router_dot_matches_levanter_bf16_policy():
    mlp = object.__new__(grugmoe.GrugMoeMLP)
    object.__setattr__(mlp, "cfg", SimpleNamespace(num_experts_per_token=3))
    router = jnp.asarray(
        [
            [0.58984375, 1.21875, -0.349609375],
            [-0.248046875, -1.046875, 0.07568359375],
            [1.09375, 1.03125, 2.75],
            [-2.0, -2.234375, 0.05908203125],
            [-1.5234375, 1.0546875, 0.365234375],
            [1.4140625, 0.474609375, -1.984375],
            [0.68359375, 0.640625, 1.1015625],
            [2.484375, 0.0791015625, 0.30078125],
        ],
        dtype=jnp.bfloat16,
    )
    object.__setattr__(mlp, "router", SimpleNamespace(value=router))
    object.__setattr__(
        mlp,
        "router_bias",
        SimpleNamespace(value=jnp.zeros((3, ), dtype=jnp.float32)),
    )
    x = jnp.asarray(
        [[
            0.5859375,
            2.546875,
            -0.671875,
            -0.18359375,
            0.80078125,
            0.435546875,
            -0.16796875,
            -0.291015625,
        ]],
        dtype=jnp.bfloat16,
    )

    selected, combine_weights = mlp.route(x)

    logits = jnp.einsum("td,de->te", x, router).astype(jnp.float32)
    expected_values, expected_indices = jax.lax.top_k(logits, 3)
    assert jnp.array_equal(selected, expected_indices)
    assert jnp.array_equal(combine_weights,
                           jax.nn.sigmoid(expected_values).astype(x.dtype))


def test_grugmoe_gated_norm_matches_levanter_bf16_policy():
    norm = object.__new__(grugmoe.GrugMoeGatedNorm)
    w_down = jnp.asarray(
        [[0.1001, -0.2002], [0.3003, 0.4004], [-0.5005, 0.6006],
         [0.7007, -0.8008]],
        dtype=jnp.bfloat16,
    )
    w_up = jnp.asarray(
        [[0.9009, -1.001, 1.101, -1.201], [-1.301, 1.401, -1.501, 1.601]],
        dtype=jnp.bfloat16,
    )
    object.__setattr__(norm, "w_down", SimpleNamespace(value=w_down))
    object.__setattr__(norm, "w_up", SimpleNamespace(value=w_up))
    x = jnp.asarray([[1.701, -1.801, 1.901, -2.001]], dtype=jnp.bfloat16)

    actual = norm(x)

    hidden = jax.nn.silu(jnp.einsum("...d,dr->...r", x, w_down))
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", hidden, w_up))
    expected = x * gate.astype(x.dtype)
    assert jnp.array_equal(actual, expected)


def test_grugmoe_half_rope_leaves_second_half_unrotated():
    q = jnp.arange(16, dtype=jnp.bfloat16).reshape(2, 1, 8)
    k = (q + 1).astype(jnp.bfloat16)

    rotated_q, rotated_k = grugmoe._apply_half_rotary(
        q,
        k,
        jnp.asarray([0, 1]),
        head_dim=8,
        theta=10_000.0,
    )

    assert rotated_q.dtype == q.dtype
    assert rotated_k.dtype == k.dtype
    assert jnp.array_equal(rotated_q[..., 4:], q[..., 4:])
    assert jnp.array_equal(rotated_k[..., 4:], k[..., 4:])
    assert not jnp.array_equal(rotated_q[1, ..., :4], q[1, ..., :4])


def test_grugmoe_attention_uses_full_context_and_no_rope_on_long_layers():
    cfg = grugmoe.GrugMoeConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=0,
        num_layers=6,
        num_heads=1,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=64,
        sliding_window=16,
        qk_mult_long_scale=1.25,
    )
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("model", ))

    short = grugmoe.GrugMoeAttention(cfg,
                                     jnp.bfloat16,
                                     nnx.Rngs(0),
                                     mesh,
                                     is_long=False)
    long = grugmoe.GrugMoeAttention(cfg,
                                    jnp.bfloat16,
                                    nnx.Rngs(1),
                                    mesh,
                                    is_long=True)

    assert short.sliding_window == 16
    assert short.use_rope is True
    assert short.qk_mult_scale == 1.0
    assert long.sliding_window is None
    assert long.use_rope is False
    assert long.qk_mult_scale == 1.25


def test_grugmoe_long_layer_schedule_includes_every_fourth_and_final_layer():
    assert grugmoe._is_long_layer(3, 6)
    assert grugmoe._is_long_layer(5, 6)
    assert not grugmoe._is_long_layer(4, 6)


def test_grugmoe_rejects_pko_checkpoint():
    with pytest.raises(NotImplementedError, match="Partial Key Offset"):
        grugmoe.GrugMoeConfig(vocab_size=32, disable_pko=False).validate()


def test_grugmoe_forwards_qb_bias_to_distributed_experts():

    class _CapturingQuantMethod:

        def __init__(self):
            self.router_logits = None
            self.correction_bias = None

        def apply_jax(
            self,
            _experts,
            x,
            *,
            router_logits,
            expert_logits_correction_bias,
            topk_weights_sum,
        ):
            self.router_logits = router_logits
            self.correction_bias = expert_logits_correction_bias
            self.topk_weights_sum = topk_weights_sum
            return x

    quant_method = _CapturingQuantMethod()
    mlp = object.__new__(grugmoe.GrugMoeMLP)
    object.__setattr__(mlp, "cfg", SimpleNamespace(num_experts_per_token=2))
    object.__setattr__(
        mlp,
        "router",
        SimpleNamespace(value=jnp.asarray([
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -1.0],
        ])),
    )
    correction_bias = jnp.asarray([0.0, 2.0, -10.0])
    object.__setattr__(
        mlp,
        "router_bias",
        SimpleNamespace(value=correction_bias),
    )
    object.__setattr__(
        mlp,
        "experts",
        SimpleNamespace(quant_method=quant_method),
    )

    x = jnp.asarray([[1.0, 2.0]])
    output, selected = mlp(x)

    assert jnp.array_equal(output, x)
    assert jnp.array_equal(selected, jnp.asarray([[1, 0]]))
    assert jnp.array_equal(quant_method.router_logits,
                           jnp.asarray([[1.0, 2.0, 0.0]]))
    assert quant_method.correction_bias is correction_bias
    assert quant_method.topk_weights_sum == 2.5


def test_compute_logits_uses_tied_token_embeddings_when_lm_head_is_missing():
    model = object.__new__(grugmoe.GrugMoeForCausalLM)
    object.__setattr__(model, "lm_head", grugmoe.PPMissingLayer())
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(token_embed=SimpleNamespace(value=jnp.asarray([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]))),
    )

    logits = model.compute_logits(jnp.asarray([[10.0, 100.0]]))

    assert jnp.array_equal(logits, jnp.asarray([[210.0, 430.0, 650.0]]))


def test_grugmoe_artifact_config_requires_schema_version(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "grug_moe"}))

    with pytest.raises(ValueError, match="grugmoe_artifact_schema_version=1"):
        grugmoe._read_artifact_config(tmp_path)

    config_path.write_text(
        json.dumps({
            "model_type": "grug_moe",
            "grugmoe_artifact_schema_version": 1,
        }))

    assert (grugmoe._read_artifact_config(tmp_path)
            ["grugmoe_artifact_schema_version"] == 1)


def test_remote_safetensors_artifact_downloads_each_shard_once(
        tmp_path, monkeypatch):
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"a": np.asarray([1.0]), "b": np.asarray([2.0])}, shard_a)
    save_file({"c": np.asarray([3.0])}, shard_b)
    fs = fsspec.filesystem("memory")
    fs.put(str(shard_a), "/artifact/model-00001-of-00002.safetensors")
    fs.put(str(shard_b), "/artifact/model-00002-of-00002.safetensors")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    tensors = grugmoe._RemoteSafetensorsArtifactTensors(
        "memory://artifact",
        {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00001-of-00002.safetensors",
            "c": "model-00002-of-00002.safetensors",
        },
    )
    loaded_shards = []
    original_load_shard = tensors._load_shard

    def load_shard(shard_name):
        loaded_shards.append(shard_name)
        original_load_shard(shard_name)

    monkeypatch.setattr(tensors, "_load_shard", load_shard)

    assert tensors["a"].tolist() == [1.0]
    assert tensors["c"].tolist() == [3.0]
    assert tensors["b"].tolist() == [2.0]
    assert loaded_shards == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert tensors._cached_shards == {}
    assert not list(tmp_path.glob("grugmoe-shard-*.safetensors"))
