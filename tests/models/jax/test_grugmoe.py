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

import jax
import jax.numpy as jnp
import pytest

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
        input_positions=jnp.arange(inputs_embeds.shape[0])
    )

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
        input_positions=jnp.arange(inputs_embeds.shape[0])
    )

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
        SimpleNamespace(
            value=jnp.asarray(
                [
                    [1.0, 0.0, 2.0],
                    [0.0, 1.0, -1.0],
                ]
            )
        ),
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


def test_compute_logits_uses_tied_token_embeddings_when_lm_head_is_missing():
    model = object.__new__(grugmoe.GrugMoeForCausalLM)
    object.__setattr__(model, "lm_head", grugmoe.PPMissingLayer())
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(
            token_embed=SimpleNamespace(
                value=jnp.asarray(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                    ]
                )
            )
        ),
    )

    logits = model.compute_logits(jnp.asarray([[10.0, 100.0]]))

    assert jnp.array_equal(logits, jnp.asarray([[210.0, 430.0, 650.0]]))


def test_grugmoe_artifact_config_requires_schema_version(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "grug_moe"}))

    with pytest.raises(ValueError, match="grugmoe_artifact_schema_version=1"):
        grugmoe._read_artifact_config(tmp_path)

    config_path.write_text(
        json.dumps(
            {
                "model_type": "grug_moe",
                "grugmoe_artifact_schema_version": 1,
            }
        )
    )

    assert (
        grugmoe._read_artifact_config(tmp_path)["grugmoe_artifact_schema_version"] == 1
    )
