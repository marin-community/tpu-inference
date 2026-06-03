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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import save_file

from tpu_inference.models.jax.grugmoe import (
    GrugMoeConfig,
    GrugMoeMLP,
    _load_artifact_tensors,
)


def test_grugmoe_artifact_tensors_support_single_safetensors(tmp_path):
    tensors = {
        "model.embed_tokens.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "model.norm.weight": np.arange(3, dtype=np.float32),
    }
    save_file(tensors, tmp_path / "model.safetensors")

    loaded = _load_artifact_tensors(tmp_path)

    assert set(loaded) == set(tensors)
    np.testing.assert_array_equal(
        loaded["model.embed_tokens.weight"],
        tensors["model.embed_tokens.weight"],
    )
    np.testing.assert_array_equal(loaded["model.norm.weight"], tensors["model.norm.weight"])


def test_grugmoe_artifact_tensors_support_sharded_safetensors(tmp_path):
    first_shard = {"model.embed_tokens.weight": np.arange(6, dtype=np.float32).reshape(2, 3)}
    second_shard = {"model.norm.weight": np.arange(3, dtype=np.float32)}
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    save_file(first_shard, tmp_path / first_name)
    save_file(second_shard, tmp_path / second_name)
    with (tmp_path / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "metadata": {"total_size": 36},
                "weight_map": {
                    "model.embed_tokens.weight": first_name,
                    "model.norm.weight": second_name,
                },
            },
            f,
        )

    loaded = _load_artifact_tensors(tmp_path)

    assert set(loaded) == {"model.embed_tokens.weight", "model.norm.weight"}
    np.testing.assert_array_equal(
        loaded["model.embed_tokens.weight"],
        first_shard["model.embed_tokens.weight"],
    )
    np.testing.assert_array_equal(loaded["model.norm.weight"], second_shard["model.norm.weight"])


def test_grugmoe_route_uses_qb_bias_for_selection_and_unbiased_sigmoid_weights(
):
    cfg = GrugMoeConfig(
        vocab_size=11,
        hidden_dim=3,
        intermediate_dim=2,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
    )
    mlp = GrugMoeMLP(cfg, jnp.float32, nnx.Rngs(jax.random.PRNGKey(0)))

    x = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    mlp.router.value = jnp.array(
        [
            [2.0, 1.0, 0.5, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    mlp.router_bias.value = jnp.array([0.0, 0.0, 3.0, 0.0], dtype=jnp.float32)

    selected, combine_weights = mlp.route(x)

    np.testing.assert_array_equal(np.array(selected), np.array([[2, 0]]))
    expected_weights = jax.nn.sigmoid(jnp.array([[0.5, 2.0]], dtype=jnp.float32))
    np.testing.assert_allclose(
        np.array(combine_weights), np.array(expected_weights), rtol=1e-6, atol=1e-6
    )
