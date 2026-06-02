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
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tpu_inference.models.jax.grugmoe import GrugMoeConfig, GrugMoeMLP


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
