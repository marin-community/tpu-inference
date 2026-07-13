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

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from tpu_inference.layers.common import moe as common_moe
from tpu_inference.layers.common.fused_moe_gmm import \
    _normalize_topk_weights, _select_topk_weights_and_indices
from tpu_inference.layers.common.moe import MoEBackend


def test_logit_correction_bias_selects_before_sigmoid():
    logits = jnp.asarray([[2.0, 1.0]], dtype=jnp.float32)
    correction_bias = jnp.asarray([0.0, 2.0], dtype=jnp.float32)

    weights, indices = _select_topk_weights_and_indices(
        logits,
        topk=1,
        scoring_fn="sigmoid",
        hash_based_topk_indices=None,
        expert_score_correction_bias=None,
        expert_logits_correction_bias=correction_bias,
    )

    assert jnp.array_equal(indices, jnp.asarray([[1]]))
    assert jnp.allclose(weights, jax.nn.sigmoid(jnp.asarray([[1.0]])))


def test_topk_weights_can_normalize_to_model_specific_sum():
    weights = jnp.asarray([[0.2, 0.3]], dtype=jnp.float32)

    normalized = _normalize_topk_weights(
        weights,
        renormalize=False,
        normalization_sum=2.5,
    )

    assert jnp.allclose(normalized, jnp.asarray([[1.0, 1.5]]))
    assert jnp.allclose(normalized.sum(axis=-1), jnp.asarray([2.5]))


def test_gmm_backend_forwards_logit_correction_bias(monkeypatch):
    correction_bias = jnp.asarray([0.0, 2.0], dtype=jnp.float32)
    captured = {}

    def fake_fused_moe_func(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(common_moe, "fused_moe_func", fake_fused_moe_func)
    layer = SimpleNamespace(
        _get_name=lambda: "test_moe",
        activation="silu",
        top_k=1,
        renormalize=False,
        use_ep=True,
        scoring_func="sigmoid",
    )
    weights = SimpleNamespace(
        w13_weight=jnp.zeros((2, 2, 2)),
        w2_weight=jnp.zeros((2, 2, 2)),
        w13_weight_scale=None,
        w2_weight_scale=None,
        w13_bias=None,
        w2_bias=None,
    )
    hidden_states = jnp.ones((1, 2))

    output = common_moe.moe_apply(
        layer,
        hidden_states,
        jnp.ones((1, 2)),
        weights,
        MoEBackend.GMM_EP,
        SimpleNamespace(),
        {
            "expert_logits_correction_bias": correction_bias,
            "topk_weights_sum": 2.5,
        },
    )

    assert output is hidden_states
    assert captured["expert_logits_correction_bias"] is correction_bias
    assert captured["topk_weights_sum"] == 2.5
