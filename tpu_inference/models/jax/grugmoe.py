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

"""Correctness-first native JAX implementation of Marin GrugMoE."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from safetensors import safe_open
from transformers import PretrainedConfig
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLmHead
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors

_GATED_NORM_RANK = 128
_ARTIFACT_CONFIG_FILE = "config.json"
_ARTIFACT_WEIGHTS_FILE = "model.safetensors"
_ARTIFACT_WEIGHTS_INDEX_FILE = "model.safetensors.index.json"
_ARTIFACT_MODEL_TYPE = "grug_moe"


class GrugMoeHfConfig(PretrainedConfig):
    model_type = _ARTIFACT_MODEL_TYPE

    def __init__(
        self,
        architectures: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        aliases = {
            "hidden_size": "hidden_dim",
            "intermediate_size": "intermediate_dim",
            "moe_intermediate_size": "intermediate_dim",
            "shared_expert_intermediate_size": "shared_expert_intermediate_dim",
            "num_hidden_layers": "num_layers",
            "num_attention_heads": "num_heads",
            "num_key_value_heads": "num_kv_heads",
            "max_position_embeddings": "max_seq_len",
        }
        for alias, source in aliases.items():
            if alias not in kwargs and source in kwargs:
                kwargs[alias] = kwargs[source]
            if source not in kwargs and alias in kwargs:
                kwargs[source] = kwargs[alias]

        super().__init__(
            architectures=architectures or ["GrugMoeForCausalLM"],
            **kwargs,
        )


@dataclass(frozen=True)
class GrugMoeArtifactLoadReport:
    consumed: frozenset[str]
    missing: frozenset[str]
    unexpected: frozenset[str]


def _config_attr(config: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _rope_theta(config: Any) -> float:
    rope = getattr(config, "rope", None)
    if isinstance(rope, dict) and "theta" in rope:
        return float(rope["theta"])
    if hasattr(rope, "theta"):
        return float(rope.theta)

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        for name in ("theta", "rope_theta"):
            if name in rope_parameters:
                return float(rope_parameters[name])

    return float(getattr(config, "rope_theta", 10000.0))


@dataclass(frozen=True)
class GrugMoeConfig:
    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    sliding_window: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.0
    rope_theta: float = 10000.0

    @classmethod
    def from_hf_config(cls, config: Any) -> "GrugMoeConfig":
        max_seq_len = int(
            _config_attr(config, ("max_seq_len", "max_position_embeddings"), 4096)
        )
        num_heads = int(
            _config_attr(config, ("num_heads", "num_attention_heads"), 16)
        )
        return cls(
            vocab_size=int(_config_attr(config, ("vocab_size",))),
            hidden_dim=int(
                _config_attr(config, ("hidden_dim", "hidden_size"), 2048)
            ),
            intermediate_dim=int(
                _config_attr(
                    config,
                    ("intermediate_dim", "moe_intermediate_size", "intermediate_size"),
                    5632,
                )
            ),
            shared_expert_intermediate_dim=int(
                _config_attr(
                    config,
                    (
                        "shared_expert_intermediate_dim",
                        "shared_expert_intermediate_size",
                    ),
                    5632,
                )
            ),
            num_experts=int(
                _config_attr(config, ("num_experts", "num_local_experts"), 8)
            ),
            num_experts_per_token=int(
                _config_attr(
                    config, ("num_experts_per_token", "num_experts_per_tok"), 2
                )
            ),
            num_layers=int(
                _config_attr(config, ("num_layers", "num_hidden_layers"), 24)
            ),
            num_heads=num_heads,
            num_kv_heads=int(
                _config_attr(
                    config, ("num_kv_heads", "num_key_value_heads"), num_heads
                )
            ),
            head_dim=_config_attr(config, ("head_dim", "attention_head_dim")),
            max_seq_len=max_seq_len,
            sliding_window=int(
                _config_attr(config, ("sliding_window",), max_seq_len)
            ),
            layer_norm_eps=float(
                _config_attr(config, ("layer_norm_eps", "rms_norm_eps"), 1e-5)
            ),
            initializer_std=float(
                _config_attr(config, ("initializer_std", "initializer_range"), 0.02)
            ),
            qk_mult=float(_config_attr(config, ("qk_mult",), 1.0)),
            rope_theta=_rope_theta(config),
        ).validate()

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by "
                f"num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    def validate(self) -> "GrugMoeConfig":
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token >= self.num_experts:
            raise ValueError(
                "num_experts_per_token must be < num_experts for QB routing"
            )
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.inferred_head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.inferred_head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.sliding_window <= 1:
            raise ValueError("sliding_window must be greater than 1")
        return self


def _read_artifact_config(artifact_dir: Path) -> dict[str, Any]:
    config_path = artifact_dir / _ARTIFACT_CONFIG_FILE
    with config_path.open() as f:
        config = json.load(f)
    if config.get("model_type") != _ARTIFACT_MODEL_TYPE:
        raise ValueError(
            f"{config_path} must declare model_type={_ARTIFACT_MODEL_TYPE!r}"
        )
    return config


class _SafetensorsArtifactTensors(Mapping[str, np.ndarray]):

    def __init__(self, artifact_dir: Path,
                 weight_map: Mapping[str, str]) -> None:
        self._artifact_dir = artifact_dir
        self._weight_map = dict(weight_map)

    def __getitem__(self, name: str) -> np.ndarray:
        shard_name = self._weight_map[name]
        shard_path = self._artifact_dir / shard_name
        with safe_open(str(shard_path), framework="np") as f:
            return f.get_tensor(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._weight_map)

    def __len__(self) -> int:
        return len(self._weight_map)


def _read_safetensors_weight_map(index_path: Path) -> dict[str, str]:
    with index_path.open() as f:
        index = json.load(f)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path} must contain a non-empty weight_map")

    invalid = [
        (name, shard)
        for name, shard in weight_map.items()
        if not isinstance(name, str) or not isinstance(shard, str)
    ]
    if invalid:
        raise ValueError(
            f"{index_path} weight_map entries must map string tensor names to string shard files"
        )

    return dict(weight_map)


def _resolve_safetensors_shard_path(artifact_dir: Path,
                                    shard_name: str) -> Path:
    artifact_root = artifact_dir.resolve()
    shard_path = (artifact_dir / shard_name).resolve()
    try:
        shard_path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError(
            f"Safetensors shard {shard_name!r} must stay within {artifact_dir}"
        ) from exc
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
    return shard_path


def _validate_sharded_safetensors_index(
    artifact_dir: Path,
    weight_map: Mapping[str, str],
) -> None:
    shard_to_names: dict[str, set[str]] = {}
    for name, shard_name in weight_map.items():
        shard_to_names.setdefault(shard_name, set()).add(name)

    for shard_name, expected_names in shard_to_names.items():
        shard_path = _resolve_safetensors_shard_path(artifact_dir, shard_name)
        with safe_open(str(shard_path), framework="np") as f:
            actual_names = set(f.keys())

        missing = expected_names - actual_names
        unexpected = actual_names - expected_names
        if missing or unexpected:
            raise ValueError(
                "Safetensors shard does not match its index entry: "
                f"file={shard_name!r} missing={sorted(missing)} "
                f"unexpected={sorted(unexpected)}"
            )


def _load_artifact_tensors(artifact_dir: Path) -> Mapping[str, np.ndarray]:
    index_path = artifact_dir / _ARTIFACT_WEIGHTS_INDEX_FILE
    if index_path.exists():
        weight_map = _read_safetensors_weight_map(index_path)
        _validate_sharded_safetensors_index(artifact_dir, weight_map)
        return _SafetensorsArtifactTensors(artifact_dir, weight_map)

    weights_path = artifact_dir / _ARTIFACT_WEIGHTS_FILE
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Expected {weights_path} or {index_path} for GrugMoE weights")

    with safe_open(str(weights_path), framework="np") as f:
        weight_map = {name: _ARTIFACT_WEIGHTS_FILE for name in f.keys()}
    return _SafetensorsArtifactTensors(artifact_dir, weight_map)


def _validate_artifact_config(
    artifact_config: dict[str, Any],
    expected_config: GrugMoeConfig,
    *,
    expected_tie_word_embeddings: bool,
) -> None:
    loaded_config = GrugMoeConfig.from_hf_config(
        SimpleNamespace(**artifact_config)
    )
    mismatches = []
    for field in fields(GrugMoeConfig):
        if field.name == "head_dim":
            continue
        expected_value = getattr(expected_config, field.name)
        loaded_value = getattr(loaded_config, field.name)
        if loaded_value != expected_value:
            mismatches.append(f"{field.name}: artifact={loaded_value!r} model={expected_value!r}")
    if loaded_config.inferred_head_dim != expected_config.inferred_head_dim:
        mismatches.append(
            "head_dim: "
            f"artifact={loaded_config.inferred_head_dim!r} "
            f"model={expected_config.inferred_head_dim!r}"
        )

    tie_word_embeddings = bool(artifact_config.get("tie_word_embeddings", False))
    if tie_word_embeddings != expected_tie_word_embeddings:
        mismatches.append(
            "tie_word_embeddings: "
            f"artifact={tie_word_embeddings!r} model={expected_tie_word_embeddings!r}"
        )

    if mismatches:
        raise ValueError(
            "GrugMoE inference artifact config does not match the initialized model: "
            + "; ".join(mismatches)
        )


def _canonical_grugmoe_tensor_names(
    cfg: GrugMoeConfig,
    *,
    tie_word_embeddings: bool,
) -> frozenset[str]:
    names = {
        "model.embed_tokens.weight",
        "model.embed_norm.weight",
        "model.embed_gated_norm.down_proj.weight",
        "model.embed_gated_norm.up_proj.weight",
        "model.norm.weight",
        "model.final_gated_norm.down_proj.weight",
        "model.final_gated_norm.up_proj.weight",
    }
    if not tie_word_embeddings:
        names.add("lm_head.weight")

    for layer_index in range(cfg.num_layers):
        prefix = f"model.layers.{layer_index}"
        names.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.attn_gated_norm.down_proj.weight",
                f"{prefix}.attn_gated_norm.up_proj.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.attn_gate.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp_gated_norm.down_proj.weight",
                f"{prefix}.mlp_gated_norm.up_proj.weight",
                f"{prefix}.mlp.router.weight",
                f"{prefix}.mlp.router.bias",
                f"{prefix}.mlp.experts.gate_proj.weight",
                f"{prefix}.mlp.experts.up_proj.weight",
                f"{prefix}.mlp.experts.down_proj.weight",
            }
        )
        if cfg.shared_expert_intermediate_dim > 0:
            names.update(
                {
                    f"{prefix}.shared_expert.gate_proj.weight",
                    f"{prefix}.shared_expert.up_proj.weight",
                    f"{prefix}.shared_expert.down_proj.weight",
                }
            )

    return frozenset(names)


def _init_weight(
    rngs: nnx.Rngs,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    std: float,
) -> nnx.Param:
    value = std * jax.random.truncated_normal(
        rngs.params(), -3.0, 3.0, shape, dtype=jnp.float32
    )
    return nnx.Param(value.astype(dtype))


def _rms_norm(
    x: jax.Array,
    weight: Optional[jax.Array] = None,
    eps: float = 1e-6,
) -> jax.Array:
    dtype = x.dtype
    x_float = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    out = x_float * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight.astype(jnp.float32)
    return out.astype(dtype)


def _align_kv_heads(x: jax.Array, num_q_heads: int) -> jax.Array:
    num_kv_heads = x.shape[1]
    if num_q_heads == num_kv_heads:
        return x
    repeat = num_q_heads // num_kv_heads
    expanded = jnp.expand_dims(x, axis=2)
    expanded = jnp.broadcast_to(expanded, (x.shape[0], num_kv_heads, repeat, x.shape[2]))
    return expanded.reshape(x.shape[0], num_q_heads, x.shape[2])


def _apply_rotary(
    q: jax.Array,
    k: jax.Array,
    positions: jax.Array,
    *,
    head_dim: int,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim)
    )
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)[:, None, :]
    sin = jnp.sin(angles)[:, None, :]

    def apply(x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return apply(q), apply(k)


class GrugMoeRMSNorm(JaxModule):

    def __init__(self, dim: int, eps: float, dtype: jnp.dtype) -> None:
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        return _rms_norm(x, self.weight.value, self.eps)


class GrugMoeGatedNorm(JaxModule):

    def __init__(
        self,
        hidden_dim: int,
        initializer_std: float,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
    ) -> None:
        self.w_down = _init_weight(
            rng, (hidden_dim, _GATED_NORM_RANK), dtype, initializer_std
        )
        self.w_up = _init_weight(
            rng, (_GATED_NORM_RANK, hidden_dim), dtype, initializer_std
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate_hidden = jnp.einsum(
            "...d,dr->...r", x.astype(jnp.float32), self.w_down.value.astype(jnp.float32)
        )
        gate_hidden = jax.nn.silu(gate_hidden)
        gate = jax.nn.sigmoid(
            jnp.einsum("...r,rd->...d", gate_hidden, self.w_up.value.astype(jnp.float32))
        )
        return x * gate.astype(x.dtype)


class GrugMoeDenseMLP(JaxModule):

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        initializer_std: float,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
    ) -> None:
        self.w_gate = _init_weight(rng, (hidden_dim, intermediate_dim), dtype, initializer_std)
        self.w_up = _init_weight(rng, (hidden_dim, intermediate_dim), dtype, initializer_std)
        self.w_down = _init_weight(rng, (intermediate_dim, hidden_dim), dtype, initializer_std)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jnp.einsum("td,df->tf", x, self.w_gate.value)
        up = jnp.einsum("td,df->tf", x, self.w_up.value)
        return jnp.einsum("tf,fd->td", jax.nn.silu(gate) * up, self.w_down.value)


class GrugMoeMLP(JaxModule):
    """QB-routed MoE with sigmoid combine weights."""

    def __init__(self, cfg: GrugMoeConfig, dtype: jnp.dtype, rng: nnx.Rngs) -> None:
        self.cfg = cfg
        self.router = _init_weight(
            rng, (cfg.hidden_dim, cfg.num_experts), dtype, cfg.initializer_std
        )
        self.router_bias = nnx.Param(jnp.zeros((cfg.num_experts,), dtype=jnp.float32))
        self.w_gate_up = _init_weight(
            rng,
            (cfg.num_experts, cfg.hidden_dim, 2 * cfg.intermediate_dim),
            dtype,
            cfg.initializer_std,
        )
        self.w_down = _init_weight(
            rng,
            (cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim),
            dtype,
            cfg.initializer_std,
        )

    def route(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        router_logits = jnp.einsum(
            "td,de->te", x.astype(jnp.float32), self.router.value.astype(jnp.float32)
        )
        biased_logits = router_logits + jax.lax.stop_gradient(
            self.router_bias.value.astype(jnp.float32)
        )
        _, selected = jax.lax.top_k(
            biased_logits, self.cfg.num_experts_per_token + 1
        )
        selected = selected[:, : self.cfg.num_experts_per_token]
        unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
        combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
        return selected.astype(jnp.int32), combine_weights

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        selected, combine_weights = self.route(x)
        expert_w13 = self.w_gate_up.value[selected]
        w13_out = jnp.einsum("td,tkdf->tkf", x, expert_w13)
        gate, up = jnp.split(w13_out, [self.cfg.intermediate_dim], axis=-1)
        expert_w2 = self.w_down.value[selected]
        expert_out = jnp.einsum("tki,tkid->tkd", jax.nn.silu(gate) * up, expert_w2)
        out = jnp.sum(expert_out * combine_weights[..., None], axis=1)
        return out.astype(x.dtype), selected


class GrugMoeAttention(JaxModule):

    def __init__(self, cfg: GrugMoeConfig, dtype: jnp.dtype, rng: nnx.Rngs) -> None:
        self.cfg = cfg
        head_dim = cfg.inferred_head_dim
        self.w_q = _init_weight(
            rng, (cfg.hidden_dim, cfg.num_heads * head_dim), dtype, cfg.initializer_std
        )
        self.w_k = _init_weight(
            rng, (cfg.hidden_dim, cfg.num_kv_heads * head_dim), dtype, cfg.initializer_std
        )
        self.w_v = _init_weight(
            rng, (cfg.hidden_dim, cfg.num_kv_heads * head_dim), dtype, cfg.initializer_std
        )
        self.w_o = _init_weight(
            rng, (cfg.num_heads * head_dim, cfg.hidden_dim), dtype, cfg.initializer_std
        )
        self.attn_gate = nnx.Param(
            jnp.zeros((cfg.hidden_dim, cfg.num_heads), dtype=jnp.float32)
        )

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        sliding_window: int,
    ) -> jax.Array:
        head_dim = self.cfg.inferred_head_dim
        q = jnp.einsum("td,dh->th", x, self.w_q.value).reshape(
            x.shape[0], self.cfg.num_heads, head_dim
        )
        k = jnp.einsum("td,dh->th", x, self.w_k.value).reshape(
            x.shape[0], self.cfg.num_kv_heads, head_dim
        )
        v = jnp.einsum("td,dh->th", x, self.w_v.value).reshape(
            x.shape[0], self.cfg.num_kv_heads, head_dim
        )

        q = _rms_norm(q)
        k = _rms_norm(k)
        q, k = _apply_rotary(
            q, k, positions, head_dim=head_dim, theta=self.cfg.rope_theta
        )
        q = q * self.cfg.qk_mult
        k = _align_kv_heads(k, self.cfg.num_heads)
        v = _align_kv_heads(v, self.cfg.num_heads)

        scores = jnp.einsum("qhd,khd->hqk", q * (head_dim**-0.5), k)
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        mask = jnp.logical_and(
            k_pos <= q_pos,
            k_pos >= q_pos - (sliding_window - 1),
        )
        scores = jnp.where(mask[None, :, :], scores, jnp.array(-1e9, dtype=scores.dtype))
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        attn_out = jnp.einsum("hqk,khd->qhd", weights, v)

        dot = jnp.sum(attn_out * v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * v
        gate = 2 * jax.nn.sigmoid(
            jnp.einsum(
                "td,dn->tn", x.astype(jnp.float32), self.attn_gate.value.astype(jnp.float32)
            )
        )
        attn_out = gate[..., None].astype(attn_out.dtype) * attn_out
        attn_out = attn_out.reshape(x.shape[0], self.cfg.num_heads * head_dim)
        return jnp.einsum("th,hd->td", attn_out, self.w_o.value)


class GrugMoeDecoderLayer(JaxModule):

    def __init__(self, cfg: GrugMoeConfig, dtype: jnp.dtype, rng: nnx.Rngs) -> None:
        self.rms_attn = GrugMoeRMSNorm(cfg.hidden_dim, cfg.layer_norm_eps, dtype)
        self.attn_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim, cfg.initializer_std, dtype, rng
        )
        self.attn = GrugMoeAttention(cfg, dtype, rng)
        self.rms_mlp = GrugMoeRMSNorm(cfg.hidden_dim, cfg.layer_norm_eps, dtype)
        self.mlp_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim, cfg.initializer_std, dtype, rng
        )
        self.mlp = GrugMoeMLP(cfg, dtype, rng)
        self.shared = (
            GrugMoeDenseMLP(
                cfg.hidden_dim,
                cfg.shared_expert_intermediate_dim,
                cfg.initializer_std,
                dtype,
                rng,
            )
            if cfg.shared_expert_intermediate_dim > 0
            else None
        )

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        positions: jax.Array,
        sliding_window: int,
    ) -> tuple[Optional[jax.Array], jax.Array, jax.Array]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = x + self.attn(attn_in, positions, sliding_window)
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, expert_ids = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in)
        return kv_cache, x + mlp_out, expert_ids


class GrugMoeModel(JaxModule):

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
        prefix: str = "model",
    ) -> None:
        del mesh, prefix
        model_config = vllm_config.model_config
        hf_config = getattr(model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = model_config.hf_config
        self.config = GrugMoeConfig.from_hf_config(hf_config)
        self.dtype = model_config.dtype
        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (
            getattr(hf_config, "tie_word_embeddings", False) and self.is_last_rank
        ):
            self.token_embed = _init_weight(
                rng,
                (self.config.vocab_size, self.config.hidden_dim),
                self.dtype,
                self.config.initializer_std,
            )
        else:
            self.token_embed = PPMissingLayer()

        self.embed_norm = GrugMoeRMSNorm(
            self.config.hidden_dim, self.config.layer_norm_eps, self.dtype
        )
        self.embed_gated_norm = GrugMoeGatedNorm(
            self.config.hidden_dim, self.config.initializer_std, self.dtype, rng
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_layers,
            lambda _layer_index: GrugMoeDecoderLayer(self.config, self.dtype, rng),
        )
        self.final_norm = (
            GrugMoeRMSNorm(
                self.config.hidden_dim, self.config.layer_norm_eps, self.dtype
            )
            if self.is_last_rank
            else PPMissingLayer()
        )
        self.final_gated_norm = (
            GrugMoeGatedNorm(
                self.config.hidden_dim,
                self.config.initializer_std,
                self.dtype,
                rng,
            )
            if self.is_last_rank
            else PPMissingLayer()
        )

    def embed_input_ids(self, input_ids: jax.Array) -> jax.Array:
        if isinstance(self.token_embed, PPMissingLayer):
            raise ValueError("token embeddings are not present on this pipeline rank")
        return self.token_embed.value[input_ids]

    def __call__(
        self,
        kv_caches: list[Optional[jax.Array]],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> tuple[list[Optional[jax.Array]], jax.Array, Optional[jax.Array]]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            x = self.embed_input_ids(input_ids)
        else:
            x = inputs_embeds

        x = self.embed_gated_norm(self.embed_norm(x))
        positions = attention_metadata.input_positions
        if positions.ndim != 1:
            positions = jnp.reshape(positions, (-1,))

        short_window = self.config.sliding_window // 2
        new_kv_caches: list[Optional[jax.Array]] = []
        all_expert_ids = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if i < len(kv_caches) else None
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_cache)
                continue
            layer_window = self.config.sliding_window if i % 4 == 3 else short_window
            kv_cache, x, expert_ids = layer(kv_cache, x, positions, layer_window)
            new_kv_caches.append(kv_cache)
            all_expert_ids.append(expert_ids)

        if self.is_last_rank:
            x = self.final_gated_norm(self.final_norm(x))

        stacked_expert_ids = jnp.stack(all_expert_ids, axis=0) if all_expert_ids else None
        return new_kv_caches, x, stacked_expert_ids


class GrugMoeForCausalLM(JaxModule):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh
        self.model = GrugMoeModel(vllm_config, rng, mesh, prefix="model")

        hf_config = vllm_config.model_config.hf_config
        self.tie_word_embeddings = bool(
            getattr(hf_config, "tie_word_embeddings", False)
        )
        if not self.tie_word_embeddings:
            if self.model.is_last_rank:
                self.lm_head = JaxLmHead(
                    hidden_size=self.model.config.hidden_dim,
                    vocab_size=self.model.config.vocab_size,
                    dtype=vllm_config.model_config.dtype,
                    param_dtype=vllm_config.model_config.dtype,
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

    def __call__(
        self,
        kv_caches: list[Optional[jax.Array]],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> tuple[
        list[Optional[jax.Array]],
        jax.Array | JaxIntermediateTensors,
        list[jax.Array],
        Optional[jax.Array],
    ]:
        del args
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        kv_caches, hidden_states, expert_ids = self.model(
            kv_caches, input_ids, attention_metadata, inputs_embeds
        )
        if not is_last_rank:
            hidden_states = JaxIntermediateTensors(
                tensors={"hidden_states": hidden_states}
            )
        return kv_caches, hidden_states, [], expert_ids

    def embed_input_ids(
        self,
        input_ids: jax.Array,
        mm_embeds: Optional[jax.Array] = None,
        is_multimodal: Optional[bool] = None,
    ) -> jax.Array:
        del mm_embeds, is_multimodal
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, "lm_head") and not isinstance(self.lm_head, PPMissingLayer):
            return self.lm_head(hidden_states)
        if isinstance(self.model.token_embed, PPMissingLayer):
            raise ValueError("token embeddings are not present on this pipeline rank")
        return jnp.einsum("td,vd->tv", hidden_states, self.model.token_embed.value)

    def _tensor(
        self,
        tensors: Mapping[str, np.ndarray],
        consumed: set[str],
        name: str,
    ) -> jax.Array:
        consumed.add(name)
        return jnp.asarray(tensors[name])

    def _assign_param(
        self,
        param: nnx.Param,
        value: jax.Array,
        *,
        tensor_name: str,
    ) -> None:
        value = jnp.asarray(value, dtype=param.value.dtype)
        if value.shape != param.value.shape:
            raise ValueError(
                f"Loaded shape for {tensor_name}: {value.shape} does not match "
                f"model shape: {param.value.shape}"
            )
        param.value = value

    def _assign_linear_param(
        self,
        param: nnx.Param,
        tensors: Mapping[str, np.ndarray],
        consumed: set[str],
        name: str,
    ) -> None:
        self._assign_param(
            param,
            jnp.swapaxes(self._tensor(tensors, consumed, name), -1, -2),
            tensor_name=name,
        )

    def load_inference_artifact(
        self,
        artifact_dir: str | Path,
    ) -> GrugMoeArtifactLoadReport:
        """Load the canonical GrugMoE inference artifact.

        The artifact is an accelerator-agnostic directory with `config.json`
        and either `model.safetensors` or the standard HuggingFace sharded
        safetensors layout: `model.safetensors.index.json` plus
        `model-*-of-*.safetensors`. Tensor names follow a stable HF/vLLM-style
        dotted schema, while linear tensors use the usual checkpoint
        orientation with output features before input features.
        """
        if not (self.model.is_first_rank and self.model.is_last_rank):
            raise NotImplementedError(
                "GrugMoE inference artifact loading currently supports a full "
                "model on a single pipeline rank."
            )

        artifact_path = Path(artifact_dir)
        artifact_config = _read_artifact_config(artifact_path)
        _validate_artifact_config(
            artifact_config,
            self.model.config,
            expected_tie_word_embeddings=self.tie_word_embeddings,
        )

        tensors = _load_artifact_tensors(artifact_path)
        expected = _canonical_grugmoe_tensor_names(
            self.model.config,
            tie_word_embeddings=self.tie_word_embeddings,
        )
        loaded = set(tensors)
        missing = expected - loaded
        unexpected = loaded - expected
        if missing or unexpected:
            raise ValueError(
                "GrugMoE inference artifact tensor set mismatch: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )

        consumed: set[str] = set()
        model = self.model
        self._assign_param(
            model.token_embed,
            self._tensor(tensors, consumed, "model.embed_tokens.weight"),
            tensor_name="model.embed_tokens.weight",
        )
        self._assign_param(
            model.embed_norm.weight,
            self._tensor(tensors, consumed, "model.embed_norm.weight"),
            tensor_name="model.embed_norm.weight",
        )
        self._assign_linear_param(
            model.embed_gated_norm.w_down,
            tensors,
            consumed,
            "model.embed_gated_norm.down_proj.weight",
        )
        self._assign_linear_param(
            model.embed_gated_norm.w_up,
            tensors,
            consumed,
            "model.embed_gated_norm.up_proj.weight",
        )

        for layer_index, layer in enumerate(model.layers):
            if isinstance(layer, PPMissingLayer):
                raise NotImplementedError(
                    "GrugMoE inference artifact loading currently supports a "
                    "full model on a single pipeline rank."
                )

            prefix = f"model.layers.{layer_index}"
            self._assign_param(
                layer.rms_attn.weight,
                self._tensor(tensors, consumed, f"{prefix}.input_layernorm.weight"),
                tensor_name=f"{prefix}.input_layernorm.weight",
            )
            self._assign_linear_param(
                layer.attn_gated_norm.w_down,
                tensors,
                consumed,
                f"{prefix}.attn_gated_norm.down_proj.weight",
            )
            self._assign_linear_param(
                layer.attn_gated_norm.w_up,
                tensors,
                consumed,
                f"{prefix}.attn_gated_norm.up_proj.weight",
            )
            self._assign_linear_param(
                layer.attn.w_q,
                tensors,
                consumed,
                f"{prefix}.self_attn.q_proj.weight",
            )
            self._assign_linear_param(
                layer.attn.w_k,
                tensors,
                consumed,
                f"{prefix}.self_attn.k_proj.weight",
            )
            self._assign_linear_param(
                layer.attn.w_v,
                tensors,
                consumed,
                f"{prefix}.self_attn.v_proj.weight",
            )
            self._assign_linear_param(
                layer.attn.w_o,
                tensors,
                consumed,
                f"{prefix}.self_attn.o_proj.weight",
            )
            self._assign_linear_param(
                layer.attn.attn_gate,
                tensors,
                consumed,
                f"{prefix}.self_attn.attn_gate.weight",
            )
            self._assign_param(
                layer.rms_mlp.weight,
                self._tensor(
                    tensors,
                    consumed,
                    f"{prefix}.post_attention_layernorm.weight",
                ),
                tensor_name=f"{prefix}.post_attention_layernorm.weight",
            )
            self._assign_linear_param(
                layer.mlp_gated_norm.w_down,
                tensors,
                consumed,
                f"{prefix}.mlp_gated_norm.down_proj.weight",
            )
            self._assign_linear_param(
                layer.mlp_gated_norm.w_up,
                tensors,
                consumed,
                f"{prefix}.mlp_gated_norm.up_proj.weight",
            )
            self._assign_linear_param(
                layer.mlp.router,
                tensors,
                consumed,
                f"{prefix}.mlp.router.weight",
            )
            self._assign_param(
                layer.mlp.router_bias,
                self._tensor(tensors, consumed, f"{prefix}.mlp.router.bias"),
                tensor_name=f"{prefix}.mlp.router.bias",
            )
            gate = jnp.swapaxes(
                self._tensor(
                    tensors,
                    consumed,
                    f"{prefix}.mlp.experts.gate_proj.weight",
                ),
                -1,
                -2,
            )
            up = jnp.swapaxes(
                self._tensor(
                    tensors,
                    consumed,
                    f"{prefix}.mlp.experts.up_proj.weight",
                ),
                -1,
                -2,
            )
            self._assign_param(
                layer.mlp.w_gate_up,
                jnp.concatenate([gate, up], axis=-1),
                tensor_name=f"{prefix}.mlp.experts.gate_proj.weight",
            )
            self._assign_linear_param(
                layer.mlp.w_down,
                tensors,
                consumed,
                f"{prefix}.mlp.experts.down_proj.weight",
            )

            if self.model.config.shared_expert_intermediate_dim > 0:
                if layer.shared is None:
                    raise ValueError(
                        f"{prefix} is missing the shared expert expected by the config"
                    )
                self._assign_linear_param(
                    layer.shared.w_gate,
                    tensors,
                    consumed,
                    f"{prefix}.shared_expert.gate_proj.weight",
                )
                self._assign_linear_param(
                    layer.shared.w_up,
                    tensors,
                    consumed,
                    f"{prefix}.shared_expert.up_proj.weight",
                )
                self._assign_linear_param(
                    layer.shared.w_down,
                    tensors,
                    consumed,
                    f"{prefix}.shared_expert.down_proj.weight",
                )

        self._assign_param(
            model.final_norm.weight,
            self._tensor(tensors, consumed, "model.norm.weight"),
            tensor_name="model.norm.weight",
        )
        self._assign_linear_param(
            model.final_gated_norm.w_down,
            tensors,
            consumed,
            "model.final_gated_norm.down_proj.weight",
        )
        self._assign_linear_param(
            model.final_gated_norm.w_up,
            tensors,
            consumed,
            "model.final_gated_norm.up_proj.weight",
        )
        if not self.tie_word_embeddings:
            if isinstance(self.lm_head, PPMissingLayer):
                raise ValueError("lm_head is not present on this pipeline rank")
            self._assign_linear_param(
                self.lm_head.weight,
                tensors,
                consumed,
                "lm_head.weight",
            )

        unconsumed_expected = expected - consumed
        if unconsumed_expected:
            raise ValueError(
                "GrugMoE inference artifact tensors were expected but not consumed: "
                f"{sorted(unconsumed_expected)}"
            )

        return GrugMoeArtifactLoadReport(
            consumed=frozenset(consumed),
            missing=frozenset(),
            unexpected=frozenset(),
        )

    def load_weights(self, rng: jax.Array) -> GrugMoeArtifactLoadReport:
        del rng
        return self.load_inference_artifact(self.vllm_config.model_config.model)


__all__ = [
    "GrugMoeAttention",
    "GrugMoeArtifactLoadReport",
    "GrugMoeConfig",
    "GrugMoeDecoderLayer",
    "GrugMoeDenseMLP",
    "GrugMoeForCausalLM",
    "GrugMoeGatedNorm",
    "GrugMoeHfConfig",
    "GrugMoeMLP",
    "GrugMoeModel",
    "GrugMoeRMSNorm",
]
