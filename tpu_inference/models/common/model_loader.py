# Copyright 2025 Google LLC
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

import functools
from dataclasses import dataclass
from typing import Any, Optional

import jax
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.runai_streamer_loader import \
    RunaiModelStreamerLoader
from vllm.utils.func_utils import supports_kw

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.qwix.qwix_utils import (
    apply_qwix_on_abstract_model, apply_qwix_quantization,
    load_random_weights_into_qwix_abstract_model)
from tpu_inference.utils import to_jax_dtype, to_torch_dtype

logger = init_logger(__name__)

_MODEL_REGISTRY = {}

# List of architectures that are preferred to use  "vllm" implementation over
# "flax_nnx" implementation due to various factors such as performance.
_VLLM_PREFERRED_ARCHITECTURES: frozenset[str] = frozenset(
    {"GptOssForCausalLM"})

# Architectures that need abstract dummy bootstrap for fast startup.
# Only includes models that do NOT implement LoadableWithIterator and
# therefore fall into the expensive concrete random-init branch.
_ABSTRACT_BOOTSTRAP_ARCHITECTURES: frozenset[str] = frozenset({
    "LlamaForCausalLM",
    "MistralForCausalLM",  # vLLM alias for LlamaForCausalLM
})

# Fallback mapping from HuggingFace model_type to JAX registry key.
# Used when vLLM rewrites hf_config.architectures (e.g. LlamaForCausalLM →
# MistralForCausalLM) before tpu-inference sees them.
_MODEL_TYPE_TO_REGISTRY_KEY: dict[str, str] = {
    "llama": "LlamaForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
}

# Architectures that prefer_jax_for_bootstrap is allowed to reroute from
# "vllm" to "flax_nnx". This is NOT the full JAX registry — only models
# we have explicitly vetted for bootstrap-mode routing.
# Empty on v0.13.2 because Qwen3MoeForCausalLM is not in the registry.
_BOOTSTRAP_JAX_ROUTING_ALLOWLIST: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TpuBootstrapConfig:
    """Typed config for TPU fast bootstrap, extracted from additional_config.

    Uses vllm_config.additional_config (not model_loader_extra_config)
    because vLLM's dummy loader rejects any model_loader_extra_config.
    Pass via CLI: --additional-config '{"tpu_bootstrap": {...}}'
    """
    model_bootstrap: str = "default"
    prefer_jax_for_bootstrap: bool = False
    weight_loader: str = "default"

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "TpuBootstrapConfig":
        extra = getattr(vllm_config, "additional_config", None) or {}
        raw = extra.get("tpu_bootstrap", {})
        if not isinstance(raw, dict):
            return cls()
        model_bootstrap = raw.get("model_bootstrap", "default")
        if model_bootstrap not in ("default", "abstract_dummy", "abstract_load"):
            raise ValueError(
                f"Invalid tpu_bootstrap.model_bootstrap: {model_bootstrap!r}. "
                "Valid options: 'default', 'abstract_dummy', 'abstract_load'")
        weight_loader = raw.get("weight_loader", "default")
        if weight_loader not in ("default", "fsspec_streamer"):
            raise ValueError(
                f"Invalid tpu_bootstrap.weight_loader: {weight_loader!r}. "
                "Valid options: 'default', 'fsspec_streamer'")
        return cls(
            model_bootstrap=model_bootstrap,
            prefer_jax_for_bootstrap=bool(
                raw.get("prefer_jax_for_bootstrap", False)),
            weight_loader=weight_loader,
        )


def _resolved_bootstrap_mode(vllm_config: VllmConfig,
                             model_class: Any) -> str:
    """Resolve which bootstrap mode to use.

    Returns "default", "abstract_dummy", or "abstract_load".

    Raises ValueError if a non-default mode was explicitly requested but
    cannot be honored (unsupported arch, quantization active, wrong
    load_format). Only "default" silently returns "default".
    """
    bootstrap = TpuBootstrapConfig.from_vllm_config(vllm_config)
    mode = bootstrap.model_bootstrap
    if mode == "default":
        return "default"
    # Non-default mode was explicitly requested — fail loudly if invalid
    arch = model_class.__name__
    if arch not in _ABSTRACT_BOOTSTRAP_ARCHITECTURES:
        raise ValueError(
            f"{mode} is not supported for architecture {arch!r}")
    if apply_qwix_on_abstract_model(vllm_config):
        raise ValueError(
            f"{mode} is incompatible with Qwix abstract quantization")
    if getattr(vllm_config.model_config.hf_config, "quantization_config",
               None):
        raise ValueError(
            f"{mode} is incompatible with hf quantization_config")
    if getattr(vllm_config.model_config, "quantization", None):
        raise ValueError(
            f"{mode} is incompatible with TPU quantization")
    # Validate mode vs load_format
    if mode == "abstract_dummy" and vllm_config.load_config.load_format != "dummy":
        raise ValueError("abstract_dummy requires load_format='dummy'")
    if mode == "abstract_load" and vllm_config.load_config.load_format == "dummy":
        raise ValueError(
            "abstract_load requires a real load_format, not 'dummy'")
    return mode


def _build_abstract_model_and_load_weights(
    create_abstract_model,
    create_jit_model,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:
    """Create abstract model, load real weights, JIT compile."""
    abstract_model_fn = create_abstract_model
    if should_apply_qwix := apply_qwix_on_abstract_model(vllm_config):
        abstract_model_fn = apply_qwix_quantization(
            vllm_config,
            create_abstract_model,
            rng,
            mesh,
            apply_to_abstract_model=True)

    bootstrap = TpuBootstrapConfig.from_vllm_config(vllm_config)

    with mesh:
        model = nnx.eval_shape(abstract_model_fn)
        if bootstrap.weight_loader == "fsspec_streamer":
            from tpu_inference.models.jax.streaming_weights import (
                fsspec_weights_iterator)
            model_path = vllm_config.model_config.model
            if getattr(vllm_config.model_config, "model_weights", ""):
                model_path = vllm_config.model_config.model_weights
            weights_iterator = fsspec_weights_iterator(model_path)
            vllm_config.model_config.model_weights_iterator = weights_iterator
            try:
                model.load_weights(rng)
            finally:
                if hasattr(vllm_config.model_config,
                           "model_weights_iterator"):
                    delattr(vllm_config.model_config,
                            "model_weights_iterator")
        elif isinstance(
                (loader := get_model_loader(vllm_config.load_config)),
                RunaiModelStreamerLoader):
            model_weights = vllm_config.model_config.model
            if getattr(vllm_config.model_config, "model_weights", ""):
                model_weights = vllm_config.model_config.model_weights
            weights_iterator = loader._get_weights_iterator(
                model_weights, vllm_config.model_config.revision)
            vllm_config.model_config.model_weights_iterator = weights_iterator
            try:
                model.load_weights(rng)
            finally:
                if hasattr(vllm_config.model_config,
                           "model_weights_iterator"):
                    delattr(vllm_config.model_config,
                            "model_weights_iterator")
        else:
            model.load_weights(rng)
        jit_model = create_jit_model(
            model, use_qwix_on_abstract_model=should_apply_qwix)
    return jit_model


class UnsupportedArchitectureError(ValueError):
    """Raised when a model architecture is not supported in the registry."""
    pass


def _get_model_architecture(config: PretrainedConfig) -> nnx.Module:
    # NOTE: Use inline imports here, otherwise the normal imports
    # would cause JAX init failure when using multi hosts with Ray.

    from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3
    from tpu_inference.models.jax.gpt_oss import GptOss
    from tpu_inference.models.jax.llama3 import LlamaForCausalLM
    from tpu_inference.models.jax.llama4 import Llama4ForCausalLM
    from tpu_inference.models.jax.llama_eagle3 import EagleLlama3ForCausalLM
    from tpu_inference.models.jax.llama_guard_4 import LlamaGuard4ForCausalLM
    from tpu_inference.models.jax.qwen2_5_vl import \
        Qwen2_5_VLForConditionalGeneration
    from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM
    _MODEL_REGISTRY["Llama4ForCausalLM"] = Llama4ForCausalLM
    _MODEL_REGISTRY["DeepseekV3ForCausalLM"] = DeepSeekV3
    _MODEL_REGISTRY["LlamaForCausalLM"] = LlamaForCausalLM
    # vLLM remaps LlamaForCausalLM → MistralForCausalLM (they share the
    # same architecture).  Register the alias so our JAX path is reached.
    _MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM
    _MODEL_REGISTRY["Llama4ForConditionalGeneration"] = LlamaGuard4ForCausalLM
    _MODEL_REGISTRY["Qwen3ForCausalLM"] = Qwen3ForCausalLM
    _MODEL_REGISTRY[
        "Qwen2_5_VLForConditionalGeneration"] = Qwen2_5_VLForConditionalGeneration
    _MODEL_REGISTRY["Eagle3LlamaForCausalLM"] = EagleLlama3ForCausalLM
    _MODEL_REGISTRY["GptOssForCausalLM"] = GptOss

    architectures = getattr(config, "architectures", [])
    model_type = getattr(config, "model_type", "unknown")
    logger.info(
        "Architecture lookup: hf_config.architectures=%s, "
        "hf_config.model_type=%s, JAX registry keys=%s",
        architectures, model_type, list(_MODEL_REGISTRY.keys()))
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]

    # Fallback: vLLM may remap architectures (e.g. LlamaForCausalLM →
    # MistralForCausalLM) before we see hf_config.  Use model_type as
    # ground truth to find the correct JAX class.
    if model_type in _MODEL_TYPE_TO_REGISTRY_KEY:
        fallback_key = _MODEL_TYPE_TO_REGISTRY_KEY[model_type]
        if fallback_key in _MODEL_REGISTRY:
            logger.info(
                "Architecture fallback: model_type=%r mapped to %r "
                "(original architectures=%s were not in JAX registry)",
                model_type, fallback_key, architectures)
            return _MODEL_REGISTRY[fallback_key]

    raise UnsupportedArchitectureError(
        f"Model architectures {architectures} (model_type={model_type}) not "
        "registered in tpu-inference. Falling back to vLLM-native "
        f"Pytorch definition. JAX-native architectures: {list(_MODEL_REGISTRY.keys())}"
    )


def _get_nnx_model(
    model_class: Any,
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nnx.Module:

    def create_abstract_model() -> nnx.Module:
        """
        Helper class to create an abstract model for `nnx.eval_shape`.

        Returns:
            An abstract model function.
        """
        return model_class(vllm_config, rng, mesh)

    @nnx.jit(donate_argnums=(0, ),
             static_argnames=('use_qwix_on_abstract_model', ))
    def create_jit_model(
            model: nnx.Module,
            use_qwix_on_abstract_model: bool = False) -> nnx.Module:
        """
        Create a jit model.

        Args:
            model: The model to jit.
            use_qwix_on_abstract_model: Whether to apply Qwix on the abstract model.

        Returns:
            The jitted model.
        """
        state = nnx.state(model)
        nnx.update(model, state)
        if not use_qwix_on_abstract_model:
            # NOTE: if Qwix is not configured, this will be a no-op
            model = apply_qwix_quantization(vllm_config,
                                            model,
                                            rng,
                                            mesh,
                                            apply_to_abstract_model=False)
        return model

    mode = _resolved_bootstrap_mode(vllm_config, model_class)

    if mode == "abstract_dummy":
        # RL path: abstract model only, caller injects weights via _sync_weights()
        logger.info(
            "Abstract dummy bootstrap for %s (RL injection mode)",
            model_class.__name__)
        with mesh:
            model = nnx.eval_shape(create_abstract_model)
        return model

    if mode == "abstract_load":
        # Fast serve path: abstract model + real weight loading
        logger.info("Abstract load bootstrap for %s", model_class.__name__)
        return _build_abstract_model_and_load_weights(
            create_abstract_model, create_jit_model, vllm_config, rng, mesh)

    # Default paths (mode == "default")
    if vllm_config.load_config.load_format == "dummy":
        # Create a sharded model with random inited weights.
        # TODO: currently Qwen2ForCausalLM is using legacy model implementation
        # will merge the random init logic when all model are migrated to new model implementation

        # Handle the case where we want to load in random weights to a Qwix-quantized model.  Here, we
        # need to run an abstract pass for Qwix first and then load in the random weights.
        if apply_qwix_on_abstract_model(vllm_config):
            abstract_model_fn = apply_qwix_quantization(
                vllm_config,
                create_abstract_model,
                rng,
                mesh,
                apply_to_abstract_model=True)

            model = nnx.eval_shape(abstract_model_fn)
            quantization_config = vllm_config.model_config.hf_config.quantization_config if hasattr(
                vllm_config.model_config.hf_config,
                "quantization_config") else {}
            load_random_weights_into_qwix_abstract_model(
                rng, model, mesh, quantization_config)
            with mesh:
                jit_model = create_jit_model(model,
                                             use_qwix_on_abstract_model=True)
            return jit_model

        @nnx.jit
        def create_sharded_model():
            model = model_class(vllm_config, rng, mesh)
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            # NOTE: we don't support quantization for the old Qwen2ForCausalLM implementation
            return model

        with mesh:
            jit_model = create_sharded_model()
            # In this case, we are applying Qwix quantization to the true, concrete model
            jit_model = apply_qwix_quantization(vllm_config,
                                                jit_model,
                                                rng,
                                                mesh,
                                                apply_to_abstract_model=False)
            if hasattr(jit_model, 'initialize_cache'):
                jit_model.initialize_cache()
    else:
        return _build_abstract_model_and_load_weights(
            create_abstract_model, create_jit_model, vllm_config, rng, mesh)
    return jit_model


# TODO(pooyam): We need to refactor this. This is returning a bunch of functions that do not work with all models and this is not very easy to see from the code.
def get_flax_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
    is_draft_model: bool = False,
) -> nnx.Module:
    model_dtype = to_jax_dtype(vllm_config.model_config.dtype)
    vllm_config.model_config.dtype = model_dtype

    if is_draft_model:
        model_class = _get_model_architecture(
            vllm_config.speculative_config.draft_model_config.hf_config)
    else:
        model_class = _get_model_architecture(
            vllm_config.model_config.hf_config)
    jit_model = _get_nnx_model(model_class, vllm_config, rng, mesh)
    kv_cache_sharding = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD))
    hidden_states_sharding = NamedSharding(mesh,
                                           PartitionSpec(
                                               ShardingAxisName.ATTN_DATA,
                                               None))  # (T, D)

    # For performance consideration, refer to:
    # https://flax.readthedocs.io/en/latest/guides/performance.html
    graphdef, state = nnx.split(jit_model)

    @functools.partial(
        jax.jit,
        out_shardings=(
            kv_cache_sharding,
            hidden_states_sharding,
            hidden_states_sharding,  # aux hidden states
        ),
        donate_argnums=2,  # 0 is graphdef, 1 is state, 2 is kv_cache
        static_argnums=(
            7, 10, 11
        ),  #7 is layer_name_to_kvcache_index, 10 is is_first_rank, 11 is is_last_rank
    )
    def run_model(graphdef, state, *args):
        model = nnx.merge(graphdef, state)
        return model(*args)

    logits_sharding = NamedSharding(
        mesh,
        PartitionSpec(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR))

    @functools.partial(
        jax.jit,
        out_shardings=(logits_sharding),
    )
    def run_compute_logits(graphdef, state, *args):
        model = nnx.merge(graphdef, state)
        hidden_state, *_ = args
        return model.compute_logits(hidden_state)

    # Multi-modal support only
    # This function calculates the image token's embeddings by VIT
    def run_embed_multimodal(graphdef, state, image_grid_thw, **kwargs):
        model = nnx.merge(graphdef, state)
        return model.embed_multimodal(image_grid_thw, **kwargs)

    embed_sharding = NamedSharding(mesh, PartitionSpec(None))
    # This function will calculates the embeddings of input texts and then merge with the image embeddings
    @functools.partial(
        jax.jit,
        out_shardings=(embed_sharding),
    )
    def run_embed_input_ids(graphdef, state, *args, **kwargs):
        model = nnx.merge(graphdef, state)
        return model.embed_input_ids(*args, **kwargs)

    # For models that want to work with EAGLE-3 speculative decoding
    @functools.partial(
        jax.jit,
        out_shardings=(logits_sharding),
    )
    def combine_hidden_states(graphdef, state, hidden_states):
        model = nnx.merge(graphdef, state)
        return model.combine_hidden_states(hidden_states)

    model = nnx.merge(graphdef, state)
    precompile_vision_encoder_fn = getattr(model, "precompile_vision_encoder",
                                           None)
    model_fn = functools.partial(run_model, graphdef)
    compute_logits_fn = functools.partial(run_compute_logits, graphdef)
    embed_multimodal_fn = functools.partial(run_embed_multimodal, graphdef)
    embed_input_ids_fn = functools.partial(run_embed_input_ids, graphdef)
    lora_manager, model = None, None
    combine_hidden_states_fn = functools.partial(combine_hidden_states,
                                                 graphdef)

    get_mrope_input_positions_fn = None if not hasattr(
        jit_model,
        "get_mrope_input_positions") else jit_model.get_mrope_input_positions

    multimodal_fns = {
        "precompile_vision_encoder_fn": precompile_vision_encoder_fn,
        "embed_multimodal_fn": embed_multimodal_fn,
        "embed_input_ids_fn": embed_input_ids_fn,
        "get_mrope_input_positions_fn": get_mrope_input_positions_fn,
    }

    return model_fn, compute_logits_fn, combine_hidden_states_fn, multimodal_fns, state, lora_manager, model


def get_vllm_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
):
    model_dtype = to_torch_dtype(vllm_config.model_config.dtype)
    vllm_config.model_config.dtype = model_dtype
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper

    model = VllmModelWrapper(
        vllm_config=vllm_config,
        rng=rng,
        mesh=mesh,
    )
    params, lora_manager = model.load_weights()

    jit_model = model.jit_step_func()
    compute_logits_fn = model.jit_compute_logits_func()
    # the model needs to be returned because lora weights are neither torch.nn.parameter nor torch.nn.buffer. After we load the lora weights and set it to the torch.nn.Module, we can shard it and move it to TPU.
    combine_hidden_states_fn = None
    return jit_model, compute_logits_fn, combine_hidden_states_fn, None, params, lora_manager, model


def get_model(
    vllm_config: VllmConfig,
    rng: jax.Array,
    mesh: Mesh,
    is_draft_model: bool = False,
) -> Any:
    impl = envs.MODEL_IMPL_TYPE
    hf_config = vllm_config.model_config.hf_config
    architectures = getattr(hf_config, "architectures", [])
    model_type = getattr(hf_config, "model_type", "unknown")
    model_path = getattr(vllm_config.model_config, "model", "unknown")
    logger.info(
        "Loading model with MODEL_IMPL_TYPE=%s | model=%s | "
        "hf_config.architectures=%s | hf_config.model_type=%s",
        impl, model_path, architectures, model_type)

    if impl == "auto":
        # Resolve "auto" based on architecture
        assert len(architectures) == 1, (
            f"Expected exactly one architecture, got {len(architectures)}: "
            f"{architectures}")
        arch = architectures[0]

        # When fast bootstrap is requested, allow specific architectures that
        # are normally vllm-preferred to route through flax_nnx instead.
        bootstrap = TpuBootstrapConfig.from_vllm_config(vllm_config)
        if (bootstrap.prefer_jax_for_bootstrap
                and arch in _BOOTSTRAP_JAX_ROUTING_ALLOWLIST):
            logger.info(
                "Bootstrap-aware routing: preferring flax_nnx for %s "
                "(overriding _VLLM_PREFERRED_ARCHITECTURES)", arch)
            impl = "flax_nnx"
        else:
            impl = "vllm" if arch in _VLLM_PREFERRED_ARCHITECTURES else "flax_nnx"
        logger.info(f"Resolved MODEL_IMPL_TYPE 'auto' to '{impl}'")

    match impl:
        case "flax_nnx":
            if vllm_config.parallel_config.pipeline_parallel_size > 1:
                logger.warning(
                    "PP is not fully supported on Jax flax_nnx models yet, fallback to vllm models."
                )
                return get_vllm_model(vllm_config, rng, mesh)
            try:
                # Try to load the flax model first
                return get_flax_model(vllm_config, rng, mesh, is_draft_model)
            except UnsupportedArchitectureError as e:
                # Convert the error message to a string to check its contents
                error_msg = str(e)

                logger.warning(error_msg)

                # Fall back to the vLLM model and updating the dtype accordingly
                return get_vllm_model(vllm_config, rng, mesh)
        case "vllm":
            return get_vllm_model(vllm_config, rng, mesh)
        case _:
            raise NotImplementedError(f"Unsupported MODEL_IMPL_TYPE: {impl}")


def _validate_model_interface(model: Any) -> None:
    """Validates that the model class has the required methods and signatures.

    A valid model must have:
    - An __init__ method that accepts a 'vllm_config' keyword argument.
    - A __call__ method that accepts 'kv_caches', 'input_ids', and
      'attention_metadata' keyword arguments.

    Args:
        model: The model class to validate.

    Raises:
        TypeError: If the model does not meet the interface requirements.
    """
    # Check for __init__ with vllm_config
    model_init = getattr(model, "__init__", None)
    if not callable(model_init):
        raise TypeError(
            f"Model {model.__name__} must have an __init__ method.")

    if not supports_kw(model_init, "vllm_config"):
        raise TypeError(
            f"Model {model.__name__} __init__ method must accept a "
            "'vllm_config' keyword argument.")

    # Check for __call__ with required arguments
    model_call = getattr(model, "__call__", None)
    # A class object is always callable (it produces an instance).
    # We need to check if the class _explicitly_ defines a __call__ method for its
    # instance, which is different from `type.__call__`.
    has_defined_call = False
    if isinstance(model, type):
        if any("__call__" in C.__dict__ for C in model.__mro__):
            has_defined_call = True
    elif callable(model_call):
        # For an instance, a simple callable check is sufficient.
        has_defined_call = True

    if not has_defined_call:
        raise TypeError(f"Model {model.__name__} must have a __call__ method.")

    required_call_args = ("kv_caches", "input_ids", "attention_metadata")
    missing_args = tuple(arg for arg in required_call_args
                         if not supports_kw(model_call, arg))

    if missing_args:
        raise TypeError(
            f"Model {model.__name__} __call__ method is missing required "
            f"keyword arguments: {missing_args}")


def register_model(arch: str, model: Any) -> None:
    """
    Registers a model class for a given architecture name.

    This function registers the model with both the tpu_inference registry
    and the vLLM registry. For vLLM, it creates a compatible wrapper
    around the JAX model.

    Args:
        arch: The name of the architecture (e.g., "LlamaForCausalLM").
        model: The JAX model class to register (e.g., a flax.nnx.Module).
    """
    _validate_model_interface(model)

    # Register with tpu_inference registry for the JAX backend
    _MODEL_REGISTRY[arch] = model

    # Create a vLLM-compatible wrapper for the JAX model class.
    # This wrapper inherits from the JAX model and torch.nn.Module
    # to pass vLLM's type checks. It is not meant to be instantiated
    # or executed by vLLM's PyTorch backend.
    def unimplemented_forward(
        self,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional["torch.Tensor"] = None,
    ) -> None:
        raise NotImplementedError(
            "This is a JAX model and does not implement the PyTorch forward method."
        )

    # Same as `forward`, this is a dummy method to satisfy vLLM's type checks.
    def unimplemented_embed_input_ids(
        self,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        inputs_embeds: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        raise NotImplementedError(
            "This is a JAX model and does not implement the PyTorch embed_input_ids method."
        )

    # We need a custom __init__ that only calls torch.nn.Module's init,
    # to avoid triggering JAX logic when vLLM inspects the class.
    def wrapper_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    # Dynamically create the wrapper class that is a subclass of both the
    # JAX model and torch.nn.Module.
    VllmCompatibleModel = type(
        f"VllmCompatible{model.__name__}",
        (model, torch.nn.Module),
        {
            "__init__": wrapper_init,
            "forward": unimplemented_forward,
            "embed_input_ids": unimplemented_embed_input_ids,
            # Prevent vLLM from trying to load weights into this dummy class.
            "load_weights": lambda self, *args, **kwargs: None,
        })

    # Register the wrapped model with vLLM's registry.
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model(arch, VllmCompatibleModel)
    logger.info(
        f"Registered JAX model {arch} with tpu_inference and vLLM registries.")
