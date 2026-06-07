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
import importlib.util
import os

from tpu_inference.layers.vllm import backends as backends
from tpu_inference.layers.vllm import custom_ops as custom_ops
from tpu_inference.layers.vllm import ops as ops
from tpu_inference.layers.vllm import quantization as quantization


def _is_tpu_stack_active() -> bool:
    if os.getenv("VLLM_TARGET_DEVICE", "").lower() == "tpu":
        return True

    jax_platforms = {
        platform.strip().lower()
        for platform in os.getenv("JAX_PLATFORMS", "").split(",")
    }
    if jax_platforms & {"tpu", "proxy"}:
        return True

    return importlib.util.find_spec("libtpu") is not None


def _register_grugmoe() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from tpu_inference.models.common.model_loader import register_model
    from tpu_inference.models.jax.grugmoe import (GrugMoeForCausalLM,
                                                  GrugMoeHfConfig)

    _CONFIG_REGISTRY[GrugMoeHfConfig.model_type] = GrugMoeHfConfig
    AutoConfig.register(
        GrugMoeHfConfig.model_type,
        GrugMoeHfConfig,
        exist_ok=True,
    )

    if "GrugMoeForCausalLM" not in ModelRegistry.get_supported_archs():
        register_model("GrugMoeForCausalLM", GrugMoeForCausalLM)


def register_layers():
    if _is_tpu_stack_active():
        _register_grugmoe()
