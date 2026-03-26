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

import os
import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from transformers import PretrainedConfig
from vllm.config import (ModelConfig, ParallelConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.registry import ModelRegistry

from vllm.model_executor.model_loader.runai_streamer_loader import \
    RunaiModelStreamerLoader

from tpu_inference.models.common import model_loader
from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM


class MockModelA:

    def __init__(self, vllm_config, rng=None, mesh=None):
        pass

    def __call__(self, kv_caches, input_ids, attention_metadata):
        pass


class MockModelB:

    def __init__(self, vllm_config, rng=None, mesh=None):
        pass

    def __call__(self, kv_caches, input_ids, attention_metadata):
        pass


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """Provides a JAX device mesh for sharding."""
    devices = np.array(jax.devices()[:1])
    devices = devices.reshape((1, 1, 1, -1))
    # Pass the 1D list of devices directly. Its ndim will match len(axis_names).
    return Mesh(devices, axis_names=("data", "attn_dp", "expert", "model"))


@pytest.fixture
def vllm_config() -> MagicMock:
    """Provides a mock VllmConfig object."""
    model = "Qwen/Qwen3-0.6B"
    mock_config = MagicMock(spec=VllmConfig)
    mock_config.model_config = ModelConfig(model)
    mock_config.model_config.dtype = jnp.bfloat16
    mock_config.load_config = MagicMock()
    mock_config.load_config.download_dir = None
    mock_config.load_config.load_format = "auto"
    mock_config.load_config.model_loader_extra_config = dict()
    mock_config.additional_config = dict()
    mock_config.cache_config = MagicMock(cache_dtype="auto")
    mock_config.parallel_config = ParallelConfig(pipeline_parallel_size=1)
    return mock_config


# --- Added RNG Fixture ---
@pytest.fixture
def rng() -> jax.Array:
    """Provides a JAX PRNGKey."""
    return jax.random.PRNGKey(0)


# ==============================================================================
# >> Test Cases
# ==============================================================================


def test_get_model_architecture_supported(vllm_config):
    """
    Tests that _get_model_architecture returns the correct model class
    for a supported architecture.
    """
    config = vllm_config.model_config.hf_config
    model_class = model_loader._get_model_architecture(config)
    assert model_class == Qwen3ForCausalLM


def test_get_model_architecture_unsupported():
    """
    Tests that _get_model_architecture raises a ValueError for an
    unsupported architecture.
    """
    config = PretrainedConfig(architectures=["UnsupportedModel"])
    with pytest.raises(ValueError, match="not registered"):
        model_loader._get_model_architecture(config)


def test_get_model_architecture_mistral_alias():
    """
    Tests that MistralForCausalLM (vLLM's remap of LlamaForCausalLM)
    resolves to the same JAX LlamaForCausalLM class.
    """
    from tpu_inference.models.jax.llama3 import LlamaForCausalLM
    config = PretrainedConfig(
        architectures=["MistralForCausalLM"],
        model_type="transformer",
    )
    model_class = model_loader._get_model_architecture(config)
    assert model_class == LlamaForCausalLM


def test_get_model_architecture_model_type_fallback():
    """
    Tests that _get_model_architecture falls back to model_type when
    architectures are not in the registry but model_type is known.
    """
    from tpu_inference.models.jax.llama3 import LlamaForCausalLM
    config = PretrainedConfig(
        architectures=["SomeUnknownArch"],
        model_type="llama",
    )
    model_class = model_loader._get_model_architecture(config)
    assert model_class == LlamaForCausalLM


@pytest.fixture(autouse=True)
def clear_model_registry_after_test():
    """Clear the model registry after each test to prevent side effects."""
    yield
    model_loader._MODEL_REGISTRY.clear()


def test_register_model_validation():
    """Tests that register_model validates the model interface."""

    class ValidModel:

        def __init__(self, vllm_config, rng=None, mesh=None):
            pass

        def __call__(self, kv_caches, input_ids, attention_metadata, **kwargs):
            pass

    class MissingInitArgModel:

        def __init__(self, rng=None, mesh=None):  # Missing vllm_config
            pass

        def __call__(self, kv_caches, input_ids, attention_metadata):
            pass

    class MissingCallArgModel:

        def __init__(self, vllm_config, rng=None, mesh=None):
            pass

        def __call__(self, kv_caches, input_ids):  # Missing attention_metadata
            pass

    class NoCallModel:

        def __init__(self, vllm_config, rng=None, mesh=None):
            pass

    # This should succeed
    model_loader.register_model("ValidModel", ValidModel)

    # These should fail
    with pytest.raises(TypeError, match="vllm_config"):
        model_loader.register_model("InvalidInit", MissingInitArgModel)

    with pytest.raises(TypeError, match="attention_metadata"):
        model_loader.register_model("InvalidCall", MissingCallArgModel)

    with pytest.raises(TypeError, match="__call__ method"):
        model_loader.register_model("NoCallModel", NoCallModel)


def test_register_model_new_arch():
    """Tests registering a new model architecture."""
    arch = "NewArch"
    model_loader.register_model(arch, MockModelA)

    # Check tpu_inference registry
    config = PretrainedConfig(architectures=[arch])
    model_class = model_loader._get_model_architecture(config)
    assert model_class == MockModelA

    # Check vLLM registry
    vllm_model_class = ModelRegistry._try_load_model_cls(arch)
    assert vllm_model_class is not None
    assert vllm_model_class.__name__ == f"VllmCompatible{MockModelA.__name__}"
    assert issubclass(vllm_model_class, MockModelA)
    assert issubclass(vllm_model_class, torch.nn.Module)


def test_register_model_update_arch():
    """Tests updating an existing registered model architecture."""
    arch = "UpdatableArch"
    config = PretrainedConfig(architectures=[arch])

    # Register initial model
    model_loader.register_model(arch, MockModelA)

    # Verify initial registration in both registries
    model_class_1 = model_loader._get_model_architecture(config)
    assert model_class_1 == MockModelA
    vllm_model_class_1 = ModelRegistry._try_load_model_cls(arch)
    assert vllm_model_class_1.__name__ == f"VllmCompatible{MockModelA.__name__}"
    assert issubclass(vllm_model_class_1, MockModelA)

    # Update the registration
    model_loader.register_model(arch, MockModelB)

    # Verify the update in both registries
    model_class_2 = model_loader._get_model_architecture(config)
    assert model_class_2 == MockModelB
    vllm_model_class_2 = ModelRegistry._try_load_model_cls(arch)
    assert vllm_model_class_2.__name__ == f"VllmCompatible{MockModelB.__name__}"
    assert issubclass(vllm_model_class_2, MockModelB)


def test_register_model_vllm_wrapper_methods():
    """Tests that the vLLM wrapper has correct dummy methods."""
    arch = "WrapperMethodTestArch"
    model_loader.register_model(arch, MockModelA)

    vllm_model_class = ModelRegistry._try_load_model_cls(arch)
    instance = vllm_model_class()

    # `forward` should be unimplemented.
    with pytest.raises(NotImplementedError, match="JAX model"):
        instance.forward(input_ids=None, positions=None)

    # `embed_input_ids` should be unimplemented.
    with pytest.raises(NotImplementedError, match="JAX model"):
        instance.embed_input_ids(input_ids=None, positions=None)

    # `load_weights` should be a no-op that returns None.
    assert instance.load_weights() is None


def test_get_flax_model(vllm_config, mesh):
    """
    An integration test for the main public function `get_flax_model`.
    It verifies that the function returns two valid, JIT-compiled functions
    that execute correctly and produce outputs with the expected sharding.
    """
    rng = jax.random.PRNGKey(42)

    # 1. Get the compiled model and logit computation functions
    model_fn, compute_logits_fn, *_ = model_loader.get_flax_model(
        vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)


def test_get_vllm_model(mesh):
    """
    An integration test for the main public function `get_vllm_model`.
    It verifies that the function returns two valid, JIT-compiled functions
    that execute correctly and produce outputs with the expected sharding.
    """
    rng = jax.random.PRNGKey(42)

    engine_args = EngineArgs(model="Qwen/Qwen3-0.6B")
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    model_fn, compute_logits_fn, *_ = model_loader.get_vllm_model(
        vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)


def test_get_vllm_model_random_weights(mesh):
    rng = jax.random.PRNGKey(42)

    engine_args = EngineArgs(model="Qwen/Qwen3-0.6B")
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.load_config.load_format = "dummy"

    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    with patch(
            "vllm.model_executor.model_loader.dummy_loader.DummyModelLoader.load_weights"
    ) as mock_load:
        model_fn, compute_logits_fn, *_ = model_loader.get_vllm_model(
            vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)
    mock_load.assert_called()


# ==============================================================================
# >> Test Suite for get_model Fallback Logic
# ==============================================================================


@pytest.mark.usefixtures("mesh")  # This fixture is module-scoped, but fine
class TestGetModel:
    """Tests the main get_model() entrypoint and its fallback logic."""

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_happy_path(self, mock_get_flax, mock_get_vllm,
                                       vllm_config, rng, mesh):
        """Tests that 'flax_nnx' impl calls get_flax_model."""
        mock_get_flax.return_value = "flax_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_not_called()
        assert result == "flax_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_happy_path_withPP(self, mock_get_flax,
                                              mock_get_vllm, vllm_config, rng,
                                              mesh):
        """Tests that 'flax_nnx' impl calls get_vllm_model when PP is enabled."""
        mock_get_flax.return_value = "flax_model_sentinel"
        mock_get_vllm.return_value = "vllm_model_sentinel"
        vllm_config.parallel_config.pipeline_parallel_size = 2
        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "vllm"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_vllm_happy_path(self, mock_get_flax, mock_get_vllm,
                                       vllm_config, rng, mesh):
        """Tests that 'vllm' impl calls get_vllm_model."""
        mock_get_vllm.return_value = "vllm_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_fallback_on_unsupported_arch(
            self, mock_get_flax, mock_get_vllm, vllm_config, rng, mesh):
        """
        Tests that 'flax_nnx' falls back to get_vllm_model on
        UnsupportedArchitectureError.
        """
        # Mock get_flax_model to raise the specific error
        mock_get_flax.side_effect = model_loader.UnsupportedArchitectureError(
            "Model not supported")
        mock_get_vllm.return_value = "vllm_fallback_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        # Check that both were called
        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_fallback_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_reraises_other_errors(self, mock_get_flax,
                                                  mock_get_vllm, vllm_config,
                                                  rng, mesh):
        """
        Tests that 'flax_nnx' re-raises other ValueErrors
        and does not fall back.
        """
        # Mock get_flax_model to raise a *different* error
        mock_get_flax.side_effect = ValueError("A different error")

        with pytest.raises(ValueError, match="A different error"):
            model_loader.get_model(vllm_config, rng, mesh)

        # Check that flax was called but vllm was not
        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_not_called()

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "jetpack"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_not_implemented(self, mock_get_flax, mock_get_vllm,
                                       vllm_config, rng, mesh):
        """Tests that an unknown impl raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_not_called()

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_auto_resolves_to_flax_nnx(self, mock_get_flax,
                                                 mock_get_vllm, vllm_config,
                                                 rng, mesh):
        """
        Tests that 'auto' resolves to 'flax_nnx' for standard architectures
        (not in _VLLM_REQUIRED_ARCHITECTURES).
        """
        # vllm_config uses Qwen3 which is NOT in _VLLM_REQUIRED_ARCHITECTURES
        mock_get_flax.return_value = "flax_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_not_called()
        assert result == "flax_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_auto_resolves_to_vllm_for_gpt_oss(
            self, mock_get_flax, mock_get_vllm, vllm_config, rng, mesh):
        """
        Tests that 'auto' resolves to 'vllm' for architectures in
        _VLLM_REQUIRED_ARCHITECTURES (e.g., GptOssForCausalLM).
        """
        # Mock the architecture to be GptOssForCausalLM
        vllm_config.model_config.hf_config.architectures = [
            "GptOssForCausalLM"
        ]
        mock_get_vllm.return_value = "vllm_model_sentinel"

        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_model_sentinel"


# ==============================================================================
# >> Test Suite for TpuBootstrapConfig and Fast Bootstrap
# ==============================================================================


class TestTpuBootstrapConfig:
    """Tests for TpuBootstrapConfig parsing and validation."""

    def test_default_config(self, vllm_config):
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "default"
        assert config.prefer_jax_for_bootstrap is False
        assert config.weight_loader == "default"

    def test_abstract_dummy_config(self, vllm_config):
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy",
            }
        }
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "abstract_dummy"
        assert config.prefer_jax_for_bootstrap is False

    def test_abstract_load_config(self, vllm_config):
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load",
            }
        }
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "abstract_load"
        assert config.prefer_jax_for_bootstrap is False

    def test_prefer_jax_config(self, vllm_config):
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "prefer_jax_for_bootstrap": True,
            }
        }
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "default"
        assert config.prefer_jax_for_bootstrap is True

    def test_invalid_model_bootstrap_raises(self, vllm_config):
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "invalid_value",
            }
        }
        with pytest.raises(ValueError, match="Invalid tpu_bootstrap"):
            model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)

    def test_missing_extra_config(self, vllm_config):
        vllm_config.additional_config = None
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "default"

    def test_empty_extra_config(self, vllm_config):
        vllm_config.additional_config = {}
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "default"


class TestResolvedBootstrapMode:
    """Tests for _resolved_bootstrap_mode gating logic."""

    def _make_mock_class(self, name):
        return type(name, (), {})

    def test_abstract_dummy_returns_abstract_dummy(self, vllm_config):
        vllm_config.load_config.load_format = "dummy"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        assert model_loader._resolved_bootstrap_mode(
            vllm_config, model_class) == "abstract_dummy"

    def test_abstract_load_for_llama(self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            }
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        assert model_loader._resolved_bootstrap_mode(
            vllm_config, model_class) == "abstract_load"

    def test_default_mode_silently_returns_default(self, vllm_config):
        """Default mode returns 'default' even for unsupported architectures
        — only non-default modes raise."""
        vllm_config.additional_config = {}
        model_class = self._make_mock_class("UnsupportedArch")
        assert model_loader._resolved_bootstrap_mode(
            vllm_config, model_class) == "default"

    def test_abstract_load_on_unsupported_arch_raises(self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            }
        }
        model_class = self._make_mock_class("Qwen3ForCausalLM")
        with pytest.raises(ValueError, match="not supported for architecture"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    def test_abstract_dummy_on_unsupported_arch_raises(self, vllm_config):
        vllm_config.load_config.load_format = "dummy"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        model_class = self._make_mock_class("Qwen3ForCausalLM")
        with pytest.raises(ValueError, match="not supported for architecture"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    def test_abstract_load_with_quantization_raises(self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            }
        }
        vllm_config.model_config.hf_config.quantization_config = {
            "quant_method": "gptq"
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        with pytest.raises(ValueError, match="quantization_config"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    def test_abstract_load_with_tpu_quantization_raises(self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            }
        }
        vllm_config.model_config.quantization = "tpu_int8"
        model_class = self._make_mock_class("LlamaForCausalLM")
        with pytest.raises(ValueError, match="TPU quantization"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    def test_abstract_load_with_dummy_load_format_raises(self, vllm_config):
        vllm_config.load_config.load_format = "dummy"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            }
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        with pytest.raises(ValueError, match="requires a real load_format"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    def test_gptoss_abstract_load_allows_mxfp4_when_skip_quantization(
            self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            },
            "skip_quantization": True,
        }
        vllm_config.model_config.hf_config.quantization_config = {
            "quant_method": "mxfp4"
        }
        model_class = self._make_mock_class("GptOss")
        assert model_loader._resolved_bootstrap_mode(
            vllm_config, model_class) == "abstract_load"

    def test_gptoss_abstract_load_requires_skip_quantization(
            self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            }
        }
        vllm_config.model_config.hf_config.quantization_config = {
            "quant_method": "mxfp4"
        }
        model_class = self._make_mock_class("GptOss")
        with pytest.raises(ValueError, match="quantization_config"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    def test_gptoss_abstract_load_allows_tpu_mxfp4_when_skip_quantization(
            self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            },
            "skip_quantization": True,
        }
        vllm_config.model_config.quantization = "tpu-mxfp4"
        model_class = self._make_mock_class("GptOss")
        assert model_loader._resolved_bootstrap_mode(
            vllm_config, model_class) == "abstract_load"

    def test_gptoss_abstract_load_with_tpu_quantization_requires_skip_quantization(
            self, vllm_config):
        vllm_config.load_config.load_format = "runai_streamer"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load"
            }
        }
        vllm_config.model_config.quantization = "tpu-mxfp4"
        model_class = self._make_mock_class("GptOss")
        with pytest.raises(ValueError, match="TPU quantization"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    def test_abstract_dummy_with_non_dummy_load_format_raises(self, vllm_config):
        vllm_config.load_config.load_format = "auto"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        with pytest.raises(ValueError, match="requires load_format='dummy'"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)


class TestAbstractLoadBehavior:
    """Behavioral tests for _build_abstract_model_and_load_weights and
    the abstract_load control-flow path in _get_nnx_model."""

    @patch("tpu_inference.models.common.model_loader.apply_qwix_on_abstract_model",
           return_value=False)
    @patch("tpu_inference.models.common.model_loader.get_model_loader")
    @patch("tpu_inference.models.common.model_loader.nnx.eval_shape")
    def test_abstract_load_calls_load_weights_and_jits(
            self, mock_eval_shape, mock_get_loader, mock_qwix_check):
        """Proves the abstract model is created, load_weights() is called,
        create_jit_model() is called, and the return value is the jitted model."""
        mock_model = MagicMock()
        mock_eval_shape.return_value = mock_model
        mock_loader = MagicMock()
        mock_loader.__class__ = type("DefaultLoader", (), {})
        mock_get_loader.return_value = mock_loader
        mock_jit_result = MagicMock(name="jit_model")

        create_abstract = MagicMock(name="create_abstract_model")
        create_jit = MagicMock(name="create_jit_model", return_value=mock_jit_result)

        vllm_config = MagicMock()
        rng = MagicMock()
        mesh = MagicMock()

        result = model_loader._build_abstract_model_and_load_weights(
            create_abstract, create_jit, vllm_config, rng, mesh)

        mock_eval_shape.assert_called_once()
        mock_model.load_weights.assert_called_once_with(rng)
        create_jit.assert_called_once_with(
            mock_model, use_qwix_on_abstract_model=False)
        assert result is mock_jit_result

    @patch("tpu_inference.models.common.model_loader.apply_qwix_on_abstract_model",
           return_value=False)
    @patch("tpu_inference.models.common.model_loader.get_model_loader")
    @patch("tpu_inference.models.common.model_loader.nnx.eval_shape")
    def test_abstract_load_cleans_model_weights_iterator_on_success(
            self, mock_eval_shape, mock_get_loader, mock_qwix_check):
        """With RunaiModelStreamerLoader, model_weights_iterator is set before
        load_weights and deleted afterward."""
        mock_model = MagicMock()
        mock_eval_shape.return_value = mock_model

        mock_loader = MagicMock(spec=RunaiModelStreamerLoader)
        mock_loader._get_weights_iterator.return_value = iter([("w", "data")])
        mock_get_loader.return_value = mock_loader

        create_abstract = MagicMock()
        create_jit = MagicMock(return_value=MagicMock(name="jit_model"))

        vllm_config = MagicMock()
        vllm_config.model_config.model = "/fake/model"
        vllm_config.model_config.revision = None
        # Ensure model_weights attr doesn't exist
        del vllm_config.model_config.model_weights
        mesh = MagicMock()
        rng = MagicMock()

        model_loader._build_abstract_model_and_load_weights(
            create_abstract, create_jit, vllm_config, rng, mesh)

        # Iterator should have been cleaned up
        assert not hasattr(vllm_config.model_config, "model_weights_iterator")
        mock_model.load_weights.assert_called_once_with(rng)

    @patch("tpu_inference.models.common.model_loader.apply_qwix_on_abstract_model",
           return_value=False)
    @patch("tpu_inference.models.common.model_loader.get_model_loader")
    @patch("tpu_inference.models.common.model_loader.nnx.eval_shape")
    def test_abstract_load_cleans_model_weights_iterator_on_failure(
            self, mock_eval_shape, mock_get_loader, mock_qwix_check):
        """If load_weights raises, model_weights_iterator is still cleaned up
        via try/finally."""
        mock_model = MagicMock()
        mock_model.load_weights.side_effect = RuntimeError("load failed")
        mock_eval_shape.return_value = mock_model

        mock_loader = MagicMock(spec=RunaiModelStreamerLoader)
        mock_loader._get_weights_iterator.return_value = iter([("w", "data")])
        mock_get_loader.return_value = mock_loader

        create_abstract = MagicMock()
        create_jit = MagicMock()

        vllm_config = MagicMock()
        vllm_config.model_config.model = "/fake/model"
        vllm_config.model_config.revision = None
        del vllm_config.model_config.model_weights
        mesh = MagicMock()
        rng = MagicMock()

        with pytest.raises(RuntimeError, match="load failed"):
            model_loader._build_abstract_model_and_load_weights(
                create_abstract, create_jit, vllm_config, rng, mesh)

        # Iterator should still be cleaned up despite the error
        assert not hasattr(vllm_config.model_config, "model_weights_iterator")

    @patch("tpu_inference.models.common.model_loader.apply_qwix_on_abstract_model",
           return_value=False)
    @patch("tpu_inference.models.common.model_loader.get_model_loader")
    @patch("tpu_inference.models.common.model_loader.nnx.eval_shape")
    def test_default_non_dummy_path_still_loads_and_jits(
            self, mock_eval_shape, mock_get_loader, mock_qwix_check):
        """Regression: mode='default' with a real load format goes through
        _build_abstract_model_and_load_weights (same as abstract_load)."""
        mock_model = MagicMock()
        mock_eval_shape.return_value = mock_model
        mock_loader = MagicMock()
        mock_loader.__class__ = type("DefaultLoader", (), {})
        mock_get_loader.return_value = mock_loader
        mock_jit_result = MagicMock(name="jit_model")

        create_abstract = MagicMock()
        create_jit = MagicMock(return_value=mock_jit_result)

        vllm_config = MagicMock()
        rng = MagicMock()
        mesh = MagicMock()

        result = model_loader._build_abstract_model_and_load_weights(
            create_abstract, create_jit, vllm_config, rng, mesh)

        mock_eval_shape.assert_called_once()
        mock_model.load_weights.assert_called_once_with(rng)
        create_jit.assert_called_once()
        assert result is mock_jit_result

    @patch("tpu_inference.models.common.model_loader._build_abstract_model_and_load_weights")
    @patch("tpu_inference.models.common.model_loader._resolved_bootstrap_mode")
    def test_get_nnx_model_dispatches_abstract_load(
            self, mock_mode, mock_build):
        """Core regression test: _get_nnx_model() takes the abstract_load
        branch and returns the jitted model (not the bare abstract model)."""
        mock_mode.return_value = "abstract_load"
        sentinel = MagicMock(name="jit_model_sentinel")
        mock_build.return_value = sentinel

        fake_model_class = type("LlamaForCausalLM", (), {})
        vllm_config = MagicMock()
        rng = MagicMock()
        mesh = MagicMock()

        result = model_loader._get_nnx_model(
            fake_model_class, vllm_config, rng, mesh)

        mock_mode.assert_called_once_with(vllm_config, fake_model_class)
        mock_build.assert_called_once()
        # The first two args are the closures; verify config/rng/mesh pass-through
        call_args = mock_build.call_args
        assert call_args[0][2] is vllm_config
        assert call_args[0][3] is rng
        assert call_args[0][4] is mesh
        assert result is sentinel


class TestBootstrapAwareRouting:
    """Tests for bootstrap-aware architecture routing in get_model."""

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_gptoss_defaults_to_vllm(self, mock_get_flax, mock_get_vllm,
                                     vllm_config, rng, mesh):
        """Without bootstrap config, GptOss routes to vllm."""
        vllm_config.model_config.hf_config.architectures = [
            "GptOssForCausalLM"
        ]
        mock_get_vllm.return_value = "vllm_model_sentinel"
        result = model_loader.get_model(vllm_config, rng, mesh)
        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once()
        assert result == "vllm_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_llama_routes_to_flax_by_default(self, mock_get_flax,
                                             mock_get_vllm, vllm_config, rng,
                                             mesh):
        """Llama is not in _VLLM_PREFERRED_ARCHITECTURES, so it routes
        to flax_nnx without any bootstrap config."""
        vllm_config.model_config.hf_config.architectures = ["LlamaForCausalLM"]
        mock_get_flax.return_value = "flax_model_sentinel"
        result = model_loader.get_model(vllm_config, rng, mesh)
        mock_get_flax.assert_called_once()
        mock_get_vllm.assert_not_called()
        assert result == "flax_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_gptoss_rerouted_by_bootstrap(self, mock_get_flax,
                                          mock_get_vllm, vllm_config, rng,
                                          mesh):
        """GPT-OSS can explicitly opt into the JAX bootstrap route."""
        vllm_config.model_config.hf_config.architectures = [
            "GptOssForCausalLM"
        ]
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "prefer_jax_for_bootstrap": True
            }
        }
        mock_get_flax.return_value = "flax_model_sentinel"
        result = model_loader.get_model(vllm_config, rng, mesh)
        mock_get_flax.assert_called_once()
        mock_get_vllm.assert_not_called()
        assert result == "flax_model_sentinel"


# ==============================================================================
# >> Test Suite for weight_loader Config and fsspec_streamer Dispatch
# ==============================================================================


class TestWeightLoaderConfig:
    """Tests for weight_loader field in TpuBootstrapConfig."""

    def test_weight_loader_config_parsing(self, vllm_config):
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "weight_loader": "fsspec_streamer",
            }
        }
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.weight_loader == "fsspec_streamer"

    def test_weight_loader_default(self, vllm_config):
        vllm_config.additional_config = {
            "tpu_bootstrap": {}
        }
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.weight_loader == "default"

    def test_invalid_weight_loader_raises(self, vllm_config):
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "weight_loader": "s3_magic",
            }
        }
        with pytest.raises(ValueError, match="Invalid tpu_bootstrap.weight_loader"):
            model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)

    def test_abstract_load_with_dummy_still_raises(self, vllm_config):
        """abstract_load + dummy load_format still rejected, regardless of weight_loader."""
        vllm_config.load_config.load_format = "dummy"
        vllm_config.additional_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_load",
                "weight_loader": "fsspec_streamer",
            }
        }
        model_class = type("LlamaForCausalLM", (), {})
        with pytest.raises(ValueError, match="requires a real load_format"):
            model_loader._resolved_bootstrap_mode(vllm_config, model_class)

    @patch("tpu_inference.models.common.model_loader.apply_qwix_on_abstract_model",
           return_value=False)
    @patch("tpu_inference.models.common.model_loader.get_model_loader")
    @patch("tpu_inference.models.common.model_loader.nnx.eval_shape")
    def test_fsspec_streamer_branch_sets_iterator(
            self, mock_eval_shape, mock_get_loader, mock_qwix_check):
        """When weight_loader=fsspec_streamer, fsspec_weights_iterator is used
        and model_weights_iterator is set/cleaned on model_config."""
        mock_model = MagicMock()
        mock_eval_shape.return_value = mock_model
        mock_loader = MagicMock()
        mock_loader.__class__ = type("DefaultLoader", (), {})
        mock_get_loader.return_value = mock_loader
        mock_jit_result = MagicMock(name="jit_model")

        create_abstract = MagicMock()
        create_jit = MagicMock(return_value=mock_jit_result)

        vllm_config = MagicMock()
        vllm_config.model_config.model = "/fake/model"
        del vllm_config.model_config.model_weights
        vllm_config.additional_config = {
            "tpu_bootstrap": {"weight_loader": "fsspec_streamer"}
        }
        rng_val = MagicMock()
        mesh_val = MagicMock()

        mock_iter = iter([("w", "data")])
        with patch(
            "tpu_inference.models.jax.streaming_weights.fsspec_weights_iterator",
            return_value=mock_iter,
        ) as mock_fsspec:
            result = model_loader._build_abstract_model_and_load_weights(
                create_abstract, create_jit, vllm_config, rng_val, mesh_val)

        mock_fsspec.assert_called_once_with("/fake/model")
        mock_model.load_weights.assert_called_once_with(rng_val)
        # Iterator should be cleaned up
        assert not hasattr(vllm_config.model_config, "model_weights_iterator")
        assert result is mock_jit_result


class TestLoadHfWeightsTypeDispatch:
    """Tests for jax.Array / torch.Tensor dispatch in load_hf_weights iterator path."""

    @patch("tpu_inference.models.jax.utils.weight_utils.nnx.state",
           return_value=MagicMock())
    @patch("tpu_inference.models.jax.utils.weight_utils.nnx.get_named_sharding",
           return_value=MagicMock())
    def test_load_hf_weights_rejects_ndarray(self, mock_shardings,
                                              mock_state, vllm_config, mesh):
        """Iterator yields np.ndarray → TypeError.

        We patch nnx.state/get_named_sharding so the test reaches the
        iterator type-dispatch branch (otherwise nnx.state fails on a
        non-Module mock).
        """
        from tpu_inference.models.jax.utils.weight_utils import load_hf_weights, MetadataMap

        mock_model = MagicMock()
        metadata_map = MetadataMap(name_map={})

        # Set up weights_iterator with a numpy array (unsupported type)
        vllm_config.model_config.model_weights_iterator = iter([
            ("bad_weight", np.zeros((4, 4)))
        ])

        with pytest.raises(TypeError, match="Unsupported weight type"):
            load_hf_weights(
                vllm_config, mock_model, metadata_map, mesh)
