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

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
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
    mock_config.quant_config = None
    return mock_config


# --- Added RNG Fixture ---
@pytest.fixture
def rng() -> jax.Array:
    """Provides a JAX PRNGKey."""
    return jax.random.PRNGKey(0)


# --- Added jax get_pp_group Fixture ---
@pytest.fixture
def mock_get_pp_group():
    with patch("tpu_inference.distributed.jax_parallel_state.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


@pytest.fixture(autouse=True)
def mock_register_oot():
    """Prevents Duplicate layer name errors from register_oot."""
    with patch(
            "vllm.model_executor.layers.mla.MultiHeadLatentAttentionWrapper.register_oot"
    ):
        yield


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


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_get_flax_model(vllm_config, mesh, tie_word_embeddings):
    """
    An integration test for the main public function `get_flax_model`.
    It verifies that the function returns two valid, JIT-compiled functions
    that execute correctly and produce outputs with the expected sharding.

    The model under test is Qwen3-0.6B, whose config sets tie_word_embeddings
    to True by default, but also provides lm_head weights in the checkpoint.
    This test runs with both tie_word_embeddings=True and False to ensure
    that the model loading logic handles both cases correctly.
    """
    rng = jax.random.PRNGKey(42)
    assert hasattr(vllm_config.model_config.hf_config, "tie_word_embeddings")
    vllm_config.model_config.hf_config.tie_word_embeddings = tie_word_embeddings

    # 1. Get the compiled model and logit computation functions
    init_pp_distributed_environment(ip="",
                                    rank=0,
                                    world_size=1,
                                    device=jax.devices()[0],
                                    need_pp=False)
    with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
        model_fn, compute_logits_fn, *_ = model_loader.get_flax_model(
            vllm_config, rng, mesh)

    assert callable(model_fn)
    assert callable(compute_logits_fn)


def test_get_vllm_model(mock_get_pp_group, mesh):
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


def test_get_vllm_model_random_weights(mock_get_pp_group, mesh):
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
    def test_get_model_flax_fallback_to_vllm_with_pp_on_unsupported_models(
            self, mock_get_flax, mock_get_vllm, vllm_config, rng, mesh):
        """Tests that 'flax_nnx' impl calls get_vllm_model when PP is enabled on
        unsupported models (e.g. GptOssForCausalLM)."""
        mock_get_flax.return_value = "flax_model_sentinel"
        mock_get_vllm.return_value = "vllm_model_sentinel"
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.model_config.hf_config.architectures = [
            "GptOssForCausalLM"
        ]
        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_not_called()
        mock_get_vllm.assert_called_once_with(vllm_config, rng, mesh)
        assert result == "vllm_model_sentinel"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "flax_nnx"}, clear=True)
    @patch("tpu_inference.models.common.model_loader.get_vllm_model")
    @patch("tpu_inference.models.common.model_loader.get_flax_model")
    def test_get_model_flax_no_fallback_with_pp_on_supported_models(
            self, mock_get_flax, mock_get_vllm, vllm_config, rng, mesh):
        """Tests that 'flax_nnx' impl calls get_flax_model when PP is enabled on
        supported models (e.g. Qwen3ForCausalLM)."""
        mock_get_flax.return_value = "flax_model_sentinel"
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.model_config.hf_config.architectures = ["Qwen3ForCausalLM"]
        result = model_loader.get_model(vllm_config, rng, mesh)

        mock_get_flax.assert_called_once_with(vllm_config, rng, mesh, False)
        mock_get_vllm.assert_not_called()
        assert result == "flax_model_sentinel"

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

    def test_abstract_dummy_config(self, vllm_config):
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy",
            }
        }
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "abstract_dummy"
        assert config.prefer_jax_for_bootstrap is False

    def test_prefer_jax_config(self, vllm_config):
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "prefer_jax_for_bootstrap": True,
            }
        }
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "default"
        assert config.prefer_jax_for_bootstrap is True

    def test_invalid_model_bootstrap_raises(self, vllm_config):
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "invalid_value",
            }
        }
        with pytest.raises(ValueError, match="Invalid tpu_bootstrap"):
            model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)

    def test_missing_extra_config(self, vllm_config):
        vllm_config.load_config.model_loader_extra_config = None
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "default"

    def test_empty_extra_config(self, vllm_config):
        vllm_config.load_config.model_loader_extra_config = {}
        config = model_loader.TpuBootstrapConfig.from_vllm_config(vllm_config)
        assert config.model_bootstrap == "default"


class TestAbstractDummyBootstrap:
    """Tests for _use_abstract_dummy_bootstrap gating logic."""

    def _make_mock_class(self, name):
        return type(name, (), {})

    def test_enabled_for_llama(self, vllm_config):
        vllm_config.load_config.load_format = "dummy"
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        assert model_loader._use_abstract_dummy_bootstrap(
            vllm_config, model_class) is True

    def test_disabled_for_qwen3(self, vllm_config):
        """Qwen3 implements LoadableWithIterator, so it should NOT use
        the abstract dummy bootstrap override."""
        vllm_config.load_config.load_format = "dummy"
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        model_class = self._make_mock_class("Qwen3ForCausalLM")
        assert model_loader._use_abstract_dummy_bootstrap(
            vllm_config, model_class) is False

    def test_disabled_without_opt_in(self, vllm_config):
        vllm_config.load_config.load_format = "dummy"
        vllm_config.load_config.model_loader_extra_config = {}
        model_class = self._make_mock_class("LlamaForCausalLM")
        assert model_loader._use_abstract_dummy_bootstrap(
            vllm_config, model_class) is False

    def test_disabled_for_non_dummy_format(self, vllm_config):
        vllm_config.load_config.load_format = "auto"
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        assert model_loader._use_abstract_dummy_bootstrap(
            vllm_config, model_class) is False

    def test_disabled_with_hf_quantization_config(self, vllm_config):
        vllm_config.load_config.load_format = "dummy"
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        vllm_config.model_config.hf_config.quantization_config = {
            "quant_method": "gptq"
        }
        model_class = self._make_mock_class("LlamaForCausalLM")
        assert model_loader._use_abstract_dummy_bootstrap(
            vllm_config, model_class) is False

    def test_disabled_with_tpu_quantization(self, vllm_config):
        """TPU quantization via model_config.quantization (e.g. 'tpu_int8')
        should also block abstract dummy bootstrap."""
        vllm_config.load_config.load_format = "dummy"
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "model_bootstrap": "abstract_dummy"
            }
        }
        vllm_config.model_config.quantization = "tpu_int8"
        model_class = self._make_mock_class("LlamaForCausalLM")
        assert model_loader._use_abstract_dummy_bootstrap(
            vllm_config, model_class) is False


class TestBootstrapAwareRouting:
    """Tests for bootstrap-aware architecture routing."""

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    def test_qwen3moe_defaults_to_vllm(self, vllm_config):
        """Without bootstrap config, Qwen3MoE routes to vllm."""
        vllm_config.model_config.hf_config.architectures = [
            "Qwen3MoeForCausalLM"
        ]
        result = model_loader.resolve_model_architecture(vllm_config)
        assert result == "vllm"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    def test_qwen3moe_bootstrap_routes_to_flax(self, vllm_config):
        """With prefer_jax_for_bootstrap, Qwen3MoE routes to flax_nnx."""
        vllm_config.model_config.hf_config.architectures = [
            "Qwen3MoeForCausalLM"
        ]
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "prefer_jax_for_bootstrap": True
            }
        }
        result = model_loader.resolve_model_architecture(vllm_config)
        assert result == "flax_nnx"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    def test_llama_routes_to_flax_by_default(self, vllm_config):
        """Llama is not in _VLLM_PREFERRED_ARCHITECTURES, so it routes
        to flax_nnx without any bootstrap config."""
        vllm_config.model_config.hf_config.architectures = ["LlamaForCausalLM"]
        result = model_loader.resolve_model_architecture(vllm_config)
        assert result == "flax_nnx"

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    def test_unsupported_arch_falls_through(self, vllm_config):
        """For unsupported architectures, prefer_jax_for_bootstrap doesn't
        override — falls through to default logic."""
        vllm_config.model_config.hf_config.architectures = ["TotallyFakeModel"]
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "prefer_jax_for_bootstrap": True
            }
        }
        result = model_loader.resolve_model_architecture(vllm_config)
        assert result == "flax_nnx"  # not in _VLLM_PREFERRED

    @patch.dict(os.environ, {"MODEL_IMPL_TYPE": "auto"}, clear=True)
    def test_gptoss_not_rerouted_by_bootstrap(self, vllm_config):
        """GptOssForCausalLM is JAX-registered AND vllm-preferred.
        prefer_jax_for_bootstrap must NOT reroute it because it is not
        in the explicit _BOOTSTRAP_JAX_ROUTING_ALLOWLIST."""
        vllm_config.model_config.hf_config.architectures = [
            "GptOssForCausalLM"
        ]
        vllm_config.load_config.model_loader_extra_config = {
            "tpu_bootstrap": {
                "prefer_jax_for_bootstrap": True
            }
        }
        result = model_loader.resolve_model_architecture(vllm_config)
        assert result == "vllm"
