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
"""Tests for streaming_weights.py (fsspec-based safetensors loader)."""

import json
import os
import struct
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.models.jax.streaming_weights import (
    ChunkSpec,
    TensorRecord,
    _build_chunks,
    _discover_shards,
    _read_metadata,
    fsspec_weights_iterator,
)


def _write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal safetensors file from a dict of numpy arrays."""
    # Build header
    header = {}
    offset = 0
    data_parts = []
    for name, arr in tensors.items():
        raw = arr.tobytes()
        dtype_map = {
            np.dtype("float32"): "F32",
            np.dtype("float16"): "F16",
            np.dtype(jnp.bfloat16): "BF16",
            np.dtype("int32"): "I32",
            np.dtype("int64"): "I64",
            np.dtype("uint8"): "U8",
        }
        dtype_str = dtype_map[arr.dtype]
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)

    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))

    with open(path, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        for part in data_parts:
            f.write(part)


@pytest.fixture
def single_shard_dir(tmp_path):
    """Create a temp dir with a single model.safetensors file."""
    tensors = {
        "model.embed_tokens.weight": np.random.randn(32, 16).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(16, 16).astype(np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(4, 16).astype(np.float32),
    }
    _write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    return tmp_path, tensors


@pytest.fixture
def multi_shard_dir(tmp_path):
    """Create a temp dir with multiple safetensors shards and an index."""
    shard1_tensors = {
        "model.embed_tokens.weight": np.random.randn(32, 16).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(16, 16).astype(np.float32),
    }
    shard2_tensors = {
        "model.layers.1.self_attn.q_proj.weight": np.random.randn(16, 16).astype(np.float32),
        "lm_head.weight": np.random.randn(32, 16).astype(np.float32),
    }

    _write_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"), shard1_tensors)
    _write_safetensors(str(tmp_path / "model-00002-of-00002.safetensors"), shard2_tensors)

    # Write index
    weight_map = {}
    for k in shard1_tensors:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard2_tensors:
        weight_map[k] = "model-00002-of-00002.safetensors"

    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    all_tensors = {**shard1_tensors, **shard2_tensors}
    return tmp_path, all_tensors


class TestReadMetadata:
    def test_parse_safetensors_header(self, single_shard_dir):
        """Correct tensor names, shapes, dtypes, offsets."""
        path, expected_tensors = single_shard_dir
        import fsspec
        fs = fsspec.filesystem("file")
        records = _read_metadata(fs, str(path / "model.safetensors"))

        assert set(records.keys()) == set(expected_tensors.keys())
        for name, arr in expected_tensors.items():
            rec = records[name]
            assert rec.shape == arr.shape
            assert rec.dtype == arr.dtype
            assert rec.byte_end > rec.byte_start
            expected_size = int(np.prod(arr.shape)) * arr.dtype.itemsize
            assert rec.byte_end - rec.byte_start == expected_size


class TestBuildChunks:
    def test_groups_correctly(self):
        """Adjacent tensors grouped within limit."""
        records = [
            TensorRecord("a", np.dtype("float32"), (4,), "f.st", 100, 116),
            TensorRecord("b", np.dtype("float32"), (4,), "f.st", 116, 132),
            TensorRecord("c", np.dtype("float32"), (4,), "f.st", 132, 148),
        ]
        # All fit in one chunk
        chunks = _build_chunks(records, chunk_limit=1000)
        assert len(chunks) == 1
        assert len(chunks[0].tensors) == 3
        assert chunks[0].byte_start == 100
        assert chunks[0].byte_end == 148

    def test_splits_on_limit(self):
        """Tensors split into multiple chunks when exceeding limit."""
        records = [
            TensorRecord("a", np.dtype("float32"), (4,), "f.st", 100, 116),
            TensorRecord("b", np.dtype("float32"), (4,), "f.st", 116, 132),
            TensorRecord("c", np.dtype("float32"), (4,), "f.st", 132, 148),
        ]
        # Force split: each tensor is 16 bytes, limit of 20 means max 1 per chunk
        chunks = _build_chunks(records, chunk_limit=20)
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk.tensors) == 1

    def test_tensor_larger_than_chunk_limit(self):
        """Single tensor larger than chunk_size gets its own chunk."""
        records = [
            TensorRecord("small", np.dtype("float32"), (4,), "f.st", 100, 116),
            TensorRecord("big", np.dtype("float32"), (1000,), "f.st", 116, 4116),
            TensorRecord("small2", np.dtype("float32"), (4,), "f.st", 4116, 4132),
        ]
        chunks = _build_chunks(records, chunk_limit=100)
        assert len(chunks) == 3
        assert chunks[1].tensors[0].key == "big"

    def test_empty_input(self):
        chunks = _build_chunks([], chunk_limit=1000)
        assert chunks == []

    def test_invalid_chunk_limit(self):
        with pytest.raises(ValueError, match="chunk_limit must be positive"):
            _build_chunks([], chunk_limit=0)


class TestDiscoverShards:
    def test_single_shard(self, single_shard_dir):
        path, _ = single_shard_dir
        import fsspec
        fs = fsspec.filesystem("file")
        shards = _discover_shards(fs, str(path))
        assert len(shards) == 1
        assert shards[0].endswith("model.safetensors")

    def test_multi_shard_index(self, multi_shard_dir):
        path, _ = multi_shard_dir
        import fsspec
        fs = fsspec.filesystem("file")
        shards = _discover_shards(fs, str(path))
        assert len(shards) == 2
        assert "model-00001-of-00002.safetensors" in shards[0]
        assert "model-00002-of-00002.safetensors" in shards[1]

    def test_no_safetensors_raises(self, tmp_path):
        import fsspec
        fs = fsspec.filesystem("file")
        with pytest.raises(FileNotFoundError, match="No safetensors"):
            _discover_shards(fs, str(tmp_path))


class TestFsspecWeightsIterator:
    def test_local_single_file(self, single_shard_dir):
        """End-to-end: yields correct (name, jax.Array) on CPU."""
        path, expected_tensors = single_shard_dir
        result = dict(fsspec_weights_iterator(str(path)))

        assert set(result.keys()) == set(expected_tensors.keys())
        for name, expected_arr in expected_tensors.items():
            jax_arr = result[name]
            assert isinstance(jax_arr, jax.Array)
            np.testing.assert_allclose(
                np.array(jax_arr), expected_arr, rtol=1e-6)

    def test_multi_shard(self, multi_shard_dir):
        """Multi-shard index discovery and loading works correctly."""
        path, expected_tensors = multi_shard_dir
        result = dict(fsspec_weights_iterator(str(path)))

        assert set(result.keys()) == set(expected_tensors.keys())
        for name, expected_arr in expected_tensors.items():
            np.testing.assert_allclose(
                np.array(result[name]), expected_arr, rtol=1e-6)

    def test_arrays_on_cpu(self, single_shard_dir):
        """All yielded arrays are on CPU, not TPU."""
        path, _ = single_shard_dir
        cpu_device = jax.devices("cpu")[0]
        for name, arr in fsspec_weights_iterator(str(path)):
            assert arr.devices() == {cpu_device}, (
                f"Tensor {name} not on CPU: {arr.devices()}")

    def test_stable_ordering(self, single_shard_dir):
        """Tensor order is deterministic across calls."""
        path, _ = single_shard_dir
        keys1 = [k for k, _ in fsspec_weights_iterator(str(path))]
        keys2 = [k for k, _ in fsspec_weights_iterator(str(path))]
        assert keys1 == keys2

    def test_bf16_tensors(self, tmp_path):
        """BF16 dtype roundtrips correctly."""
        bf16_arr = np.random.randn(8, 4).astype(jnp.bfloat16)
        _write_safetensors(str(tmp_path / "model.safetensors"),
                           {"bf16_weight": bf16_arr})

        result = dict(fsspec_weights_iterator(str(tmp_path)))
        assert "bf16_weight" in result
        loaded = result["bf16_weight"]
        assert loaded.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.array(loaded), bf16_arr)

    def test_small_chunk_size(self, single_shard_dir):
        """Small chunk_size forces multiple chunks per shard."""
        path, expected_tensors = single_shard_dir
        # Use tiny chunk size to force splitting
        result = dict(fsspec_weights_iterator(str(path), chunk_size_bytes=64))
        assert set(result.keys()) == set(expected_tensors.keys())
        for name, expected_arr in expected_tensors.items():
            np.testing.assert_allclose(
                np.array(result[name]), expected_arr, rtol=1e-6)
