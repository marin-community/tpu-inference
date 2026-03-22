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
"""Streaming weight loader using fsspec for fast GCS/local safetensors loading.

Adapted from Levanter's fsspec_safetensor.py. Streams weights shard-by-shard,
chunk-by-chunk sequentially. All arrays materialized on CPU.
Peak host RAM ≈ chunk_size_bytes (~2 GiB).
"""

import json
import resource
import struct
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from fsspec import AbstractFileSystem

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Safetensors dtype string → numpy dtype.
# Ported from levanter/compat/fsspec_safetensor.py lines 24-37.
_SAFETENSOR_DTYPE_MAP: Dict[str, np.dtype] = {
    "F16": np.dtype("float16"),
    "BF16": np.dtype(jnp.bfloat16),
    "F32": np.dtype("float32"),
    "F64": np.dtype("float64"),
    "I8": np.dtype("int8"),
    "I16": np.dtype("int16"),
    "I32": np.dtype("int32"),
    "I64": np.dtype("int64"),
    "U8": np.dtype("uint8"),
    "U16": np.dtype("uint16"),
    "U32": np.dtype("uint32"),
    "U64": np.dtype("uint64"),
    "BOOL": np.dtype("bool"),
}

DEFAULT_CHUNK_SIZE_BYTES = 2 * 1024**3  # 2 GiB


@dataclass(frozen=True)
class TensorRecord:
    """Per-tensor metadata and file location within a safetensors shard."""
    key: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    file_path: str
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class ChunkSpec:
    """A contiguous byte range grouping multiple tensors for a single read."""
    file_path: str
    byte_start: int
    byte_end: int
    tensors: Tuple[TensorRecord, ...]

    @property
    def size(self) -> int:
        return self.byte_end - self.byte_start


def _get_filesystem(path: str) -> AbstractFileSystem:
    """Return an fsspec filesystem for the given path."""
    protocol, _ = fsspec.core.split_protocol(path)
    if protocol is None:
        protocol = "file"
    if protocol == "gs" or protocol == "gcs":
        import gcsfs
        return gcsfs.GCSFileSystem()
    return fsspec.filesystem(protocol)


def _read_metadata(fs: AbstractFileSystem, path: str) -> Dict[str, TensorRecord]:
    """Parse the safetensors header to extract tensor metadata.

    Safetensors format:
    - 8 bytes: little-endian uint64 header length N
    - N bytes: UTF-8 JSON header of shapes/dtypes/data offsets
    - remaining bytes: raw tensor data blobs
    """
    header_len_bytes = fs.cat_file(path, start=0, end=8)
    (header_len,) = struct.unpack("<Q", header_len_bytes)
    metadata_bytes = fs.cat_file(path, start=8, end=8 + header_len)
    metadata = json.loads(metadata_bytes.decode("utf-8"))

    tensors: Dict[str, TensorRecord] = {}
    data_offset_base = 8 + header_len

    for key, meta in metadata.items():
        if key == "__metadata__":
            continue
        dtype_name: str = meta["dtype"]
        dtype = _SAFETENSOR_DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_name}")

        rel_start, rel_end = meta["data_offsets"]
        tensors[key] = TensorRecord(
            key=key,
            dtype=dtype,
            shape=tuple(meta["shape"]),
            file_path=path,
            byte_start=data_offset_base + rel_start,
            byte_end=data_offset_base + rel_end,
        )

    return tensors


def _build_chunks(tensors: Iterable[TensorRecord], chunk_limit: int) -> List[ChunkSpec]:
    """Group tensor records into download chunks respecting byte limit.

    Ported from levanter/compat/fsspec_safetensor.py lines 136-187.
    """
    if chunk_limit <= 0:
        raise ValueError("chunk_limit must be positive")

    sorted_records = sorted(tensors, key=lambda t: (t.file_path, t.byte_start))
    chunks: List[ChunkSpec] = []

    current: List[TensorRecord] = []
    current_start = 0
    current_end = 0
    current_path: Optional[str] = None

    for record in sorted_records:
        if not current:
            current = [record]
            current_start = record.byte_start
            current_end = record.byte_end
            current_path = record.file_path
            continue

        same_file = record.file_path == current_path
        proposed_end = max(current_end, record.byte_end)
        proposed_size = proposed_end - current_start

        if (not same_file) or (proposed_size > chunk_limit):
            chunks.append(
                ChunkSpec(
                    file_path=current_path or record.file_path,
                    byte_start=current_start,
                    byte_end=current_end,
                    tensors=tuple(current),
                )
            )
            current = [record]
            current_start = record.byte_start
            current_end = record.byte_end
            current_path = record.file_path
        else:
            current.append(record)
            current_end = proposed_end

    if current:
        chunks.append(
            ChunkSpec(
                file_path=current_path or sorted_records[-1].file_path,
                byte_start=current_start,
                byte_end=current_end,
                tensors=tuple(current),
            )
        )

    return chunks


def _discover_shards(fs: AbstractFileSystem, model_path: str) -> List[str]:
    """Find safetensors shard files for a model.

    Checks for model.safetensors.index.json first (multi-shard),
    then falls back to single model.safetensors file.
    """
    # Normalize path (strip trailing slash for consistency)
    model_path = model_path.rstrip("/")

    # Try multi-shard index first
    index_path = f"{model_path}/model.safetensors.index.json"
    try:
        index_bytes = fs.cat_file(index_path)
        index_data = json.loads(index_bytes.decode("utf-8"))
        weight_map = index_data.get("weight_map", {})
        # Get unique shard filenames in sorted order
        shard_files = sorted(set(weight_map.values()))
        full_paths = [f"{model_path}/{f}" for f in shard_files]
        logger.info("Discovered %d shards from index: %s", len(full_paths),
                     [f.rsplit("/", 1)[-1] for f in full_paths])
        return full_paths
    except FileNotFoundError:
        pass

    # Fall back to single file
    single_path = f"{model_path}/model.safetensors"
    try:
        fs.info(single_path)
        logger.info("Using single shard: model.safetensors")
        return [single_path]
    except FileNotFoundError:
        pass

    # Last resort: glob for any .safetensors files
    pattern = f"{model_path}/*.safetensors"
    found = sorted(fs.glob(pattern))
    if not found:
        raise FileNotFoundError(
            f"No safetensors files found at {model_path}")
    logger.info("Discovered %d safetensors files via glob", len(found))
    return found


def fsspec_weights_iterator(
    model_path: str,
    *,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> Iterator[Tuple[str, jax.Array]]:
    """Yield (name, jax.Array) pairs from remote safetensors via fsspec.

    Streams weights shard-by-shard, chunk-by-chunk sequentially.
    All arrays materialized on CPU. Peak host RAM ≈ chunk_size_bytes.

    Args:
        model_path: Path to model directory (gs://, local, etc.)
        chunk_size_bytes: Max bytes per download chunk (default 2 GiB)

    Yields:
        (tensor_name, jax_array) pairs with arrays on CPU device
    """
    cpu_device = jax.devices("cpu")[0]
    fs = _get_filesystem(model_path)
    shard_files = _discover_shards(fs, model_path)

    total_tensors = 0
    total_bytes = 0
    t_all = time.time()

    for shard_file in shard_files:
        t0 = time.time()
        records = _read_metadata(fs, shard_file)
        chunks = _build_chunks(records.values(), chunk_size_bytes)
        shard_bytes = 0
        shard_tensors = 0

        for chunk in chunks:
            raw = fs.cat_file(chunk.file_path,
                              start=chunk.byte_start, end=chunk.byte_end)
            with jax.default_device(cpu_device):
                for record in chunk.tensors:
                    offset = record.byte_start - chunk.byte_start
                    count = int(np.prod(record.shape, dtype=int))
                    arr = np.frombuffer(
                        raw, dtype=record.dtype,
                        count=count, offset=offset,
                    ).reshape(record.shape)
                    yield record.key, jnp.asarray(arr)
                    tensor_bytes = record.byte_end - record.byte_start
                    shard_bytes += tensor_bytes
                    shard_tensors += 1
            del raw  # free chunk memory

        elapsed = time.time() - t0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        shard_name = shard_file.rsplit("/", 1)[-1]
        logger.info(
            "Shard %s: %d tensors, %.1f GiB in %.1fs (%.0f MiB/s), "
            "peak RSS=%.0f MB",
            shard_name, shard_tensors, shard_bytes / 2**30, elapsed,
            (shard_bytes / 2**20) / max(elapsed, 0.001), peak_rss_mb)
        total_tensors += shard_tensors
        total_bytes += shard_bytes

    total_elapsed = time.time() - t_all
    logger.info(
        "All shards complete: %d tensors, %.1f GiB in %.1fs (%.0f MiB/s avg)",
        total_tensors, total_bytes / 2**30, total_elapsed,
        (total_bytes / 2**20) / max(total_elapsed, 0.001))
