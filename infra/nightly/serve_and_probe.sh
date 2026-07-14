#!/usr/bin/env bash
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

# Serve the model under test and probe it. This runs INSIDE the nightly's Iris job on
# the TPU worker -- not on the GitHub runner -- from the bundled repo checkout. See
# .github/workflows/marin-nightly.yaml.
#
# Inputs, passed by the nightly as Iris job env vars:
#   MODEL                 HF model id to serve
#   VLLM_REV              Marin's vLLM fork SHA (whatever marin currently pins)
#   TPU_INFERENCE_REV     this repo's commit -- the code under test
#   TENSOR_PARALLEL_SIZE  vLLM tensor-parallel size for this slice
#   TPU                   slice type, recorded as provenance
#   RECORD                "1" to rewrite the gate spec from this run instead of gating
set -euo pipefail

: "${MODEL:?}" "${VLLM_REV:?}" "${TPU_INFERENCE_REV:?}" "${TENSOR_PARALLEL_SIZE:?}"

export VLLM_TARGET_DEVICE=tpu

# The serving environment mirrors marin-core's own isolated TPU-vLLM path
# (IsolatedTpuVllm: uvx, the vLLM fork, --torch-backend cpu, VLLM_TARGET_DEVICE=tpu),
# with one substitution: tpu-inference comes from the commit under test rather than
# from marin's pin. That substitution is the entire point of this nightly.
uvx --from "vllm @ git+https://github.com/marin-community/vllm.git@${VLLM_REV}" \
    --with "tpu-inference @ git+https://github.com/marin-community/tpu-inference.git@${TPU_INFERENCE_REV}" \
    --python 3.12 --torch-backend cpu \
    vllm serve "${MODEL}" \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-num-batched-tokens 512 \
    --dtype bfloat16 &
server=$!
trap 'kill "${server}" 2>/dev/null || true' EXIT

record_flag=()
if [ "${RECORD:-0}" = "1" ]; then
  record_flag=(--record)
fi

# The probe waits for the server, warms it up (the first request compiles the model),
# then gates a timed batch against the spec. Its exit code is the job's exit code.
python3 infra/nightly/probe.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model "${MODEL}" \
  --spec infra/nightly/serving-spec.json \
  --tpu "${TPU:-}" \
  --vllm-rev "${VLLM_REV}" \
  --tpu-inference-rev "${TPU_INFERENCE_REV}" \
  "${record_flag[@]}"
