#!/usr/bin/env python3
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
"""Serve the model under test on the TPU slice, probe it, and always tear it down.

Runs INSIDE the nightly's Iris job on the TPU worker -- not on the GitHub runner --
from the bundled repo checkout. See .github/workflows/marin-e2e-nightly.yaml. Its
exit code is the job's exit code, which is the nightly's result.

Stdlib only: the worker has no third-party packages, and vLLM itself lives in the
ephemeral uvx environment this script builds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import probe

VLLM_FORK = "https://github.com/marin-community/vllm.git"
TPU_INFERENCE_FORK = "https://github.com/marin-community/tpu-inference.git"

HOST = "127.0.0.1"
PORT = 8000
SPEC = Path(__file__).parent / "serving-spec.json"

# Marin's serving defaults for a TPU slice (marin-core's quick_serve).
MAX_NUM_BATCHED_TOKENS = 512
DTYPE = "bfloat16"

SHUTDOWN_GRACE = 30.0


def serve_command(model: str, vllm_rev: str, tpu_inference_rev: str,
                  tensor_parallel_size: int) -> list[str]:
    """Build the command that starts vLLM on the slice.

    This mirrors marin-core's own isolated TPU-vLLM environment (IsolatedTpuVllm:
    uvx, the vLLM fork, --torch-backend cpu, VLLM_TARGET_DEVICE=tpu) with one
    substitution: tpu-inference comes from the commit under test rather than from
    Marin's pin. That substitution is the entire point of this nightly.
    """
    return [
        "uvx",
        "--from",
        f"vllm @ git+{VLLM_FORK}@{vllm_rev}",
        "--with",
        f"tpu-inference @ git+{TPU_INFERENCE_FORK}@{tpu_inference_rev}",
        "--python",
        "3.12",
        "--torch-backend",
        "cpu",
        "vllm",
        "serve",
        model,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--dtype",
        DTYPE,
    ]


def stop(server: subprocess.Popen) -> None:
    """Stop the server, escalating to a kill if it will not go quietly."""
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=SHUTDOWN_GRACE)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-rev",
                        required=True,
                        help="Marin's vLLM fork SHA")
    parser.add_argument("--tpu-inference-rev",
                        required=True,
                        help="This repo's commit -- the code under test")
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    parser.add_argument("--tpu", default="", help="Slice type, for provenance")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Rewrite the gate spec from this run instead of gating")
    args = parser.parse_args()

    command = serve_command(args.model, args.vllm_rev, args.tpu_inference_rev,
                            args.tensor_parallel_size)
    print(f"serving: {' '.join(command)}", flush=True)

    # vLLM's logs stream to the job log, which is how a failed serve gets diagnosed.
    server = subprocess.Popen(command,
                              env={
                                  **os.environ, "VLLM_TARGET_DEVICE": "tpu"
                              },
                              stdout=sys.stdout,
                              stderr=sys.stderr)
    try:
        return probe.run(
            base_url=f"http://{HOST}:{PORT}/v1",
            model=args.model,
            spec_path=SPEC,
            provenance=probe.Provenance(
                tpu=args.tpu,
                vllm_rev=args.vllm_rev,
                tpu_inference_rev=args.tpu_inference_rev),
            record=args.record,
            is_alive=lambda: server.poll() is None,
        )
    finally:
        stop(server)


if __name__ == "__main__":
    raise SystemExit(main())
