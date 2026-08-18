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

from __future__ import annotations

import json
from pathlib import Path

import pytest

import probe
import resolve_marin_vllm_rev
import serve_and_probe

VLLM_REV = "1" * 40
TPU_INFERENCE_REV = "2" * 40
GATE_SPEC = {
    "gate": {
        "min_completions": 8,
        "min_output_tokens_per_second": 200.0,
    }
}


def test_resolve_vllm_revision_reads_canonical_pin(tmp_path: Path) -> None:
    dependency_file = tmp_path / "dependency.py"
    dependency_file.write_text(
        'VLLM_FORK_REQUIREMENT = "vllm @ '
        f'git+https://github.com/marin-community/vllm.git@{VLLM_REV}"\n')

    assert resolve_marin_vllm_rev.resolve_vllm_revision(
        dependency_file) == VLLM_REV


def test_resolve_vllm_revision_rejects_missing_pin(tmp_path: Path) -> None:
    dependency_file = tmp_path / "dependency.py"
    dependency_file.write_text("VLLM_FORK_REQUIREMENT = 'vllm'\n")

    with pytest.raises(ValueError):
        resolve_marin_vllm_rev.resolve_vllm_revision(dependency_file)


def test_placed_tpu_reads_iris_worker_device(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_WORKER_DEVICE",
                       json.dumps({"tpu": {
                           "variant": "v6e-8"
                       }}))

    assert serve_and_probe.placed_tpu() == "v6e-8"


@pytest.mark.parametrize("device", [{}, {"tpu": {}}, {"tpu": {"variant": ""}}])
def test_placed_tpu_rejects_missing_variant(
        monkeypatch: pytest.MonkeyPatch, device: dict[str, object]) -> None:
    monkeypatch.setenv("IRIS_WORKER_DEVICE", json.dumps(device))

    with pytest.raises(ValueError):
        serve_and_probe.placed_tpu()


def test_serve_command_pins_both_forks_and_slice_size() -> None:
    command = serve_and_probe.serve_command(
        model="Qwen/Qwen3-0.6B",
        vllm_rev=VLLM_REV,
        tpu_inference_rev=TPU_INFERENCE_REV,
        tensor_parallel_size=8,
    )

    assert (
        f"vllm @ git+{serve_and_probe.VLLM_FORK}@{VLLM_REV}" in command)
    assert (f"tpu-inference @ git+{serve_and_probe.TPU_INFERENCE_FORK}"
            f"@{TPU_INFERENCE_REV}" in command)
    assert command[command.index("--tensor-parallel-size") + 1] == "8"
    assert command[command.index("--dtype") + 1] == "bfloat16"


def test_gate_failures_accepts_healthy_batch() -> None:
    observed = probe.Observed(
        completions=8,
        empty_completions=0,
        output_tokens=512,
        elapsed=2.0,
    )

    assert probe.gate_failures(GATE_SPEC, observed) == []


@pytest.mark.parametrize(
    "observed",
    [
        probe.Observed(
            completions=7,
            empty_completions=0,
            output_tokens=400,
            elapsed=1.0,
        ),
        probe.Observed(
            completions=8,
            empty_completions=1,
            output_tokens=400,
            elapsed=1.0,
        ),
        probe.Observed(
            completions=8,
            empty_completions=0,
            output_tokens=100,
            elapsed=1.0,
        ),
    ],
)
def test_gate_failures_rejects_invalid_batch(observed: probe.Observed) -> None:
    assert probe.gate_failures(GATE_SPEC, observed)


def test_record_spec_preserves_provenance_and_sets_quarter_floor() -> None:
    observed = probe.Observed(
        completions=8,
        empty_completions=0,
        output_tokens=400,
        elapsed=2.0,
    )
    provenance = probe.Provenance(
        tpu="v6e-8",
        vllm_rev=VLLM_REV,
        tpu_inference_rev=TPU_INFERENCE_REV,
    )

    recorded = probe.record_spec(observed, "Qwen/Qwen3-0.6B", provenance)

    assert recorded["provenance"]["tpu"] == "v6e-8"
    assert recorded["provenance"]["vllm_fork_rev"] == VLLM_REV
    assert recorded["provenance"][
        "tpu_inference_rev"] == TPU_INFERENCE_REV
    assert recorded["gate"]["min_completions"] == len(probe.PROMPTS)
    assert recorded["gate"]["min_output_tokens_per_second"] == 50.0
    assert recorded["observed"]["output_tokens_per_second"] == 200.0
