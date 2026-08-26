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

from io import BytesIO
from urllib.error import URLError
from urllib.request import Request

import pytest

import probe
import serve_and_probe

VLLM_REQUIREMENT = "vllm @ https://example.invalid/vllm.whl"
TPU_INFERENCE_REV = "2" * 40
GATE_SPEC = {
    "gate": {
        "min_completions": 8,
        "min_output_tokens_per_second": 200.0,
    }
}


def test_physical_tpu_type_reads_gcp_instance_metadata(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: Request, **_: object) -> BytesIO:
        assert request.get_header("Metadata-flavor") == "Google"
        return BytesIO(b"v6e-8\n")

    monkeypatch.setattr(serve_and_probe.urllib.request, "urlopen", urlopen)

    assert serve_and_probe.physical_tpu_type() == "v6e-8"


def test_physical_tpu_type_rejects_empty_metadata(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        serve_and_probe.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(),
    )

    with pytest.raises(RuntimeError, match="empty physical TPU type"):
        serve_and_probe.physical_tpu_type()


def test_physical_tpu_type_propagates_metadata_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise URLError("unavailable")

    monkeypatch.setattr(serve_and_probe.urllib.request, "urlopen", fail)

    with pytest.raises(URLError, match="unavailable"):
        serve_and_probe.physical_tpu_type()


def test_serve_command_uses_marin_release_and_head_override() -> None:
    command = serve_and_probe.serve_command(
        model="Qwen/Qwen3-0.6B",
        vllm_requirement=VLLM_REQUIREMENT,
        tpu_inference_rev=TPU_INFERENCE_REV,
        tensor_parallel_size=8,
    )

    assert VLLM_REQUIREMENT in command
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
        vllm_requirement=VLLM_REQUIREMENT,
        tpu_inference_rev=TPU_INFERENCE_REV,
    )

    recorded = probe.record_spec(observed, "Qwen/Qwen3-0.6B", provenance)

    assert recorded["provenance"]["tpu"] == "v6e-8"
    assert recorded["provenance"]["vllm_requirement"] == VLLM_REQUIREMENT
    assert recorded["provenance"][
        "tpu_inference_rev"] == TPU_INFERENCE_REV
    assert recorded["gate"]["min_completions"] == len(probe.PROMPTS)
    assert recorded["gate"]["min_output_tokens_per_second"] == 50.0
    assert recorded["observed"]["output_tokens_per_second"] == 200.0
