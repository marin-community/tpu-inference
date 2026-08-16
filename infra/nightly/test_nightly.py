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

from pathlib import Path

import pytest

from resolve_marin_vllm_rev import resolve_vllm_revision
from serve_and_probe import actual_tpu


def test_resolve_vllm_revision(tmp_path: Path) -> None:
    pin_file = tmp_path / "external_dependencies.py"
    revision = "1" * 40
    pin_file.write_text(
        'VLLM_FORK_REQUIREMENT = "vllm @ git+https://example.com/vllm.git@'
        f'{revision}"\n'
    )

    assert resolve_vllm_revision(pin_file) == revision


def test_resolve_vllm_revision_rejects_missing_pin(tmp_path: Path) -> None:
    pin_file = tmp_path / "external_dependencies.py"
    pin_file.write_text("OTHER_REQUIREMENT = 'package'\n")

    with pytest.raises(ValueError, match="Could not find"):
        resolve_vllm_revision(pin_file)


def test_actual_tpu_uses_worker_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "IRIS_WORKER_DEVICE",
        '{"tpu": {"variant": "v6e-8", "topology": "2x4"}}',
    )

    assert actual_tpu("v5litepod-8,v6e-8") == "v6e-8"


def test_actual_tpu_falls_back_to_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_WORKER_DEVICE", raising=False)

    assert actual_tpu("v5litepod-8,v6e-8") == "v5litepod-8,v6e-8"
