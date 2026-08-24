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
"""Print Marin's pinned vLLM fork revision."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def resolve_vllm_revision(path: Path) -> str:
    source = path.read_text()
    match = re.search(
        r'^VLLM_FORK_REQUIREMENT\s*=\s*"[^"\n]+@([0-9a-f]{40})"$',
        source,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not find Marin's pinned vLLM revision in {path}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(resolve_vllm_revision(args.path))


if __name__ == "__main__":
    main()
