#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin to an exact marin-style revision so every contributor and CI run
#     # uses the same checks. Bump the rev to adopt a new version.
#     "marin-style @ git+https://github.com/marin-community/marin-style@ccbf03e7ca58486d61fff7a4e73031673d7fd8a4",
# ]
# ///
"""Consumer-repo pre-commit shim. Delegates to the shared marin-style checks.

Scoped to the Marin delta via `[tool.marin-style]` in pyproject.toml; upstream
tpu-inference code keeps its own yapf/isort pre-commit matrix untouched.
"""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
