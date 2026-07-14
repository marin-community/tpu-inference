#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pin to an exact marin-style revision so every contributor and CI run
#     # uses the same checks. Bump the rev to adopt a new version.
#     "marin-style @ git+https://github.com/marin-community/marin-style@4469660ef1ba6ecd4c8d81bdbb17091bc501a682",
# ]
# ///
"""Consumer-repo pre-commit shim. Delegates to the shared marin-style checks.

Scoped to the Marin delta via `[tool.marin-style]` in pyproject.toml; upstream
tpu-inference code keeps its own yapf/isort pre-commit matrix untouched.
"""

from marin_style.precommit import main

if __name__ == "__main__":
    raise SystemExit(main())
