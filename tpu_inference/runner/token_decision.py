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
"""Registry for an in-process per-step token decision callback (Marin
joint-decode overlay).

The joint-decode worker registers a callback before its first engine step.
When registered, the TPU runner computes top-k logits per scheduled request,
invokes the callback with (req_ids, topk_by_request_id), and rewrites the
returned rows to one-hot before native sampling. Rows absent from the
returned map are left to native sampling. Exceptions propagate through the
engine step. The callback runs on the runner's execution thread and blocks
the step until it returns."""

from typing import Callable, Dict, List, Optional, Union

TokenDecisionFn = Callable[[List[str], Dict[str, List[Dict[str, Union[int, float]]]]],
                           Dict[str, int]]

CALLBACK: Optional[TokenDecisionFn] = None
TOP_K: int = 0


def register(callback: TokenDecisionFn, *, top_k: int) -> None:
    global CALLBACK, TOP_K
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    CALLBACK = callback
    TOP_K = top_k
