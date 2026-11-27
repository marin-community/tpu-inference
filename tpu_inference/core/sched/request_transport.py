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

import copy
from dataclasses import replace

from vllm.v1.request import Request


def request_for_ipc(request: Request) -> Request:
    """Copy structured requests without process-local grammar compilation state."""
    if request.structured_output_request is None:
        return request

    # EngineCore starts grammar compilation before DP dispatch. Its Future
    # owns thread locks, and compiled grammars/reasoners belong to that process.
    # Keep the constraints and reasoning state; the rank compiles its own copy.
    result = copy.copy(request)
    result.structured_output_request = replace(
        request.structured_output_request, _grammar=None, reasoner=None)
    return result
