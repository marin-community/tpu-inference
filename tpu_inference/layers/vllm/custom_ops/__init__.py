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

from tpu_inference.layers.vllm.custom_ops import embedding  # noqa: F401
from tpu_inference.layers.vllm.custom_ops import fused_moe  # noqa: F401
from tpu_inference.layers.vllm.custom_ops import linear  # noqa: F401
from tpu_inference.layers.vllm.custom_ops import mla_attention  # noqa: F401


# NOTE: this empty function exists for an entry_points target for vllm plugin.
def register_custom_ops():
    pass
