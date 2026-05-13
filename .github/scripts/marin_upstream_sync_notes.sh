#!/usr/bin/env bash
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

set -euo pipefail

metadata_file=${1:?metadata file is required}
headline=${2:-"MERGED WITH UPSTREAM MAIN"}

if [[ ! -f "$metadata_file" ]]; then
  cat <<EOF
## Upstream Sync

**UPSTREAM SYNC METADATA MISSING**

No upstream sync metadata file was present when this release was created.
EOF
  exit 0
fi

status=$(jq -r '.status // "recorded"' "$metadata_file")

if [[ "$status" != "recorded" ]]; then
  message=$(jq -r '.message // "No upstream sync metadata was present when this artifact was validated."' "$metadata_file")
  cat <<EOF
## Upstream Sync

**UPSTREAM SYNC METADATA MISSING**

${message}
EOF
  exit 0
fi

upstream_repo=$(jq -r '.upstream_repo // "unknown"' "$metadata_file")
upstream_branch=$(jq -r '.upstream_branch // "main"' "$metadata_file")
upstream_sha=$(jq -r '.upstream_sha // "unknown"' "$metadata_file")
upstream_short_sha=$(jq -r '.upstream_short_sha // "unknown"' "$metadata_file")
synced_at=$(jq -r '.synced_at_utc // "unknown"' "$metadata_file")
issue_url=$(jq -r '.sync_issue_url // empty' "$metadata_file")

cat <<EOF
## Upstream Sync

**${headline}**

Upstream branch: \`${upstream_repo}@${upstream_branch}\`
Upstream commit: \`${upstream_sha}\` (${upstream_short_sha})
Synced at: \`${synced_at}\`
EOF

if [[ -n "$issue_url" ]]; then
  printf 'Tracking issue: %s\n' "$issue_url"
fi
