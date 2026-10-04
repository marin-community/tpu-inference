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
"""Probe a served model over its OpenAI-compatible API and gate the result.

Runs inside the nightly's Iris job, against the vLLM server on localhost, and
exits nonzero if the served model fails the gate in the spec file. Stdlib only:
the job serves from an ephemeral uvx environment, so this must run under a bare
`python3` with nothing installed.

    python3 probe.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-0.6B \
        --spec infra/nightly/serving-spec.json

`--record` rewrites the spec from the observed run instead of gating, which is how
the throughput floor gets seeded from real hardware.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# A small, fixed prompt set. These are a smoke of the serving path, not an eval:
# they only have to elicit a non-empty, bounded answer.
PROMPTS = (
    "Name the largest planet in our solar system.",
    "What is 17 + 25? Answer with the number only.",
    "Write one sentence about the ocean.",
    "Translate 'good morning' into French.",
    "List three primary colors.",
    "In one word, what is the capital of Japan?",
    "Finish the sequence: 2, 4, 6, 8,",
    "What sound does a cat make?",
)

MAX_TOKENS = 64
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 180.0

# --record sets the floor this far below the rate it observed.
FLOOR_MARGIN = 4
RECORDED_NOTE = "Recorded from a real serving run by infra/nightly/probe.py --record."

# The first request pays for XLA compilation, which on TPU runs into minutes. It is
# served before the timed batch and excluded from it, so the throughput number
# reflects steady-state serving rather than compile time.
WARMUP_TIMEOUT = 1800.0
READINESS_POLL_INTERVAL = 5.0


@dataclass(frozen=True)
class Provenance:
    """What was served, recorded into a spec so a floor can be traced to a run."""

    tpu: str
    vllm_rev: str
    tpu_inference_rev: str


@dataclass(frozen=True)
class Completion:
    """One model response and the tokens the server billed for it."""

    text: str
    output_tokens: int


@dataclass(frozen=True)
class Observed:
    """What the timed batch actually did."""

    completions: int
    empty_completions: int
    output_tokens: int
    elapsed: float

    @property
    def output_tokens_per_second(self) -> float:
        return self.output_tokens / self.elapsed


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def complete(base_url: str, model: str, prompt: str,
             timeout: float) -> Completion:
    """Send one chat completion and return the response text and token count."""
    body = _post_json(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        timeout,
    )
    text = body["choices"][0]["message"]["content"] or ""
    return Completion(text=text.strip(),
                      output_tokens=body["usage"]["completion_tokens"])


def wait_for_models(base_url: str,
                    deadline: float,
                    is_alive: Callable[[], bool] = lambda: True) -> None:
    """Block until the server lists its models.

    Raises if the deadline passes, or as soon as ``is_alive`` goes false -- a vLLM
    that died on startup should fail the run in seconds, not after the warmup
    timeout that a still-compiling one needs.
    """
    while True:
        if not is_alive():
            raise RuntimeError("the server exited before it began serving")
        try:
            with urllib.request.urlopen(f"{base_url}/models",
                                        timeout=10) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error: Exception = e

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"server did not serve {base_url}/models in time: {last_error}"
            )
        time.sleep(READINESS_POLL_INTERVAL)


def run_timed_batch(base_url: str, model: str) -> Observed:
    """Serve every prompt concurrently and measure the aggregate output rate."""
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        completions = list(
            pool.map(lambda p: complete(base_url, model, p, REQUEST_TIMEOUT),
                     PROMPTS))
    elapsed = time.monotonic() - start

    return Observed(
        completions=len(completions),
        empty_completions=sum(1 for c in completions if not c.text),
        output_tokens=sum(c.output_tokens for c in completions),
        elapsed=elapsed,
    )


def gate_failures(spec: dict, observed: Observed) -> list[str]:
    """Return the reasons the run fails the spec. Empty means it passed."""
    gate = spec["gate"]
    failures = []

    if observed.completions < gate["min_completions"]:
        failures.append(
            f"only {observed.completions} of {len(PROMPTS)} prompts completed "
            f"(need {gate['min_completions']})")
    if observed.empty_completions:
        failures.append(
            f"{observed.empty_completions} completions came back empty")

    floor = gate["min_output_tokens_per_second"]
    if observed.output_tokens_per_second < floor:
        failures.append(
            f"output throughput {observed.output_tokens_per_second:.1f} tok/s "
            f"is below the floor of {floor} tok/s")
    return failures


def record_spec(observed: Observed, model: str,
                provenance: Provenance) -> dict:
    """Return a gate spec built from this run's numbers, with provenance."""
    recorded = {
        "model": model,
        "tpu": provenance.tpu,
        "vllm_fork_rev": provenance.vllm_rev,
        "tpu_inference_rev": provenance.tpu_inference_rev,
        "prompts": len(PROMPTS),
        "max_tokens": MAX_TOKENS,
        "recorded_at": datetime.now(UTC).isoformat(),
        "note": RECORDED_NOTE,
    }
    rate = round(observed.output_tokens_per_second, 1)
    # The floor sits well under the observed rate: this gate catches a collapse (a CPU
    # fallback, a recompile every step), it does not track drift.
    floor = round(observed.output_tokens_per_second / FLOOR_MARGIN, 1)
    gate = {
        "min_completions": len(PROMPTS),
        "min_output_tokens_per_second": floor,
    }
    measured = {
        "output_tokens_per_second": rate,
        "output_tokens": observed.output_tokens,
        "elapsed": round(observed.elapsed, 1),
    }
    return {"provenance": recorded, "gate": gate, "observed": measured}


def run(base_url: str,
        model: str,
        spec_path: Path,
        provenance: Provenance,
        record: bool = False,
        is_alive: Callable[[], bool] = lambda: True) -> int:
    """Probe a served model and gate it against the spec. Returns an exit code."""
    base_url = base_url.rstrip("/")
    spec = json.loads(spec_path.read_text())

    print(f"waiting for {base_url}/models", flush=True)
    wait_for_models(base_url,
                    deadline=time.monotonic() + WARMUP_TIMEOUT,
                    is_alive=is_alive)

    print("warming up (compiles the model; not timed)", flush=True)
    warmup = complete(base_url, model, PROMPTS[0], WARMUP_TIMEOUT)
    print(f"warmup returned {warmup.output_tokens} tokens", flush=True)

    observed = run_timed_batch(base_url, model)
    print(
        f"served {observed.completions} prompts, {observed.output_tokens} output "
        f"tokens in {observed.elapsed:.1f}s = "
        f"{observed.output_tokens_per_second:.1f} tok/s",
        flush=True,
    )

    if record:
        recorded = json.dumps(record_spec(observed, model, provenance),
                              indent=2)
        spec_path.write_text(recorded + "\n")
        # Also to stdout: this runs on a TPU worker whose filesystem the workflow
        # cannot reach, so the streamed job log is how a spec gets back to a human.
        print(f"recorded {spec_path}:\n{recorded}", flush=True)
        return 0

    failures = gate_failures(spec, observed)
    for failure in failures:
        print(f"GATE FAILED: {failure}", flush=True)
    if failures:
        return 1

    print("gate passed", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",
                        required=True,
                        help="OpenAI-compatible root, e.g. .../v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--record",
                        action="store_true",
                        help="Rewrite the spec from this run")
    parser.add_argument("--tpu",
                        default="",
                        help="Slice type, recorded as provenance")
    parser.add_argument("--vllm-rev",
                        default="",
                        help="vLLM fork SHA, recorded as provenance")
    parser.add_argument("--tpu-inference-rev",
                        default="",
                        help="This repo's SHA, as provenance")
    args = parser.parse_args()

    provenance = Provenance(tpu=args.tpu,
                            vllm_rev=args.vllm_rev,
                            tpu_inference_rev=args.tpu_inference_rev)
    return run(base_url=args.base_url,
               model=args.model,
               spec_path=args.spec,
               provenance=provenance,
               record=args.record)


if __name__ == "__main__":
    raise SystemExit(main())
