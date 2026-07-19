# tpu-inference (Marin fork)

@.agents/marin-style/AGENTS-core.md

The core standards above apply to the code Marin owns in this repo. This file
records what is different here, because this is not a normal Marin repo: it is a
fork that tracks an upstream project.

## Fork policy

This repo is a fork of [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference),
periodically refreshed from upstream. Marin's changes ride on top as a small
delta. Upstream owns almost every file; we own a handful.

**Minimize the upstream diff.** Every file we touch is a file the next upstream
refresh has to merge. Before changing an upstream file, ask whether the change
belongs upstream instead.

Rules that follow from that:

- **Never reformat or restyle upstream code.** Upstream's formatters are yapf +
  isort + ruff, configured in `.pre-commit-config.yaml`. Do not "fix" upstream
  style, do not run Marin's formatters over upstream files, and do not edit
  `.pre-commit-config.yaml`, `.buildkite/`, or `.github/workflows/` files that
  are not ours.
- Marin-owned paths: `infra/`, `.github/workflows/marin-*.yaml`, `.agents/`,
  `AGENTS.md`, `CLAUDE.md`, and the delta files listed below.
- Keep new code in new files where you can. A new module is a clean merge; a
  hunk in the middle of an upstream file is a conflict.
- A refresh must carry the Marin-owned paths forward. An earlier set of
  `.github/workflows/marin-*.yml` files (release and sync automation) is present
  in history but no longer on `main`, so this has been lost at least once. After
  a refresh, check that `marin-ci.yaml` and `marin-e2e-nightly.yaml` are still
  there.

### The Marin delta

Everything Marin adds on top of upstream, as of this writing:

| File | What it is |
| --- | --- |
| `tpu_inference/models/jax/grugmoe.py` | Native JAX implementation of Marin's GrugMoE model. |
| `tpu_inference/layers/vllm/__init__.py` | Registers GrugMoE with the TPU model loader (hunk in an upstream file). |
| `tpu_inference/runner/token_decision.py` | Registry for the in-process per-step token-decision callback. |
| `tpu_inference/runner/tpu_runner.py` | Calls that callback from the engine step (hunk in an upstream file). |
| `tests/models/jax/test_grugmoe.py` | GrugMoE behavior tests. Run on CPU. |
| `tests/models/common/test_model_loader.py` | GrugMoE loader/registration coverage. Needs Marin's vLLM fork. |
| `tpu_inference/utils.py` | Pads arbitrary GQA head counts for TPU tensor parallelism. |
| `tpu_inference/layers/common/attention_interface.py` | Pads and trims arbitrary-GQA attention projections. |
| `tpu_inference/layers/common/fused_moe_gmm.py` | Accepts custom routers' precomputed expert assignments. |
| `tpu_inference/layers/common/moe.py` | Types precomputed custom-router assignments. |
| `tpu_inference/layers/vllm/interface/moe.py` | Converts precomputed router outputs to JAX arrays. |
| `tpu_inference/layers/vllm/quantization/unquantized.py` | Uses the modular GMM path for custom routing. |
| `tpu_inference/layers/vllm/process_weights/cleanup_sharding.py` | Tolerates non-resizable parameter storage during sharding cleanup. |
| `tests/layers/vllm/test_unquantized.py` | Covers the custom-router modular GMM path. |
| `tests/layers/common/test_attention_interface.py` | Covers arbitrary-GQA padding and output trimming. |
| `tests/test_utils.py` | Covers arbitrary-GQA head padding. |
| `tests/layers/vllm/process_weights/test_cleanup_sharding.py` | Covers non-resizable parameter storage cleanup. |
| `requirements.txt` | TorchVision pinned to the pair Marin's vLLM stack resolves. |
| `setup.py` | Source installs report the selected release version. |

The arbitrary-GQA, precomputed-routing, and sharding-cleanup entries are generic
fixes intended for upstream. Retain them in the Marin delta only until their
upstream equivalents land.

To see the real delta rather than trusting this table:

```bash
git fetch upstream
git diff --stat "$(git merge-base upstream/main HEAD)"..HEAD
```

That diff also contains upstream's own release-branch cherry-picks, which are
upstream-owned even though they are not on `upstream/main`.

### Pairing with the vLLM fork

This fork is used together with Marin's vLLM fork: tpu-inference provides the TPU
backend, vLLM provides the engine, and the two are refreshed and pinned as a pair
from the Marin repo. A change here that needs a matching vLLM change is not
complete until both SHAs move together. `tests/models/common/test_model_loader.py`
depends on the vLLM fork, which is why it cannot run against stock PyPI vLLM.

## Development

Marin's lint entry point, scoped by `[tool.marin-style]` in `pyproject.toml` to
the delta files above:

```bash
uv run infra/pre-commit.py --all-files        # check
uv run infra/pre-commit.py --all-files --fix  # check and fix
```

It runs `ruff check` only — no formatter — so it cannot churn upstream style.
Upstream's own hooks (`pre-commit run`) still govern upstream files; leave them
to upstream.

Re-vendor the shared standards after bumping the pinned `marin-style` rev in
`infra/pre-commit.py`:

```bash
uvx --from git+https://github.com/marin-community/marin-style@<rev> marin-style sync
```

## Tests

Tests live in `tests/`, mirroring the `tpu_inference/` package layout.

**Most of this suite needs a real TPU.** Upstream runs it in Docker on TPU
runners via `.buildkite/`, which GitHub-hosted CI cannot do: no accelerator, and
`libtpu` is not installable there. Do not add TPU-dependent tests to the Marin
GitHub workflows.

What does run without a TPU is the GrugMoE model test, against the CPU subset of
the stack (`infra/cpu-test-requirements.txt`, which is what PR CI installs):

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -r infra/cpu-test-requirements.txt
uv run --no-project python -m pytest tests/models/jax/test_grugmoe.py
```

Two things to know before adding to that set:

- Importing *anything* under `tpu_inference.` pulls in vLLM, because
  `tpu_inference/__init__.py` does. Even an import smoke needs vLLM installed.
- That is stock PyPI vLLM, not Marin's fork. `tests/models/common/test_model_loader.py`
  imports a quantization module only the fork ships, so it cannot run on CPU CI;
  the nightly covers it on the real pinned pair.

## CI

Two Marin workflows, both prefixed `marin-` to keep them clearly ours:

- `.github/workflows/marin-ci.yaml` — per-PR, CPU-only, minutes. Delta-scoped
  lint, a vendored-standards drift check, and the CPU tests above. It does not
  invoke upstream's buildkite matrix or upstream's pre-commit hooks.
- `.github/workflows/marin-e2e-nightly.yaml` — nightly TPU end-to-end (11:00
  UTC, plus `workflow_dispatch`). Provisions a v5litepod-8 through Iris, installs
  Marin's pinned vLLM fork plus this repo at the commit under test, serves a
  model, probes the endpoint, and gates on `infra/nightly/serving-spec.json`.
  The slice is torn down in an `if: always()` step.

The gate spec is recorded from real hardware, not guessed: dispatch the nightly
with `record: true` and it prints a fresh spec to the job log for you to commit.
Re-record it when the model, the slice type, or the prompt set changes — a floor
recorded against a different setup is not a floor.

The nightly is the only thing that exercises this repo on real hardware. If you
change the runner, the model, or the serving path, expect PR CI to stay green and
the nightly to be the test that tells you the truth.

Upstream's own `pre-commit` workflow also runs on PRs here. It is currently red
for reasons that predate the Marin CI: upstream's yapf wants to reformat the
delta files, which were never run through it. Do not "fix" that by reformatting
them in this repo — the delta is replayed from a Marin overlay on every refresh,
so the drift would come straight back. It has to be fixed at the overlay source.
