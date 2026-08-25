# 0016 — pixi is simACE's canonical environment; conda is demoted to the family environment

Date: 2026-08-14

Status: accepted, **amended by
[ADR 0018](0018-retire-conda-family-environment.md)** (2026-08-21). The
pixi-is-canonical-for-simACE decision stands and is unchanged. What no longer
applies is the *other half* of this ADR — the conda env's survival as the
family environment: `envs/` was deleted outright (all three recipes, including
`environment.yml`), the same-commit pin-sync rule retired with it, and the
family environment moved to `fitACE/pixi.toml` (ADR 0017). The macOS-via-conda
path in Decision and Consequences below is likewise dropped — the pipeline is
Linux/WSL2 only. See ADR 0018 for what replaced each dependent.

## Context

simACE work ran in one shared conda env that also hosts fitACE and the seven
method sisters (editable installs, R, native toolchains). A uv pilot
(2026-08-13) gave simACE a locked dev loop but could not cover Snakemake or the
scientific stack, leaving three environment systems (conda YAMLs, uv.lock,
pip editables) with drift between them.

A pixi 0.76 spike (plans/pixi-spike-findings.md) showed one pixi manifest can
cover everything conda does for simACE — conda-forge + bioconda + PyPI in a
single committed lock — with the full suite, golden digests, and a Snakemake
scenario all green, and cold env materialization in ~31s. The enabling change
was pedigree-graph's move to PyPI (range dep replaces the git-URL pin that
caused editable-override conflicts).

## Decision

- pixi is the canonical environment for every simACE-scoped command: pytest,
  ruff, ty, snakefmt, and Snakemake pipeline runs (`pixi run …`).
- The conda env remains alive and maintained solely as the **family
  environment**: fitACE and sister work, editable pedigree-graph development,
  and family-scoped tools (`tools/typecheck_family.py`). `envs/environment.yml`
  stays, because `environment-fitace.yml` layers on it.
- `pixi.toml` is the authoritative pin source for simACE; dependency bumps
  update `envs/environment.yml` in the same commit (sync rule) until fitACE
  migrates.
- The uv pilot retires completely (config, lock, docs, allowlist, binary).
- Manifest shape: single default environment, no task aliases, `requires-pixi`
  pinned to the tested series, linux-64 only (conda still covers macOS).
- Migration gate: a repeated-run (N≥3) conda-vs-pixi pipeline comparison;
  >5% median wall-clock or peak-RSS regression per stage blocks adoption.

## Consequences

- One committed lockfile governs simACE correctness *and* pipeline execution;
  "works on my machine" drift between dev checks and pipeline runs ends.
- Two environments still exist on a family machine (pixi + conda), but with a
  scope rule instead of overlapping claims: simACE = pixi, family = conda.
- The pin sync rule is manual until fitACE migrates; a future fitACE-layer
  pixi spike (features/environments, R, sisters' editables) decides whether
  the conda env retires entirely.
- macOS users rely on the conda path until someone verifies an osx lock.
