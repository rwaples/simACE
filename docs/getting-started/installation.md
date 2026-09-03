# Installation

simACE runs in a locked [pixi](https://pixi.sh) environment (ADR 0016). One
committed lockfile means every install materializes the same environment.
The supported pixi release is pinned in `pixi.toml` (`requires-pixi`).

## Prerequisites

- Linux. Windows users can use
  [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install). For macOS see
  the library install below.
- `git` and `curl`

## Quick start

```bash
# 1. Install pixi (single user-space binary; no root beyond curl/git)
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL                    # restart the shell so ~/.pixi/bin is on PATH

# 2. Get simACE
git clone https://github.com/rwaples/simACE.git
cd simACE

# 3. Materialize the exact locked environment (first run downloads for a few minutes)
pixi install --locked

# 4. Smoke-test the whole pipeline (a minute or two)
pixi run snakemake --cores 4 results/test/small_test/scenario.done
```

## Run a real scenario

For a single full scenario (simulate → phenotype → censor → ascertain →
analyze → stats → plot atlas), target its `stats.done` plus the atlas.
`scenario.done` also pulls in the folder-wide report gather, which
wants the sibling scenarios built too:

```bash
pixi run snakemake -n      --cores 4 results/base/baseline100K/stats.done results/base/baseline100K/plots/atlas.html
pixi run snakemake --cores 4 results/base/baseline100K/stats.done results/base/baseline100K/plots/atlas.html
```

Outputs land under `results/base/baseline100K/`: per-replicate parquet and YAML
in `rep{1..3}/`, plots and the browsable `atlas.html` in `plots/`. Scenario
parameters live in `config/base.yaml` (per-scenario overrides) and
`config/_default.yaml` (defaults). The stock run needs no edits.

## Development checks

Every check runs in the same pixi environment. The editable `simace` install
includes the dev extras: pytest, ruff, ty, snakefmt, mkdocs.

```bash
pixi run pytest tests/
pixi run ruff check
pixi run ty check
pixi run mkdocs serve          # docs live-reload at http://127.0.0.1:8000
```

Normal commands never rewrite `pixi.lock`. Dependency upgrades are deliberate
lock-update work (edit `pixi.toml`, run `pixi lock`, review the diff, then
re-run the checks above).

## Using simace as a library

You do not need pixi to use simace from your own environment. A plain pip
install resolves every dependency from PyPI:

```bash
pip install "simace @ git+https://github.com/rwaples/simACE"
# or, from a clone: pip install -e .          (add ".[dev]" for the dev tools)
```

## Platform support

The pipeline environment is Linux-only (`linux-64`; ADR 0018). Windows
users can run it under WSL2. There is no macOS pipeline environment; on macOS
the plain-pip library install above is the supported path.

Model fitting lives in the private [`fitACE`](https://github.com/rwaples/fitACE)
monorepo (ADR 0017), which depends on simace and is developed as a nested
checkout at `./fitACE/`. A fresh simACE clone does not include it. Family
development runs in fitACE's own committed pixi environment
(`pixi run --manifest-path fitACE/pixi.toml …`; see `RELEASE.md`).
