# Installation

simACE runs in a locked [pixi](https://pixi.sh) environment (ADR 0016): one
committed lockfile, so every install materializes the exact same environment.
The supported pixi release is pinned in `pixi.toml` (`requires-pixi`).

## Prerequisites

- Linux (Windows users can use
  [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install); for macOS see
  the conda fallback below)
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

# 4. Smoke-test the whole pipeline (~30 s)
pixi run snakemake --cores 4 results/test/small_test/scenario.done
```

## Run a real scenario

For a single full scenario — simulate → phenotype → censor → ascertain →
analyze → stats → plot atlas — target its `stats.done` plus the atlas
(`scenario.done` additionally pulls in the folder-wide report gather, which
wants the sibling scenarios built too):

```bash
pixi run snakemake -n      --cores 4 results/base/baseline100K/stats.done results/base/baseline100K/plots/atlas.html
pixi run snakemake --cores 4 results/base/baseline100K/stats.done results/base/baseline100K/plots/atlas.html
```

Outputs land under `results/base/baseline100K/`: per-replicate parquet + YAML
in `rep{1..3}/`, plots and the browsable `atlas.html` in `plots/`. Scenario
parameters live in `config/base.yaml` (per-scenario overrides) and
`config/_default.yaml` (defaults); the stock run needs no edits.

## Development checks

Every check runs in the same pixi environment (the editable `simace` install
includes the dev extras — pytest, ruff, ty, snakefmt, mkdocs):

```bash
pixi run pytest tests/
pixi run ruff check
pixi run ty check
pixi run mkdocs serve          # docs live-reload at http://127.0.0.1:8000
```

Normal commands never rewrite `pixi.lock` — dependency upgrades are deliberate
lock-update work (edit `pixi.toml`, run `pixi lock`, review the diff, update
`envs/environment.yml` in the same commit, then re-run the checks above).

## Using simace as a library

You do not need pixi (or conda) to *consume* simace from your own
environment — a plain pip install resolves every dependency from PyPI:

```bash
pip install "simace @ git+https://github.com/rwaples/simACE"
# or, from a clone: pip install -e .          (add ".[dev]" for the dev tools)
```

## Conda fallback (macOS; family development)

The pixi lock currently covers `linux-64` only. On macOS, use the conda
environment instead:

```bash
conda env create -f envs/environment.yml   # creates environment and installs simace
conda activate simACE
```

Model fitting lives in the private [`fitACE`](https://github.com/rwaples/fitACE)
monorepo (ADR 0017), which depends on simace and is developed as a nested
checkout at `./fitACE/` — a fresh simACE clone does not include it. Combined
simACE + fitACE development uses the conda environment above as the family
base (see `RELEASE.md`); fitACE also carries its own committed pixi
environment at `fitACE/pixi.toml`.
