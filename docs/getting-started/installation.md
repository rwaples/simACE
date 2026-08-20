# Installation

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Miniforge)
- Python 3.13+
- Linux or macOS (Windows users can try [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install))

## Environment setup

```bash
git clone <repo-url>
cd simACE
conda env create -f envs/environment.yml   # creates environment and installs simace
conda activate simACE
```

## Combined simACE + fitACE development (optional)

Model fitting lives in the private sister repo
[`fitACE`](https://github.com/rwaples/fitACE), which depends on simace. A
fresh simACE clone does not include it, so the base environment installs
simace only. If you have the `fitACE` checkout alongside simACE (at
`../fitACE` relative to `envs/`), layer it in:

```bash
conda env update -n simACE -f envs/environment-fitace.yml
```

## pixi: the canonical simACE environment (Linux)

On Linux, simACE work runs in a locked [pixi](https://pixi.sh) environment
(ADR 0016) covering the Snakemake pipeline and every development check. The
supported pixi release is pinned in `pixi.toml` (`requires-pixi`).

```bash
pixi install --locked          # materialize .pixi/ from the committed pixi.lock
pixi run pytest tests/
pixi run ruff check
pixi run ty check
pixi run snakemake --cores 4 results/test/small_test/scenario.done
```

Normal commands never rewrite `pixi.lock` — dependency upgrades are deliberate
lock-update work (edit `pixi.toml`, run `pixi lock`, review the diff, update
`envs/environment.yml` in the same commit, then re-run the checks above).

The conda environment remains the path for macOS, for combined simACE + fitACE
development, and for editable `pedigree-graph` work (see `RELEASE.md`).

## Verify installation

```bash
pytest tests/           # unit tests, should complete in ~1s
```

## Developer dependencies

The conda environment installs the developer dependencies from
`pyproject.toml`. For an existing environment, install them manually with:

```bash
pip install -e ".[dev]"
```

This adds: mkdocs, mkdocs-material, mkdocstrings, ruff, pytest, snakemake, and snakefmt.

## Building the docs locally

```bash
mkdocs serve       # live-reload at http://127.0.0.1:8000
mkdocs build       # static site in site/
```
