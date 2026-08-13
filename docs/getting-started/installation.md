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

## Optional: uv development environment

For simACE-only development checks (pytest, Ruff, ty) you can use
[uv](https://docs.astral.sh/uv/) with the committed `uv.lock` instead of conda.
Conda remains the reference environment for Snakemake pipelines and all
scientific execution — the uv environment covers only the edit–test loop.
The supported uv release is pinned in `pyproject.toml`
(`tool.uv.required-version`).

```bash
uv sync --locked --extra test --extra plot --extra typecheck
uv run --no-sync pytest tests/
uv run --no-sync ruff check
uv run --no-sync ty check
```

`uv sync --locked` validates `uv.lock` and refreshes the root `.venv/`;
`uv run --no-sync` keeps individual commands from re-syncing or narrowing the
installed extras. Normal commands never rewrite `uv.lock` — dependency
upgrades are deliberate lock-update work (`uv lock --upgrade`, review the
`uv.lock` diff, then re-run the four commands above).

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
