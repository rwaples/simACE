# Installation

simACE runs in a locked [pixi](https://pixi.sh) environment (ADR 0016).
`pixi.toml` pins the supported pixi release in `requires-pixi`.

## Prerequisites

- Linux. On Windows, use
  [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install). macOS has no
  pipeline environment. To use simace on macOS, see
  [Use simace as a library](#use-simace-as-a-library).
- `git` and `curl`.

## Install pixi

pixi installs one binary under `~/.pixi/bin`. It does not need root. The
installer fetches the latest release by default, which `pixi.toml` refuses
once it moves past the `requires-pixi` range, so pin the version:

```bash
curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=v0.76.2 bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

The `export` puts pixi on `PATH` for the current shell. The installer also
appends the same line to `~/.bashrc` (or your shell's rc file) unless
`PIXI_NO_PATH_UPDATE` is set or the shell is one it does not recognise. If
`pixi --version` fails in a new terminal, add the `export` line to your rc
file yourself.

An existing install on the wrong version moves with:

```bash
pixi self-update --version v0.76.2
```

The pinned version and `requires-pixi` move together. Edit both, and the
`requires-pixi` line in the fitACE, pedigree-graph, and pedsum manifests;
`tests/test_pixi_pin_consistency.py` fails when they disagree.

## Install simACE

```bash
git clone https://github.com/rwaples/simACE.git
cd simACE
pixi install --locked
```

`pixi install --locked` builds the environment recorded in `pixi.lock`. The
first run downloads every package. Later runs reuse the cache.

To confirm that the install works, follow the [Quick start](quickstart.md).

## Run the development checks

Every check runs inside the pixi environment. The environment installs
`simace` in editable mode with the `dev` extras from `pyproject.toml`, which
include pytest, ruff, ty, and mkdocs among others.

```bash
pixi run pytest tests/
pixi run ruff check
pixi run ty check
pixi run mkdocs serve
```

`mkdocs serve` serves the docs at `http://127.0.0.1:8000` and reloads on edit.

## Update a dependency

Normal pixi commands never rewrite `pixi.lock`. To upgrade a dependency:

1. Edit the pin in `pixi.toml`.
2. Run `pixi lock`.
3. Review the diff of `pixi.lock`.
4. Run the development checks above.

## Use simace as a library

You do not need pixi to import simace from your own Python environment. pip
resolves every dependency from PyPI. This is also the supported path on macOS.

simace needs Python 3.14 or newer. Check with `python --version` first. On
Python 3.13, pip says simace requires a different Python. On 3.12 and older, it
fails on a dependency first and never names the Python version:
`No matching distribution found for pedigree-graph<0.13,>=0.12`.

```bash
pip install "simace @ git+https://github.com/rwaples/simACE"
```

To install a release rather than the default branch, append its tag, for
example `git+https://github.com/rwaples/simACE@v0.1.0`.

A pip install gives you the stage subcommands (`simace simulate`, `simace
cohort`, `simace analyze`, ...), which take explicit paths. `simace run`,
`simace show`, and `simace ls` read the scenario files in `config/`, so they
need a clone.

From a clone, run `pip install -e .` instead. Add `".[dev]"` to include the
development tools.
