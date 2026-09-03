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

pixi installs one binary under `~/.pixi/bin`. It does not need root.

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL
```

The second command restarts your shell so that `~/.pixi/bin` is on `PATH`.

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
include pytest, ruff, ty, snakefmt, and mkdocs among others.

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

```bash
pip install "simace @ git+https://github.com/rwaples/simACE"
```

From a clone, run `pip install -e .` instead. Add `".[dev]"` to include the
development tools.
