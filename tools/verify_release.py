#!/usr/bin/env python
r"""Assert that every lockstep family artifact reports one release version.

Run in the fitACE pixi env, which installs all nine family distributions:

    pixi run --manifest-path fitACE/pixi.toml python tools/verify_release.py 0.1.0
    pixi run --manifest-path fitACE/pixi.toml python tools/verify_release.py 0.1.0 \
        --provenance results/test/small_test/rep*/params.yaml ...

Checks each family import package's ``__version__``, every family console
script's ``--version``, both ``ace_iter_reml`` builds' ``--version``, and every
``*_version`` key in each ``--provenance`` file (``params.yaml``, ``run.yaml``,
``*.vc.tsv.meta``).  The binary stamps the raw tag (``v0.1.0``), so a leading
``v`` is dropped before comparing.  Prints one line per check; exits 1 if any
check fails.  RELEASE.md §4 runs it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DISTRIBUTIONS = (
    "simace",
    "fitace",
    "fitace-epimight",
    "fitace-pcgc",
    "fitace-iter-reml",
    "fitace-tetraher",
    "fitace-pafgrs",
    "fitace-stan",
    "fitace-frailty",
)
BINARY_BUILDS = ("build-fp64", "build-fp32")
_BINARY_DIR = ROOT / "fitACE" / "fitACE_iter_reml" / "ace_iter_reml"
_VERSION_KEY = re.compile(r"""^\s*(\w+_version)(?:\t|:\s*)['"]?([^\s'"]+)""", re.MULTILINE)


def _report(label: str, observed: str, version: str) -> bool:
    ok = observed.removeprefix("v") == version
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {observed}")
    return ok


def _last_token(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"<{exc.strerror}>"
    tokens = result.stdout.split()
    return tokens[-1] if result.returncode == 0 and tokens else f"<exit {result.returncode}>"


def _package_version(pkg: str) -> str:
    try:
        return import_module(pkg).__version__
    except (ImportError, AttributeError) as exc:
        return f"<{type(exc).__name__}: {exc}>"


def check_installed(version: str) -> list[bool]:
    """Every family import package and console script in this environment; a missing distribution fails."""
    bindir = Path(sys.executable).parent
    results = []
    for name in DISTRIBUTIONS:
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            results.append(_report(f"{name} distribution", "<not installed>", version))
            continue
        packages = (dist.read_text("top_level.txt") or "").split()
        scripts = dist.entry_points.select(group="console_scripts")
        results.extend(_report(f"{pkg}.__version__", _package_version(pkg), version) for pkg in packages)
        results.extend(
            _report(f"{ep.name} --version", _last_token([str(bindir / ep.name), "--version"]), version)
            for ep in scripts
        )
    return results


def check_binaries(version: str) -> list[bool]:
    """Both ``ace_iter_reml`` builds."""
    return [
        _report(
            f"{build}/ace_iter_reml --version",
            _last_token([str(_BINARY_DIR / build / "ace_iter_reml"), "--version"]),
            version,
        )
        for build in BINARY_BUILDS
    ]


def check_provenance(version: str, paths: list[Path]) -> list[bool]:
    """Every ``*_version`` key in each file; a missing file, or one with no key, fails."""
    results = []
    for path in paths:
        try:
            text = path.read_text()
        except OSError as exc:
            results.append(_report(str(path), f"<{exc.strerror}>", version))
            continue
        stamps = _VERSION_KEY.findall(text)
        if not stamps:
            print(f"FAIL {path}: no *_version key")
            results.append(False)
        results.extend(_report(f"{path} {key}", value, version) for key, value in stamps)
    return results


def main(argv: list[str] | None = None) -> int:
    """Run every check for the expected version; exit 1 if any fails."""
    parser = argparse.ArgumentParser(prog="verify_release.py", description=__doc__.splitlines()[0])
    parser.add_argument("version", help="Expected family version without the v, e.g. 0.1.0")
    parser.add_argument("--provenance", nargs="+", type=Path, default=[], metavar="FILE")
    args = parser.parse_args(argv)

    results = check_installed(args.version) + check_binaries(args.version)
    results += check_provenance(args.version, args.provenance)
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks report {args.version}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
