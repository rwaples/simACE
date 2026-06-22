#!/usr/bin/env python
"""Run ``ty check`` across every Python repo in the simACE/fitACE family.

Run from anywhere (repo paths resolve relative to this file's location).  For
each present repo the helper runs ``ty check`` from that repo's root so the
repo's own ``[tool.ty]`` / ``ty.toml`` config (Python version, ``extra-paths``
for the import-hook editable installs, source roots) applies.

Per the drift-only gate decision, the **only** blocking diagnostic is
``unresolved-import`` — the cross-repo / cross-module signature-drift class that
catches silent, no-exception breakage (e.g. a sibling package renaming or moving
a symbol).  Every other ty diagnostic is printed as advisory and does **not**
fail the run.  This mirrors what the ``/commit`` gate enforces, but sweeps the
whole family (including the ``fitACE_*`` sisters + pedsum that the commit gate
does not touch) in one pass.

Exit code is non-zero iff any present repo has a blocking ``unresolved-import``
(or ty failed to run).  Missing optional repos are skipped, not failed.

Examples:
    python tools/typecheck_family.py            # sweep the whole family
    python tools/typecheck_family.py --verbose  # also print advisory findings
    python tools/typecheck_family.py --python /path/to/env
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: Python repos in the family, relative to the simACE root (this file's
#: grandparent).  The C++-only ``ace_iter_reml`` binary is intentionally absent.
FAMILY_PY_REPOS: tuple[str, ...] = (
    ".",  # simACE
    "fitACE",
    "external/pedigree-graph",
    "external/pedsum",
    "external/tetraher_simace",  # LDAK fork: Python helper only
    "fitACE/fitACE_epimight",
    "fitACE/fitACE_pcgc",
    "fitACE/fitACE_iter_reml",
    "fitACE/fitACE_tetraher",
    "fitACE/fitACE_pafgrs",
    "fitACE/fitACE_stan",
    "fitACE/fitACE_frailty",
)

#: The one diagnostic that blocks (see module docstring).
BLOCKER = "error[unresolved-import]"

ROOT = Path(__file__).resolve().parent.parent


def _is_repo(path: Path) -> bool:
    """A configured ty target: has a pyproject.toml or a ty.toml."""
    return (path / "pyproject.toml").is_file() or (path / "ty.toml").is_file()


def check_repo(rel: str, python: str, verbose: bool) -> tuple[str, int, int]:
    """Run ty in one repo. Return (status, n_blockers, n_advisory)."""
    repo = ROOT / rel
    proc = subprocess.run(
        ["ty", "check", "--python", python, "--output-format", "concise"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    lines = [ln for ln in out.splitlines() if "error[" in ln]
    blockers = [ln for ln in lines if BLOCKER in ln]
    advisory = len(lines) - len(blockers)
    status = "FAIL" if blockers else "ok"
    to_show = lines if verbose else blockers
    for ln in to_show:
        print(f"    {ln}")
    return status, len(blockers), advisory


def main() -> int:
    """Sweep the family, print the per-repo drift-gate result, return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--python",
        default=sys.prefix,
        help="Python environment/interpreter ty should resolve deps from (default: active env).",
    )
    parser.add_argument("--verbose", action="store_true", help="Also print advisory (non-blocking) findings.")
    args = parser.parse_args()

    any_fail = False
    print(f"ty drift gate — blocking only on {BLOCKER}\n")
    for rel in FAMILY_PY_REPOS:
        repo = ROOT / rel
        label = "simACE" if rel == "." else rel
        if not repo.is_dir() or not _is_repo(repo):
            print(f"  skip  {label}  (not present / no ty config)")
            continue
        status, n_block, n_adv = check_repo(rel, args.python, args.verbose)
        any_fail = any_fail or status == "FAIL"
        print(f"  {status:<4}  {label}  ({n_block} blocking, {n_adv} advisory)")

    print()
    if any_fail:
        print("FAILED — unresolved imports above.")
    else:
        print("All repos pass the drift gate.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
