#!/usr/bin/env python
"""Run ``ty check`` across every Python repo in the simACE/fitACE family.

Run from anywhere (repo paths come from :mod:`family_repos`, the one tracked
manifest).  For each present repo the helper runs ``ty`` from that repo's root so
the repo's own ``[tool.ty]`` / ``ty.toml`` config (Python version, ``extra-paths``
for the import-hook editable installs, source roots) applies.

**Hard-zero policy.**  A single ``ty check ... --error-on-warning`` runs per repo
and the *exit code is authoritative*: ``0`` means clean, anything non-zero fails
the sweep.  ``--error-on-warning`` makes ty exit non-zero on *any* finding -- an
error or a warning -- so a warning-severity false positive can't slip through a
zero exit.  The whole family is kept at zero findings; a new one (e.g. after a
``ty`` pin bump or a library-stub change) must be cleared with a specific
``# ty: ignore[rule]`` suppression (whose rule code is enforced by
``tests/test_ty_suppressions_coded.py``).

The diagnostic *label* -- ``DRIFT`` (an ``unresolved-import``, the high-value
cross-repo / cross-module signature-drift class) vs ``ADVISORY`` (a library-stub
false positive) -- is parsed from the output for the human only; it never gates.
``--verbose`` prints every finding; ``--repo <label> ...`` limits the sweep.

Exit code is non-zero iff any present repo has findings or ty itself failed.
Missing optional repos are skipped, not failed.

Examples:
    python tools/typecheck_family.py                 # sweep the whole family
    python tools/typecheck_family.py --verbose       # also print advisory findings
    python tools/typecheck_family.py --repo simACE fitACE
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from family_repos import ROOT, python_repos

if TYPE_CHECKING:
    from pathlib import Path

#: The high-value drift rule, used only to *label* a failure (DRIFT vs ADVISORY).
BLOCKER_RULE = "unresolved-import"


def _is_repo(path: Path) -> bool:
    """A configured ty target: has a ``pyproject.toml`` or a ``ty.toml``."""
    return (path / "pyproject.toml").is_file() or (path / "ty.toml").is_file()


def _ty(repo: Path, python: str) -> subprocess.CompletedProcess[str]:
    """Run the hard-zero ``ty check`` in *repo*; capture output, never raise."""
    return subprocess.run(
        [
            "ty",
            "check",
            "--python",
            python,
            "--output-format",
            "concise",
            "--color",
            "never",
            "--error-on-warning",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def check_repo(repo: Path, python: str, verbose: bool) -> tuple[str, int]:
    """Type-check one repo with a single hard-zero ``ty check``.

    Returns ``(status, n_findings)``.  Status is ``"ok"`` (exit 0, clean),
    ``"DRIFT"`` (exit 1 with an ``unresolved-import`` among the findings),
    ``"ADVISORY"`` (exit 1, other findings only), or ``"ty-error"`` (ty itself
    failed -- exit not in ``{0, 1}``).  The block decision is the *exit code*;
    the findings are parsed only to count and label them.
    """
    proc = _ty(repo, python)
    if proc.returncode not in (0, 1):
        print(f"    ty failed (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()}")
        return "ty-error", 0
    if proc.returncode == 0:
        return "ok", 0
    # Non-zero (with --error-on-warning): at least one finding. Parse to count
    # and label -- never to decide pass/fail (the exit code already did that).
    findings = [ln for ln in (proc.stdout + proc.stderr).splitlines() if "error[" in ln or "warning[" in ln]
    is_drift = any(f"[{BLOCKER_RULE}]" in ln for ln in findings)
    if verbose or is_drift:
        for ln in findings:
            print(f"    {ln}")
    status = "DRIFT" if is_drift else "ADVISORY"
    # Exit code is authoritative, so report at least one finding even if a future
    # concise-output change means the parse above matched none.
    return status, max(len(findings), 1)


def main() -> int:
    """Sweep the family, print the per-repo hard-zero result, return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--python",
        default=sys.prefix,
        help="Python environment/interpreter ty should resolve deps from (default: active env).",
    )
    parser.add_argument("--verbose", action="store_true", help="Also print advisory (library-stub) findings.")
    parser.add_argument("--repo", nargs="+", metavar="LABEL", help="Limit to these repo labels (e.g. simACE fitACE).")
    args = parser.parse_args()

    if shutil.which("ty") is None:
        print("ERROR: `ty` is not on PATH. Install the typecheck extra (ty==0.0.51).", file=sys.stderr)
        return 2

    repos = python_repos()
    if args.repo:
        wanted = set(args.repo)
        unknown = wanted - {r.label for r in repos}
        if unknown:
            print(f"ERROR: unknown repo label(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        repos = tuple(r for r in repos if r.label in wanted)

    failed = False
    print("ty hard-zero sweep — every finding (DRIFT or ADVISORY) fails the repo\n")
    for repo in repos:
        path = ROOT / repo.path
        if not path.is_dir() or not _is_repo(path):
            print(f"  skip      {repo.label}  (not present / no ty config)")
            continue
        status, n_findings = check_repo(path, args.python, args.verbose)
        if status != "ok":
            failed = True
        note = "clean" if status == "ok" else f"{n_findings} finding(s)"
        print(f"  {status:<9} {repo.label}  ({note})")

    print()
    if failed:
        print("FAILED — drift, advisory findings, or ty errors above (the family must stay at zero).")
        return 1
    print("All repos clean — zero ty findings family-wide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
