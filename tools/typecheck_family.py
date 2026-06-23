#!/usr/bin/env python
"""Run ``ty check`` across every Python repo in the simACE/fitACE family.

Run from anywhere (repo paths come from :mod:`family_repos`, the one tracked
manifest).  For each present repo the helper runs ``ty`` from that repo's root so
the repo's own ``[tool.ty]`` / ``ty.toml`` config (Python version, ``extra-paths``
for the import-hook editable installs, source roots) applies.

Two things are enforced, per the rollout decisions:

* **Drift gate (blocking).**  The authoritative block decision is the *exit code*
  of ``ty check --ignore all --error unresolved-import`` -- ty 0.0.51's native
  wildcard severity, so we never parse stdout to decide.  A non-zero exit means
  the cross-repo / cross-module signature-drift class (a sibling package renaming
  or moving a symbol) that fails with no exception.
* **Advisory ratchet (blocking on regression only).**  A second plain ``ty check``
  enumerates the non-blocking findings; each repo's count is compared to its
  budget in ``tools/ty_budget.json``.  Exceeding the budget fails the sweep, so
  advisory findings can't silently re-accumulate.  ``--update-budget`` rewrites
  the budget from the current counts (a deliberate, reviewed step -- like bumping
  the ty pin).

Exit code is non-zero iff any present repo has an ``unresolved-import`` blocker,
ty itself failed, or a repo is over its advisory budget.  Missing optional repos
are skipped, not failed.

Examples:
    python tools/typecheck_family.py                 # sweep the whole family
    python tools/typecheck_family.py --verbose       # also print advisory findings
    python tools/typecheck_family.py --repo simACE fitACE
    python tools/typecheck_family.py --update-budget  # accept current advisory counts
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from family_repos import ROOT, python_repos

if TYPE_CHECKING:
    from pathlib import Path

#: The one diagnostic rule that blocks the gate (see module docstring).
BLOCKER_RULE = "unresolved-import"

#: Tracked per-repo advisory budget (label -> max tolerated advisory findings).
BUDGET_PATH = ROOT / "tools" / "ty_budget.json"


def _is_repo(path: Path) -> bool:
    """A configured ty target: has a ``pyproject.toml`` or a ``ty.toml``."""
    return (path / "pyproject.toml").is_file() or (path / "ty.toml").is_file()


def _ty(repo: Path, python: str, *extra: str) -> subprocess.CompletedProcess[str]:
    """Run ``ty check`` in *repo* with *extra* flags; capture output, never raise."""
    return subprocess.run(
        ["ty", "check", "--python", python, "--output-format", "concise", *extra],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def check_repo(repo: Path, python: str, verbose: bool) -> tuple[str, int, int]:
    """Type-check one repo.

    Returns ``(status, n_blockers, n_advisory)`` where status is ``"ok"``,
    ``"FAIL"`` (``unresolved-import`` drift) or ``"ty-error"`` (ty itself failed).
    The caller upgrades ``"ok"`` to ``"OVER"`` after the budget compare.
    """
    # 1. Authoritative drift gate: exit code, no stdout parsing.
    #    0 = clean, 1 = unresolved-import present, anything else = ty failure.
    gate = _ty(repo, python, "--ignore", "all", "--error", BLOCKER_RULE)
    if gate.returncode not in (0, 1):
        print(f"    ty failed (exit {gate.returncode}): {(gate.stderr or gate.stdout).strip()}")
        return "ty-error", 0, 0
    blocker_lines = [ln for ln in (gate.stdout + gate.stderr).splitlines() if f"error[{BLOCKER_RULE}]" in ln]
    if gate.returncode == 1:
        # Exit code is authoritative: report at least one blocker even if a
        # future concise-output change means the parse above finds none.
        n_blockers = len(blocker_lines) or 1
    else:
        n_blockers = 0

    # 2. Advisory enumeration: plain check, parsed only for the (non-blocking)
    #    report + budget.  Parsing here is low-stakes -- it never gates.
    plain = _ty(repo, python)
    if plain.returncode not in (0, 1):
        print(f"    ty failed (exit {plain.returncode}): {(plain.stderr or plain.stdout).strip()}")
        return "ty-error", n_blockers, 0
    diagnostics = [ln for ln in (plain.stdout + plain.stderr).splitlines() if "error[" in ln or "warning[" in ln]
    advisory_lines = [ln for ln in diagnostics if f"[{BLOCKER_RULE}]" not in ln]

    for ln in blocker_lines + (advisory_lines if verbose else []):
        print(f"    {ln}")
    status = "FAIL" if n_blockers else "ok"
    return status, n_blockers, len(advisory_lines)


def _load_budget() -> dict[str, int]:
    """Per-repo advisory budget from :data:`BUDGET_PATH` (empty if absent)."""
    if BUDGET_PATH.is_file():
        return json.loads(BUDGET_PATH.read_text())
    return {}


def _write_budget(values: dict[str, int]) -> None:
    """Write the budget in manifest order, appending any unrecognised labels."""
    ordered = {r.label: values[r.label] for r in python_repos() if r.label in values}
    ordered.update({k: v for k, v in values.items() if k not in ordered})
    BUDGET_PATH.write_text(json.dumps(ordered, indent=2) + "\n")


def main() -> int:
    """Sweep the family, print the per-repo gate + budget result, return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--python",
        default=sys.prefix,
        help="Python environment/interpreter ty should resolve deps from (default: active env).",
    )
    parser.add_argument("--verbose", action="store_true", help="Also print advisory (non-blocking) findings.")
    parser.add_argument("--repo", nargs="+", metavar="LABEL", help="Limit to these repo labels (e.g. simACE fitACE).")
    parser.add_argument(
        "--update-budget",
        action="store_true",
        help="Rewrite tools/ty_budget.json from the current advisory counts (still fails on drift / ty errors).",
    )
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

    budget = _load_budget()
    new_counts: dict[str, int] = {}
    hard_fail = False
    over_fail = False
    print(f"ty drift gate — blocking on error[{BLOCKER_RULE}]; advisory ratchet via {BUDGET_PATH.name}\n")
    for repo in repos:
        path = ROOT / repo.path
        if not path.is_dir() or not _is_repo(path):
            print(f"  skip     {repo.label}  (not present / no ty config)")
            continue
        status, n_block, n_adv = check_repo(path, args.python, args.verbose)
        new_counts[repo.label] = n_adv
        allowed = budget.get(repo.label, 0)
        if status in ("FAIL", "ty-error"):
            hard_fail = True
        elif status == "ok" and n_adv > allowed:
            status = "OVER"
            over_fail = True
        note = f"{n_block} blocking, {n_adv} advisory"
        if n_adv != allowed:
            note += f" (budget {allowed})"
        print(f"  {status:<8} {repo.label}  ({note})")

    if args.update_budget:
        _write_budget({**budget, **new_counts})
        print(f"\nWrote {BUDGET_PATH} ({len(new_counts)} repo(s) updated).")
        return 1 if hard_fail else 0

    print()
    if hard_fail or over_fail:
        print("FAILED — unresolved imports, ty errors, or advisory over budget above.")
        return 1
    print("All repos pass the drift gate and stay within advisory budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
