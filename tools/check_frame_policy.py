"""Structural migration check: pandas is confined to a documented allowlist.

ADR 0015 makes polars the primary DataFrame library. Pandas survives only
where a third party forces it. This script scans **all** tracked Python in a
repo (not just the package) and fails on any pandas import or conversion
outside the allowlist below.

Run over one repo::

    python tools/check_frame_policy.py [REPO_ROOT ...]

Exit status is non-zero if any unallowed pandas use is found. Residual
pandas-index idioms are reported separately as warnings — every retained
occurrence should be allowlisted deliberately rather than emulated by
accident.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Allowed pandas sites, as (path glob, reason). A file matching any entry may
# import pandas; everything else may not.
ALLOWLIST: tuple[tuple[str, str], ...] = (
    ("simace/plotting/plot_validation.py", "seaborn boundary: stripplot renderers take frames"),
    ("workflow/scripts/simace/tskit/*.py", "pandas-native tstrait/tskit scripts (out of scope)"),
    ("tests/core/test_null_contract.py", "pandas round-trip compatibility coverage"),
    ("tests/core/test_pedigree_arrays.py", "PedigreeArrays.from_frame is structurally dual-frame"),
    ("tests/core/test_schema.py", "asserts the pandas-rejection TypeError"),
    ("tests/core/test_parquet.py", "asserts the pandas-rejection TypeError"),
    ("tests/core/test_trait_schema.py", "asserts the pandas-rejection TypeError"),
    ("tests/phenotype/test_blended_post.py", "asserts the pandas-rejection TypeError"),
    ("tests/ascertainment/test_run_ascertainment.py", "asserts the pandas-rejection TypeError"),
    ("tests/analysis/test_stats_sampling.py", "asserts the pandas-rejection TypeError"),
    ("tests/analysis/test_effective_size.py", "feeds a pandas fixture to external pedigree_graph"),
    ("tests/tskit/*.py", "covers the pandas-native tskit scripts"),
    (
        "tests/core/test_parquet_to_tsv.py",
        "pandas is the byte-for-byte reference oracle for the R-facing TSV rendering contract",
    ),
    # fitACE family boundaries
    ("fitace/kinship/export.py", "LDAK/sparseREML byte contract: polars cannot reproduce the bytes"),
    ("fitace/plotting/*.py", "seaborn boundary (only where a renderer takes a frame)"),
    # pedigree-graph: the package is frame-library-neutral; pandas lives only
    # in its test extra, for coverage that is *about* pandas compatibility.
    ("tests/test_frame_inputs.py", "focused pandas compatibility surface (incl. nullable ints)"),
    (
        "tests/test_pedigree_graph.py",
        "legacy golden reference derives pairs via pandas — a deliberately independent toolchain",
    ),
)

_PANDAS_IMPORT = re.compile(r"^(?P<indent>[ \t]*)(?:import\s+pandas|from\s+pandas\s+import)", re.MULTILINE)
# .to_pandas() is the sanctioned one-way conversion at a seaborn/third-party
# edge; pd.DataFrame(...) construction outside the allowlist is not.
_INDEX_IDIOMS = re.compile(r"\.(set_index|reset_index|reindex)\(|\.(loc|iloc)\[|\.index\b")


def _is_type_checking_only(text: str, match: re.Match[str]) -> bool:
    """True when this pandas import sits inside an ``if TYPE_CHECKING:`` block.

    Such an import creates no runtime dependency — it only types an
    annotation — so it is reported separately from real pandas use.
    """
    if not match.group("indent"):
        return False
    before = text[: match.start()]
    for line in reversed(before.splitlines()):
        if not line.strip():
            continue
        # Walk back to the nearest enclosing block header at lower indent.
        indent = len(line) - len(line.lstrip())
        if indent < len(match.group("indent")):
            return line.strip().startswith("if TYPE_CHECKING")
    return False


def _repo_label(root: Path) -> str:
    """Name the repo, not the checkout directory.

    Family repos are migrated in per-repo worktrees that often share a branch
    name, so ``root.name`` alone renders eight identical ``polars-wave2``
    headers. Resolve the shared git dir instead, which points at the real
    repository even from a linked worktree.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return root.name
    common = Path(out.stdout.strip())
    if not common.is_absolute():
        common = (root / common).resolve()
    return common.parent.name or root.name


def _tracked_python(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [root / line for line in out.stdout.splitlines() if line]


def _allowed(rel: str) -> str | None:
    for pattern, reason in ALLOWLIST:
        if Path(rel).match(pattern) or rel == pattern:
            return reason
    return None


def check(root: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (violations, annotation-only imports, index-idiom warnings)."""
    violations: list[str] = []
    annotations: list[str] = []
    warnings: list[str] = []
    for path in _tracked_python(root):
        rel = str(path.relative_to(root))
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in _PANDAS_IMPORT.finditer(text):
            if _allowed(rel) is not None:
                break
            if _is_type_checking_only(text, match):
                annotations.append(rel)
            else:
                violations.append(f"{rel}: imports pandas at runtime, outside the allowlist")
            break
        for match in _INDEX_IDIOMS.finditer(text):
            line = text[: match.start()].count("\n") + 1
            warnings.append(f"{rel}:{line}: pandas-style index idiom {match.group(0)!r}")
    return violations, annotations, warnings


def main() -> int:
    """Check every repo root given on the command line; return the exit status."""
    roots = [Path(a).resolve() for a in sys.argv[1:]] or [Path.cwd()]
    failed = False
    for root in roots:
        violations, annotations, warnings = check(root)
        print(f"=== {_repo_label(root)} ===")
        for v in violations:
            print(f"  FAIL {v}")
        for a in annotations:
            print(f"  ANNOTATION-ONLY (no runtime dependency) {a}")
        if warnings:
            print(f"  ({len(warnings)} residual index idioms — review, they may be polars/numpy)")
        if violations:
            failed = True
        else:
            print("  pandas confined to the allowlist at runtime")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
