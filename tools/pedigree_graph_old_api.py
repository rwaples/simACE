#!/usr/bin/env python
"""Inventory of pedigree-graph 0.7.1 API use across the simACE family.

pedigree-graph 0.8.0 removes the 0.7.1 surface. Slice 8 of the 0.8.0 migration
updates every consumer in one commit per checkout, so it needs a list of what to
update. This tool produces that list, and
``tests/test_pedigree_graph_old_api_guard.py`` compares a fresh scan against the
committed snapshot so a new 0.7.1 consumer cannot land before slice 8 runs. Tool
and snapshot are both deleted in slice 8, once a regenerated inventory is empty
in production code.

Detection is regex only. Nothing here imports the code it reads, which is what
lets one scanner cover ``.py``, ``.pyi``, ``.sh``, ``.smk``, ``.bash`` and
``Snakefile`` alike. That reach matters: a live ``PAIR_KINSHIP`` import sits
inside a ``python -c`` string in ``scripts/verify/verify_simace_epimight.sh``.
Each repo's file list comes from ``git ls-files``, so the scan sees tracked
sources and nothing else. ``PUBLIC_0_7_1`` is ``pedigree_graph.__all__``
verbatim at tag v0.7.1. Import statements sit outside ``RULES`` because a
parenthesized import spans lines while the rules are searched one line at a time.

Consequences of scanning text rather than parsing it:

- Outside import statements only ``PAIR_KINSHIP``, ``REL_REGISTRY`` and
  ``compute_all_ne`` are matched. The other 22 public names of 0.7.1 are
  inventoried at their import sites alone, because the ``ne_*`` names double as
  dict, YAML and JSON string keys family wide, and a whole-word rule for them
  reports far more string literals than API uses.
- ``context`` is a line-level heuristic, not a parser. A line is ``doc`` when it
  reads as a comment or carries reST markup, and ``code`` otherwise. Docstring
  prose naming the old API still has to be rewritten in slice 8, so both kinds
  are inventoried; the field only says which sort of edit a line needs.
- Rules marked ``low_confidence`` fire on an attribute name whose receiver may
  not be a ``PedigreeGraph`` at all. They are inventoried for triage, not as
  confirmed uses.

Usage::

    python tools/pedigree_graph_old_api.py scan --out tools/pedigree_graph_old_api_inventory.json
    python tools/pedigree_graph_old_api.py check --snapshot tools/pedigree_graph_old_api_inventory.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from family_repos import ROOT, checkout_repos

PUBLIC_0_7_1: tuple[str, ...] = (
    "PAIR_KINSHIP",
    "REL_REGISTRY",
    "CohortWindow",
    "FrameLike",
    "GenerationInterval",
    "NeCaballeroToroResult",
    "NeCoancestryResult",
    "NeHillResult",
    "NeInbreedingResult",
    "NeIndividualDeltaFResult",
    "NeLTCResult",
    "NeSexRatioResult",
    "NeVarianceResult",
    "PedigreeGraph",
    "RelType",
    "compute_all_ne",
    "eligible_cohort_range",
    "ne_caballero_toro",
    "ne_coancestry",
    "ne_hill_overlapping",
    "ne_inbreeding",
    "ne_individual_delta_f",
    "ne_long_term_contributions",
    "ne_sex_ratio",
    "ne_variance_family_size",
)


@dataclass(frozen=True)
class Rule:
    """One regex probe for a single 0.7.1 symbol."""

    symbol: str
    """Name reported for a match, and part of the inventory key."""
    kind: str
    """``constructor``, ``classmethod``, ``method``, ``attribute``, ``private`` or ``public_name``."""
    pattern: re.Pattern[str]
    """Searched against each source line."""
    file_gated: bool = False
    """Applied only in files whose text mentions ``pedigree_graph`` or ``PedigreeGraph``."""
    low_confidence: bool = False
    """The receiver may not be a ``PedigreeGraph``, so the match needs manual triage."""
    scan_args: bool = False
    """Record which of the tracked keyword arguments the call passes."""


_METHOD_SYMBOLS: tuple[str, ...] = (
    "extract_pairs",
    "sibling_pairs",
    "count_pairs_streaming",
    "count_pairs",
    "compute_pair_kinship",
    "kinship_matrix",
    "compute_inbreeding",
    "per_gen_mean_kinship",
    "compute_n_ancestors",
    "compute_n_descendants",
)

_AMBIGUOUS_ATTRIBUTES: tuple[str, ...] = ("generation", "mother", "father", "twin", "sex", "birth_year", "n")

_PRIVATE_MEMBERS: tuple[str, ...] = (
    "_Am",
    "_Af",
    "_A5",
    "_A4",
    "_A3",
    "_A2",
    "_A",
    "_kinship_cache",
    "_depth",
    "_ids",
    "_orig_mother",
    "_orig_father",
    "_sample_mask",
    "_subsample_remap",
    "_subsample_inverse",
    "_inbreeding",
    "_n_ancestors",
    "_n_descendants",
    "_get_Ak",
    "_ensure_parent_csr",
    "_release_pair_matrices",
)

_PUBLIC_NAME_SYMBOLS: tuple[str, ...] = ("PAIR_KINSHIP", "REL_REGISTRY", "compute_all_ne")

RULES: tuple[Rule, ...] = (
    Rule("PedigreeGraph_call", "constructor", re.compile(r"\bPedigreeGraph\s*\(")),
    Rule("PedigreeGraph.from_dataframe", "classmethod", re.compile(r"\bPedigreeGraph\.from_dataframe\s*\(")),
    Rule("PedigreeGraph.from_subsample", "classmethod", re.compile(r"\bPedigreeGraph\.from_subsample\s*\(")),
    Rule(
        "PedigreeGraph.from_arrays",
        "classmethod",
        re.compile(r"\bPedigreeGraph\.from_arrays\s*\("),
        scan_args=True,
    ),
    *(Rule(name, "method", re.compile(rf"\.{name}\s*\("), scan_args=True) for name in _METHOD_SYMBOLS),
    Rule("generation_interval", "attribute", re.compile(r"\.generation_interval\b")),
    *(
        Rule(name, "attribute", re.compile(rf"\.{name}\b"), file_gated=True, low_confidence=True)
        for name in _AMBIGUOUS_ATTRIBUTES
    ),
    *(Rule(name, "private", re.compile(rf"\.{name}\b"), file_gated=True) for name in _PRIVATE_MEMBERS),
    Rule("_compute_depth", "private", re.compile(r"\b_compute_depth\b")),
    # The other 22 public names are import-site only: the ne_* names are dict/YAML/JSON string keys family wide.
    *(Rule(name, "public_name", re.compile(rf"\b{name}\b")) for name in _PUBLIC_NAME_SYMBOLS),
)

_SCHEMA = 1
_TOOL = "tools/pedigree_graph_old_api.py"
_SELF = Path(__file__).resolve()
_PURPOSE = "Every pedigree-graph 0.7.1 API use in the family, so slice 8 of the 0.8.0 migration can update them all."

_FROM_IMPORT = re.compile(r"from\s+pedigree_graph((?:\.\w+)+)?\s+import\s+")
_MODULE_IMPORT = re.compile(r"import\s+pedigree_graph((?:\.\w+)+)?\b")
_PUBLIC_NAME_RE = re.compile(r"\b(" + "|".join(PUBLIC_0_7_1) + r")\b")
_FILE_GATE = re.compile(r"pedigree_graph|PedigreeGraph")

_SCANNED_SUFFIXES = frozenset({".py", ".pyi", ".sh", ".smk", ".bash"})
_SCANNED_NAMES = frozenset({"Snakefile"})
_SPHINX_ROLES = (":meth:", ":func:", ":class:", ":attr:", ":mod:")
_ARG_KEYWORDS = ("max_degree", "min_kinship", "scope")
_FROM_ARRAYS_KEYWORDS = (*_ARG_KEYWORDS, "fathers", "mothers", "twins")
_ARG_SPAN_LIMIT = 2000
_MAX_IMPORT_LINES = 20
_TEXT_LIMIT = 160
_GIT_TIMEOUT = 60


def _tracked_files(repo_root: Path) -> list[str] | None:
    """Tracked paths in one repo, or ``None`` when the repo cannot be listed."""
    # git ls-files is the deny list: .snakemake, results/, .pixi/, build/ and the nested checkouts, unmaintained.
    if not repo_root.is_dir():
        return None
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [name for name in done.stdout.split("\0") if name]


def _is_scanned(rel: str) -> bool:
    """Whether a tracked path is one of the file kinds the scanner reads."""
    path = PurePosixPath(rel)
    return path.name in _SCANNED_NAMES or path.suffix in _SCANNED_SUFFIXES


def _is_test(rel: str) -> bool:
    """Whether a tracked path is test code rather than production code."""
    path = PurePosixPath(rel)
    if "tests" in path.parts or "test" in path.parts:
        return True
    return path.name == "conftest.py" or (path.name.startswith("test_") and path.suffix == ".py")


def _line_of(text: str, offset: int) -> int:
    """One-based line number of a character offset in ``text``."""
    return text.count("\n", 0, offset) + 1


def _context(line: str) -> str:
    """Classify a source line as ``doc`` prose or ``code``."""
    prose = line.strip().startswith("#") or "``" in line or any(role in line for role in _SPHINX_ROLES)
    return "doc" if prose else "code"


def _call_flags(symbol: str, text: str, after_open_paren: int) -> tuple[str, ...]:
    """Tracked keyword arguments passed inside one call's own parentheses."""
    keywords = _FROM_ARRAYS_KEYWORDS if symbol == "PedigreeGraph.from_arrays" else _ARG_KEYWORDS
    limit = min(len(text), after_open_paren + _ARG_SPAN_LIMIT)
    end, depth = after_open_paren, 1
    while end < limit and depth:
        char = text[end]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        end += 1
    return tuple(sorted(k for k in keywords if f"{k}=" in text[after_open_paren:end]))


def _import_region(text: str, start: int) -> str:
    """The imported-name text after ``import``, continued while a ``(`` stays open."""
    end = text.find("\n", start)
    end = len(text) if end == -1 else end
    for _ in range(_MAX_IMPORT_LINES):
        if end >= len(text) or text.count("(", start, end) <= text.count(")", start, end):
            break
        nxt = text.find("\n", end + 1)
        end = len(text) if nxt == -1 else nxt
    return text[start:end]


def _import_hits(text: str) -> list[tuple[int, str, str]]:
    """``(line, symbol, kind)`` for every ``pedigree_graph`` import in one file."""
    hits: list[tuple[int, str, str]] = []
    for match in _FROM_IMPORT.finditer(text):
        submodule = match.group(1)
        if submodule:
            hits.append((_line_of(text, match.start()), f"pedigree_graph{submodule}", "private_import"))
            continue
        region = _import_region(text, match.end())
        hits.extend(
            (_line_of(text, match.end() + name.start()), name.group(0), "import")
            for name in _PUBLIC_NAME_RE.finditer(region)
        )
    for match in _MODULE_IMPORT.finditer(text):
        submodule = match.group(1)
        line = _line_of(text, match.start())
        if submodule:
            hits.append((line, f"pedigree_graph{submodule}", "private_import"))
        else:
            hits.append((line, "pedigree_graph", "import_module"))
    return hits


def _rule_hits(text: str) -> list[tuple[int, str, str, tuple[str, ...], bool]]:
    """``(line, symbol, kind, flags, low_confidence)`` for every rule match in one file."""
    gated = _FILE_GATE.search(text) is not None
    hits: list[tuple[int, str, str, tuple[str, ...], bool]] = []
    offset = 0
    for line, raw in enumerate(text.split("\n"), 1):
        for rule in RULES:
            if rule.file_gated and not gated:
                continue
            match = rule.pattern.search(raw)
            if match is None:
                continue
            flags = _call_flags(rule.symbol, text, offset + match.end()) if rule.scan_args else ()
            hits.append((line, rule.symbol, rule.kind, flags, rule.low_confidence))
        offset += len(raw) + 1
    return hits


def _scan_file(repo: str, rel: str, text: str) -> list[dict]:
    """Every inventory entry in one file, one per ``(line, symbol)``."""
    hits: list[tuple[int, str, str, tuple[str, ...], bool]] = [
        (line, symbol, kind, (), False) for line, symbol, kind in _import_hits(text)
    ]
    hits.extend(_rule_hits(text))
    lines = text.split("\n")
    seen: set[tuple[int, str]] = set()
    entries: list[dict] = []
    for line, symbol, kind, flags, low_confidence in hits:
        if (line, symbol) in seen:
            continue
        seen.add((line, symbol))
        source = lines[line - 1]
        stripped = source.strip()
        entry: dict = {
            "repo": repo,
            "path": rel,
            "line": line,
            "symbol": symbol,
            "kind": kind,
            "context": _context(source),
            "text": stripped if len(stripped) <= _TEXT_LIMIT else stripped[:_TEXT_LIMIT] + "…",
        }
        if flags:
            entry["flags"] = list(flags)
        if low_confidence:
            entry["low_confidence"] = True
        entries.append(entry)
    return entries


def _sort_key(entry: dict) -> tuple[str, str, int, str, str]:
    """Stable ordering for the two entry lists."""
    return (entry["repo"], entry["path"], entry["line"], entry["symbol"], entry["kind"])


def _entry_key(entry: dict) -> tuple[str, str, str]:
    """The identity a guard run compares against the snapshot."""
    return (entry["repo"], entry["path"], entry["symbol"])


def _counts(entries: list[dict], repos: list[str]) -> dict:
    """Per-repo entry totals, listing every scanned repo including the empty ones."""
    counts = dict.fromkeys(sorted(repos), 0)
    for entry in entries:
        counts[entry["repo"]] += 1
    return counts


def scan(root: Path) -> dict:
    """Scan the family checkouts under ``root`` and build the inventory report."""
    scanned: list[str] = []
    unavailable: list[str] = []
    production: list[dict] = []
    tests: list[dict] = []
    for repo in checkout_repos():
        if repo.label == "pedigree-graph":
            continue
        repo_root = root / repo.path
        files = _tracked_files(repo_root)
        if files is None:
            unavailable.append(repo.label)
            continue
        scanned.append(repo.label)
        for rel in files:
            source = repo_root / rel
            # This scanner names every old symbol in its own rule table, so it
            # would inventory itself the moment it is tracked.
            if not _is_scanned(rel) or source.resolve() == _SELF:
                continue
            try:
                text = source.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            bucket = tests if _is_test(rel) else production
            bucket.extend(_scan_file(repo.label, rel, text))
    production.sort(key=_sort_key)
    tests.sort(key=_sort_key)
    return {
        "schema": _SCHEMA,
        "tool": _TOOL,
        "purpose": _PURPOSE,
        "repos_scanned": sorted(scanned),
        "repos_unavailable": sorted(unavailable),
        "counts": {"production": _counts(production, scanned), "tests": _counts(tests, scanned)},
        "production": production,
        "tests": tests,
    }


def new_entries(report: dict, snapshot: dict) -> list[dict]:
    """Report entries whose ``(repo, path, symbol)`` key is absent from the snapshot."""
    known = {_entry_key(entry) for entry in snapshot["production"] + snapshot["tests"]}
    return [entry for entry in report["production"] + report["tests"] if _entry_key(entry) not in known]


def main(argv: list[str] | None = None) -> int:
    """Run ``scan`` or ``check`` and return the process exit status."""
    parser = argparse.ArgumentParser(description="Inventory pedigree-graph 0.7.1 API use across the family.")
    commands = parser.add_subparsers(dest="command", required=True)
    scanner = commands.add_parser("scan", help="write the inventory as JSON")
    scanner.add_argument("--out", type=Path, help="write here instead of stdout")
    scanner.add_argument("--root", type=Path, default=ROOT, help="family umbrella root")
    checker = commands.add_parser("check", help="fail when a new old-API use appeared")
    checker.add_argument("--snapshot", type=Path, required=True, help="committed inventory to compare against")
    checker.add_argument("--root", type=Path, default=ROOT, help="family umbrella root")
    args = parser.parse_args(argv)

    report = scan(args.root)
    if args.command == "scan":
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out is None:
            sys.stdout.write(rendered)
        else:
            args.out.write_text(rendered, encoding="utf-8")
        return 0

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    live = {_entry_key(entry) for entry in report["production"] + report["tests"]}
    gone = {_entry_key(entry) for entry in snapshot["production"] + snapshot["tests"]} - live
    if gone:
        print(f"{len(gone)} inventoried uses are gone; regenerate the snapshot to drop them.")
    added = new_entries(report, snapshot)
    for entry in added:
        print(f"{entry['repo']}/{entry['path']}:{entry['line']}: {entry['symbol']} ({entry['kind']})")
        print(f"    {entry['text']}")
    if added:
        print(f"{len(added)} new pedigree-graph 0.7.1 uses. Migrate them, or regenerate the snapshot with:")
        print(f"    python {_TOOL} scan --out tools/pedigree_graph_old_api_inventory.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
