"""Guardrail: no private name is imported across a package boundary (simACE#70).

fitACE ADR 0001 §5 and the ``GLOSSARY.md`` *Public surface* entry: code in one
family package imports only the public names of another, and a private name
another package needs is promoted rather than reached into.  Two boundaries
are checked over the import statements of every ``*.py`` file tracked in the
family checkouts:

- **Package**: a file in one family check unit (``tools/family_repos.py``)
  importing a private module or name from a package another unit owns.  Tests,
  tools, and scripts count; they belong to the unit whose directory holds them.
- **simace sub-package**: a file under ``simace/<sub>/`` importing a private
  module or name from ``simace.<other>``.  A top-level module
  ``simace/<name>.py`` is its own sub-package.

A name is private when it starts with one underscore (dunders are public).
Only ``import`` / ``from ... import`` statements are checked: private access
through an attribute (``mod._name``) or a string (a ``monkeypatch.setattr``
target) is not.  A deliberate exception goes in :data:`_ALLOWED` with the
exact private names it permits and its reason; an entry or name that no
longer matches an import fails too, so the list cannot rot.  A checkout that
is absent (a standalone simACE clone) is skipped.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from family_repos import checkout_repos, python_repos  # noqa: E402  (needs the sys.path tweak above)

#: ``(checkout label, path in that checkout, imported module) -> (private names, reason)``.
#: The names are the private module parts and imported names the entry permits.
_ALLOWED: dict[tuple[str, str, str], tuple[frozenset[str], str]] = {
    ("pg-phenotype", "tools/make_simace_aj_golden.py", "simace.analysis.stats.incidence"): (
        frozenset({"_aalen_johansen", "_build_entry_times", "_exit_event_arrays"}),
        "golden generator pinning simACE's private Aalen-Johansen helpers as the oracle",
    ),
    ("pg-phenotype", "tools/make_simace_po_regression_golden.py", "simace.analysis.validate.heritability"): (
        frozenset({"_validate_parent_offspring"}),
        "golden generator pinning simACE's private parent-offspring regression as the oracle",
    ),
    ("pg-phenotype", "tools/make_simace_po_regression_golden.py", "simace.phenotype.models._prevalence"): (
        frozenset({"_prevalence"}),
        "golden generator; must run against simACE releases older than the public re-export",
    ),
    ("pg-phenotype", "tools/make_pedsum_am_golden.py", "pedsum.pedigree_ops"): (
        frozenset({"_parent_rows", "_structural_depth"}),
        "golden generator pinning pedsum's private parent/depth helpers as the oracle",
    ),
    ("pedsum", "tests/test_ecg_and_reference.py", "pedigree_graph._ne_rates"): (
        frozenset({"_ne_rates", "_equivalent_generations"}),
        "differential oracle: pedsum's ECG recurrence against pedigree-graph's internal one",
    ),
    ("pedsum", "tests/test_progress.py", "pedigree_graph._progress"): (
        frozenset({"_progress"}),
        "monkeypatches pedigree-graph's progress tick so the tqdm test runs fast",
    ),
}


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _package_owners() -> dict[str, str]:
    """Top-level package name -> label of the check unit that owns it."""
    owners = {}
    for repo in python_repos():
        unit = _ROOT / repo.path
        if not unit.is_dir():
            continue
        for child in unit.iterdir():
            if child.name != "tests" and (child / "__init__.py").is_file():
                owners[child.name] = repo.label
    return owners


def _unit_roots() -> list[tuple[Path, str]]:
    """``(directory, label)`` of every check unit, deepest first."""
    roots = [(_ROOT / repo.path, repo.label) for repo in python_repos()]
    return sorted(roots, key=lambda root: len(root[0].parts), reverse=True)


def _unit_of(path: Path, roots: list[tuple[Path, str]]) -> str:
    """Label of the deepest check unit whose directory holds ``path``."""
    return next(label for directory, label in roots if path.is_relative_to(directory))


def _module_parts(path: Path) -> list[str] | None:
    """Dotted module path of a file inside a package, or None outside one."""
    parts = [] if path.name == "__init__.py" else [path.stem]
    parent = path.parent
    while (parent / "__init__.py").is_file():
        parts.insert(0, parent.name)
        parent = parent.parent
    return parts if (path.parent / "__init__.py").is_file() else None


def _imports(tree: ast.Module, path: Path):
    """``(lineno, absolute module, imported names)`` for every import in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, []
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = _module_parts(path)
                if parts is None:
                    continue
                package = parts if path.name == "__init__.py" else parts[:-1]
                base = package[: len(package) - (node.level - 1)]
                module = ".".join([*base, module] if module else base)
            yield node.lineno, module, [alias.name for alias in node.names]


def _simace_sub(parts: list[str]) -> str | None:
    """The simace sub-package a dotted path falls in, or None outside simace."""
    if len(parts) < 2 or parts[0] != "simace":
        return None
    return parts[1].removesuffix(".py")


def _violations() -> tuple[list[str], dict[tuple[str, str, str], set[str]]]:
    owners, roots = _package_owners(), _unit_roots()
    found: list[str] = []
    allowed_hits: dict[tuple[str, str, str], set[str]] = {}
    for repo in checkout_repos():
        checkout = _ROOT / repo.path
        if not (checkout / ".git").exists():
            continue
        listed = subprocess.run(
            ["git", "-C", str(checkout), "ls-files", "-z", "*.py"], capture_output=True, text=True, check=True
        ).stdout.split("\0")
        for rel in filter(None, listed):
            path = checkout / rel
            assert path.is_file(), f"{repo.label}: tracked file {rel!r} is missing from the working tree"
            unit = _unit_of(path, roots)
            file_sub = _simace_sub(list(path.relative_to(_ROOT).parts)) if repo.label == "simACE" else None
            for lineno, module, names in _imports(ast.parse(path.read_bytes(), filename=str(path)), path):
                parts = module.split(".")
                private = [p for p in parts if _private(p)] + [n for n in names if _private(n)]
                if not private:
                    continue
                owner = owners.get(parts[0])
                cross_package = owner is not None and owner != unit
                mod_sub = _simace_sub(parts)
                cross_sub = file_sub is not None and mod_sub is not None and mod_sub != file_sub
                if not (cross_package or cross_sub):
                    continue
                key = (repo.label, rel, module)
                if key in _ALLOWED and set(private) <= _ALLOWED[key][0]:
                    allowed_hits.setdefault(key, set()).update(private)
                    continue
                found.append(f"{repo.label}:{rel}:{lineno}: {module} -> {', '.join(private)}")
    return found, allowed_hits


def test_no_private_imports_across_packages() -> None:
    found, allowed_hits = _violations()
    assert not found, (
        "private names imported across a package boundary; promote each to public API "
        "(see GLOSSARY.md 'Public surface') or add a reasoned entry to _ALLOWED:\n" + "\n".join(found)
    )
    checkouts = {repo.label: _ROOT / repo.path for repo in checkout_repos()}
    stale = [
        (key, sorted(names - allowed_hits.get(key, set())))
        for key, (names, _reason) in _ALLOWED.items()
        if (checkouts[key[0]] / ".git").exists() and names - allowed_hits.get(key, set())
    ]
    assert not stale, f"_ALLOWED names that no longer match an import: {stale}"
