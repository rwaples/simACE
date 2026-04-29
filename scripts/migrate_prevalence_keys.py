"""One-shot migration: move ``phenotype.traitN.prevalence`` into per-model params.

After PR3, ``prevalence`` is a property of the threshold-based phenotype models
(``adult`` and ``cure_frailty``) and lives inside their per-trait ``params:``
sub-dict. Frailty and first_passage no longer accept it.

This script walks every YAML under ``config/`` and rewrites the placement:

    phenotype:
      trait1:
        model: adult              # (or cure_frailty)
        prevalence: 0.1           # ← deleted at this level
        params:
          method: ltm
          cip_x0: 50
          prevalence: 0.1         # ← inserted here

    phenotype:
      trait2:
        model: frailty            # (or first_passage)
        prevalence: 0.2           # ← deleted (was silently ignored)
        params:
          ...

The script is idempotent. Running on a file that's already migrated produces
no further changes.

Usage:
    python scripts/migrate_prevalence_keys.py            # rewrite in place
    python scripts/migrate_prevalence_keys.py --check    # dry-run; non-zero exit if any file would change

Comments are NOT preserved (PyYAML drops them on round-trip). The repo's
scenario YAMLs (epimight_demo, long, solmi, etc.) have no comments. The
single comment-bearing file, ``config/_default.yaml``, must be hand-edited
to preserve its comments — this script skips it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
SKIP_FILES = {"_default.yaml"}  # hand-edited to preserve comments

PREVALENCE_BEARING = frozenset({"adult", "cure_frailty"})
ALL_MODELS = frozenset({"frailty", "cure_frailty", "adult", "first_passage"})


def _migrate_trait_block(trait_block: dict[str, Any], default_model: str = "frailty") -> bool:
    """Apply the migration to a single ``phenotype.traitN`` mapping in place.

    Returns True if anything changed.
    """
    if not isinstance(trait_block, dict):
        return False
    if "prevalence" not in trait_block:
        return False
    model = trait_block.get("model", default_model)
    if model not in ALL_MODELS:
        return False
    prevalence = trait_block.pop("prevalence")
    if model in PREVALENCE_BEARING:
        params = trait_block.setdefault("params", {})
        if not isinstance(params, dict):
            # Pathological case: params is non-dict. Restore prevalence and bail.
            trait_block["prevalence"] = prevalence
            return False
        params["prevalence"] = prevalence
    return True


def _walk(node: Any) -> bool:
    """Recursively migrate any ``phenotype.{trait1,trait2}`` blocks under ``node``.

    Returns True if any changes were made.
    """
    changed = False
    if isinstance(node, dict):
        phenotype = node.get("phenotype")
        if isinstance(phenotype, dict):
            for trait_key in ("trait1", "trait2"):
                trait_block = phenotype.get(trait_key)
                if isinstance(trait_block, dict) and _migrate_trait_block(trait_block):
                    changed = True
        for value in node.values():
            if _walk(value):
                changed = True
    elif isinstance(node, list):
        for item in node:
            if _walk(item):
                changed = True
    return changed


def migrate_file(path: Path) -> bool:
    """Migrate one YAML file in place. Returns True if file was changed."""
    text = path.read_text()
    data = yaml.safe_load(text)
    if data is None:
        return False
    if not _walk(data):
        return False
    new_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, indent=2)
    path.write_text(new_text)
    return True


def main() -> int:
    """CLI entry point for the migration script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Dry run; non-zero exit if any change is needed.")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="Directory of YAML configs to walk.")
    args = parser.parse_args()

    if not args.config_dir.exists():
        print(f"config dir not found: {args.config_dir}", file=sys.stderr)
        return 2

    needs_change = []
    for path in sorted(args.config_dir.glob("*.yaml")):
        if path.name in SKIP_FILES:
            continue
        text = path.read_text()
        data = yaml.safe_load(text)
        if data is None:
            continue
        # Run on a deep copy to detect change without mutating the file in --check mode.
        import copy

        probe = copy.deepcopy(data)
        if not _walk(probe):
            continue
        needs_change.append(path)
        if args.check:
            print(f"would migrate: {path.relative_to(REPO_ROOT)}")
        else:
            migrate_file(path)
            print(f"migrated:      {path.relative_to(REPO_ROOT)}")

    if not needs_change:
        print("All config files already migrated; no changes needed.")
        return 0
    if args.check:
        print(f"\n{len(needs_change)} file(s) need migration. Run without --check to apply.")
        return 1
    print(f"\n{len(needs_change)} file(s) migrated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
