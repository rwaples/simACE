#!/usr/bin/env python3
"""Audit the disk footprint of a simACE/fitACE ``results/`` tree.

The script is deliberately conservative: it reports every file by category and
writes a TSV manifest, but deletion is dry-run by default.  Deletion requires all
of:

    --delete --category <auto-safe-category> --yes

Only categories marked ``auto_delete_safe`` can be deleted.  Today that means
legacy simACE files that are shadowed by their canonical renamed sibling.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_RENAMED = {
    "phenotype.parquet": "trait.parquet",
}

AUTO_DELETE_SAFE = frozenset({"legacy-renamed-simace-shadowed"})

CURRENT_SIMACE_BASENAMES = frozenset(
    {
        "pedigree.full.parquet",
        "pedigree.parquet",
        "trait.full.parquet",
        "trait.parquet",
        "params.yaml",
        "report.yaml",
        "plot_payload.yaml",
        "report_summary.tsv",
        "scenario.done",
        "simulate.done",
        "phenotype.done",
        "validate.done",
        "stats.done",
        "folder.done",
    }
)

SIMACE_TEMP_BASENAMES = frozenset(
    {
        "trait.raw.parquet",
        "plotting_sample.parquet",
    }
)

FITACE_EPIMIGHT_BASENAMES = frozenset(
    {
        "phenotype.blended.parquet",
        "pipeline_input.parquet",
        "trait1.epimight_in.parquet",
        "trait2.epimight_in.parquet",
        "fit.rel",
        "fit.relatives.tsv",
        "scores.parquet",
        "A.grm.sp",
        "A.grm.sp.bin",
    }
)

FITACE_EPIMIGHT_SUFFIXES = (
    ".mle",
    ".rel",
    ".grm.sp",
    ".grm.sp.bin",
    ".epimight_in.parquet",
)

GENOTYPE_SUFFIXES = (
    ".trees",
    "_meta.json",
)

PLOT_SUFFIXES = frozenset({".png", ".pdf", ".html", ".svg"})
TSV_SUFFIXES = (".tsv", ".tsv.gz")


@dataclass(frozen=True)
class FileRecord:
    """One classified file in the results footprint audit."""

    path: Path
    category: str
    reason: str
    size_bytes: int
    disk_bytes: int
    auto_delete_safe: bool
    canonical_sibling: Path | None = None
    deleted: bool = False


def _human_size(num_bytes: int) -> str:
    """Return a compact binary-unit size string."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def _relative(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible, else as a string."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _is_genotype_or_tstrait_output(name: str) -> bool:
    """Return True for genotype-drop / tstrait intermediate output names."""
    if name.startswith("genotypes_chrom_") and name.endswith(".trees"):
        return True
    if name.startswith("gv_chrom_") and name.endswith(".parquet"):
        return True
    if name.startswith("gv_chrom_") and name.endswith("_meta.json"):
        return True
    return name in {
        "pedigree_tables.trees",
        "tstrait_phenotype.parquet",
        "tstrait_phenotype_meta.json",
        "pedigree.full.tstrait.parquet",
        "pedigree.full.tstrait_meta.json",
        "causal_effects.parquet",
        "causal_effects_meta.json",
    }


def _classify(path: Path, root: Path) -> FileRecord:
    """Classify one file into a cleanup/audit category."""
    name = path.name
    stat = path.stat()
    size_bytes = stat.st_size
    disk_bytes = getattr(stat, "st_blocks", 0) * 512

    canonical = LEGACY_RENAMED.get(name)
    if canonical is not None:
        sibling = path.with_name(canonical)
        if sibling.exists():
            category = "legacy-renamed-simace-shadowed"
            reason = f"legacy basename; canonical sibling exists ({canonical})"
        else:
            category = "legacy-renamed-simace-orphaned"
            reason = f"legacy basename; canonical sibling is missing ({canonical})"
        return FileRecord(
            path=path,
            category=category,
            reason=reason,
            size_bytes=size_bytes,
            disk_bytes=disk_bytes,
            auto_delete_safe=category in AUTO_DELETE_SAFE,
            canonical_sibling=sibling,
        )

    if name in FITACE_EPIMIGHT_BASENAMES or name.endswith(FITACE_EPIMIGHT_SUFFIXES):
        return FileRecord(
            path=path,
            category="fitace-epimight-manual-review",
            reason="fitACE/EPIMIGHT-adjacent output; never auto-deleted by this script",
            size_bytes=size_bytes,
            disk_bytes=disk_bytes,
            auto_delete_safe=False,
        )

    if _is_genotype_or_tstrait_output(name) or name.endswith(GENOTYPE_SUFFIXES):
        return FileRecord(
            path=path,
            category="genotype-tstrait-manual-review",
            reason="genotype-drop or tstrait output; manual review only",
            size_bytes=size_bytes,
            disk_bytes=disk_bytes,
            auto_delete_safe=False,
        )

    if name in SIMACE_TEMP_BASENAMES:
        return FileRecord(
            path=path,
            category="simace-temp-leftover-manual-review",
            reason="Snakemake temp/intermediate simACE output; use Snakemake cleanup or review manually",
            size_bytes=size_bytes,
            disk_bytes=disk_bytes,
            auto_delete_safe=False,
        )

    if name in CURRENT_SIMACE_BASENAMES:
        return FileRecord(
            path=path,
            category="current-simace-output",
            reason="current simACE output contract",
            size_bytes=size_bytes,
            disk_bytes=disk_bytes,
            auto_delete_safe=False,
        )

    if path.parent.name == "plots" or path.suffix in PLOT_SUFFIXES:
        return FileRecord(
            path=path,
            category="plot-output",
            reason="plot or atlas artifact",
            size_bytes=size_bytes,
            disk_bytes=disk_bytes,
            auto_delete_safe=False,
        )

    if name.endswith(TSV_SUFFIXES):
        return FileRecord(
            path=path,
            category="tsv-export-manual-review",
            reason="TSV export or tabular sidecar; manual review only",
            size_bytes=size_bytes,
            disk_bytes=disk_bytes,
            auto_delete_safe=False,
        )

    return FileRecord(
        path=path,
        category="unknown-manual-review",
        reason="not recognized by the audit classifier",
        size_bytes=size_bytes,
        disk_bytes=disk_bytes,
        auto_delete_safe=False,
    )


def scan(root: Path, categories: set[str] | None = None) -> list[FileRecord]:
    """Scan ``root`` recursively and return classified file records."""
    records: list[FileRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        record = _classify(path, root)
        if categories is None or record.category in categories:
            records.append(record)
    return records


def write_manifest(records: list[FileRecord], manifest_path: Path, root: Path) -> None:
    """Write a TSV manifest for ``records``."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            delimiter="\t",
            fieldnames=[
                "category",
                "auto_delete_safe",
                "deleted",
                "size_bytes",
                "disk_bytes",
                "size_human",
                "path",
                "canonical_sibling",
                "reason",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "category": record.category,
                    "auto_delete_safe": str(record.auto_delete_safe).lower(),
                    "deleted": str(record.deleted).lower(),
                    "size_bytes": record.size_bytes,
                    "disk_bytes": record.disk_bytes,
                    "size_human": _human_size(record.size_bytes),
                    "path": _relative(record.path, root),
                    "canonical_sibling": ""
                    if record.canonical_sibling is None
                    else _relative(record.canonical_sibling, root),
                    "reason": record.reason,
                }
            )


def delete_records(records: list[FileRecord], selected_categories: set[str]) -> list[FileRecord]:
    """Delete selected auto-safe records and return updated records."""
    updated: list[FileRecord] = []
    for record in records:
        if record.category not in selected_categories:
            updated.append(record)
            continue
        record.path.unlink()
        updated.append(
            FileRecord(
                path=record.path,
                category=record.category,
                reason=record.reason,
                size_bytes=record.size_bytes,
                disk_bytes=record.disk_bytes,
                auto_delete_safe=record.auto_delete_safe,
                canonical_sibling=record.canonical_sibling,
                deleted=True,
            )
        )
    return updated


def _summarize(records: list[FileRecord]) -> dict[str, dict[str, Any]]:
    """Return per-category count/size summary rows."""
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "size_bytes": 0, "disk_bytes": 0})
    for record in records:
        row = summary[record.category]
        row["count"] += 1
        row["size_bytes"] += record.size_bytes
        row["disk_bytes"] += record.disk_bytes
        row["auto_delete_safe"] = record.auto_delete_safe
    return dict(summary)


def print_summary(records: list[FileRecord], *, root: Path, manifest_path: Path, deleted: bool) -> None:
    """Print a human-readable audit summary."""
    total_size = sum(record.size_bytes for record in records)
    total_disk = sum(record.disk_bytes for record in records)
    print(f"Results root: {root}")
    print(f"Files listed: {len(records):,}")
    print(f"Logical size listed: {_human_size(total_size)}")
    if total_disk:
        print(f"Allocated disk listed: {_human_size(total_disk)}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Category summary:")
    print(f"{'category':45s} {'files':>8s} {'logical':>12s} {'disk':>12s} {'auto-delete':>12s}")
    print("-" * 93)
    for category, row in sorted(_summarize(records).items(), key=lambda item: item[1]["size_bytes"], reverse=True):
        print(
            f"{category:45s} "
            f"{row['count']:8,d} "
            f"{_human_size(row['size_bytes']):>12s} "
            f"{_human_size(row['disk_bytes']):>12s} "
            f"{str(row.get('auto_delete_safe', False)).lower():>12s}"
        )
    if deleted:
        deleted_records = [record for record in records if record.deleted]
        deleted_size = sum(record.size_bytes for record in deleted_records)
        print()
        print(f"Deleted files: {len(deleted_records):,} ({_human_size(deleted_size)} logical)")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and conservatively clean a simACE/fitACE results tree.")
    parser.add_argument("--root", type=Path, default=Path("results"), help="Results directory to scan.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results_footprint_manifest.tsv"),
        help="TSV manifest path to write.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Restrict report/delete to a category. Repeat for multiple categories.",
    )
    parser.add_argument("--delete", action="store_true", help="Delete selected auto-safe category files.")
    parser.add_argument("--yes", action="store_true", help="Required together with --delete.")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace, categories: set[str] | None) -> None:
    if not args.root.exists():
        raise SystemExit(f"results root does not exist: {args.root}")
    if not args.root.is_dir():
        raise SystemExit(f"results root is not a directory: {args.root}")
    if not args.delete:
        return
    if not args.yes:
        raise SystemExit("refusing to delete without --yes")
    if not categories:
        raise SystemExit("refusing to delete without at least one --category")
    unsafe = sorted(categories - AUTO_DELETE_SAFE)
    if unsafe:
        safe = ", ".join(sorted(AUTO_DELETE_SAFE))
        raise SystemExit(
            f"refusing to delete non-auto-safe categories: {', '.join(unsafe)}. Auto-safe categories: {safe}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    categories = set(args.category) if args.category else None
    _validate_args(args, categories)

    root = args.root.resolve()
    records = scan(root, categories=categories)
    if args.delete:
        records = delete_records(records, selected_categories=categories or set())
    write_manifest(records, args.manifest, root)
    print_summary(records, root=root, manifest_path=args.manifest, deleted=args.delete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
