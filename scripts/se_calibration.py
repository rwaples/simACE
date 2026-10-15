r"""Compare reported tetrachoric SEs with the spread of r across replicates (simACE #55).

For every complete rep of each configured scenario in a folder, read the
tetrachoric entries of ``report.yaml`` and, per section, trait and relationship
category, write the mean reported ``se``, the SD of ``r`` across reps, and
their ratio. A ratio above 1 means the reported SE is too small.
``same_person`` (one row per person) is the control.

A category whose ``r`` or ``se`` is withheld in some rep (a boundary fit, or
too few pairs) keeps only its non-boundary reps, which biases its SD down.
Its ``all_estimated`` is false and the printed summary leaves it out.

Example::

    pixi run simace run se_calibration --jobs 8 --no-plots
    pixi run python scripts/se_calibration.py se_calibration
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from simace.cli.layout import RepArtifact, add_root_args, resolve_roots
from simace.cli.manifest import RepState
from simace.cli.stages import ResolvedRep
from simace.cli.status import resolve_all, status_on_disk
from simace.core.yaml_io import load_yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

    from simace.cli.layout import Layout

_SUMMARY_SECTIONS = ("tetrachoric", "cross_trait.cross_person", "cross_trait.same_person")


def _entries(sample: dict[str, Any]) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    """Yield ``(section, trait, category, entry)`` for every tetrachoric entry with an ``se``."""
    for trait, by_cat in sample["tetrachoric"].items():
        for cat, entry in by_cat.items():
            yield "tetrachoric", trait, cat, entry
    for gen, by_trait in sample["tetrachoric_by_generation"].items():
        for trait, by_cat in by_trait.items():
            for cat, entry in by_cat.items():
                yield f"tetrachoric_by_generation.{gen}", trait, cat, entry
    cross = sample["cross_trait_tetrachoric"]
    for cat, entry in cross["cross_person"].items():
        yield "cross_trait.cross_person", "trait1_x_trait2", cat, entry
    yield "cross_trait.same_person", "trait1_x_trait2", "self", cross["same_person"]


def _complete_reports(folder: str, scenario: str, params: dict[str, Any], layout: Layout) -> list[Path]:
    """Report paths of the scenario's reps that are complete under the current config, naming the others."""
    reports = []
    for r in range(1, int(params["replicates"]) + 1):
        rep = ResolvedRep(folder, scenario, r, params)
        status = status_on_disk(rep, layout)
        if status.state is RepState.COMPLETE:
            reports.append(rep.path(layout, RepArtifact.REPORT))
        else:
            print(f"se_calibration: skipping {rep.dir(layout)} ({status.state})", file=sys.stderr)
    return reports


def collect(scenario: str, reports: list[Path]) -> pl.DataFrame:
    """One row per section, trait and category over *reports*."""
    values: dict[tuple[str, str, str], list[tuple[float, float, int]]] = defaultdict(list)
    for report in reports:
        sample = load_yaml(report)["observed"]["analysis_sample"]
        for section, trait, cat, entry in _entries(sample):
            if entry["r"] is not None and entry["se"] is not None:
                values[section, trait, cat].append((entry["r"], entry["se"], entry.get("n_pairs", entry.get("n"))))
    rows = []
    for (section, trait, cat), triples in values.items():
        r, se, n = (np.array(col, dtype=float) for col in zip(*triples, strict=True))
        sd_r = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
        rows.append(
            {
                "scenario": scenario,
                "section": section,
                "trait": trait,
                "category": cat,
                "n_reps": len(r),
                "all_estimated": len(r) == len(reports),
                "mean_n_pairs": float(n.mean()),
                "mean_r": float(r.mean()),
                "sd_r": sd_r,
                "mean_se": float(se.mean()),
                "sd_over_se": sd_r / float(se.mean()),
            }
        )
    return pl.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Collect every configured scenario of the folder, write one TSV, and print the pooled summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="Folder under the results root, as in config/{folder}.yaml")
    parser.add_argument(
        "--output", type=Path, default=None, help="TSV path (default: results/{folder}/se_calibration.tsv)"
    )
    add_root_args(parser)
    args = parser.parse_args(argv)
    config_dir, layout = resolve_roots(args)

    configured = {name: p for name, p in resolve_all(config_dir).items() if p["folder"] == args.folder}
    frames = [collect(name, _complete_reports(args.folder, name, p, layout)) for name, p in sorted(configured.items())]
    table = pl.concat([f for f in frames if f.height]).sort("scenario", "section", "trait", "category")
    output = args.output or layout.root / args.folder / "se_calibration.tsv"
    table.write_csv(output, separator="\t", float_precision=4)

    summary = table.filter(pl.col("section").is_in(_SUMMARY_SECTIONS))
    dropped = summary.filter(~pl.col("all_estimated"))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(
            summary.filter(pl.col("all_estimated"))
            .group_by("section", "trait")
            .agg(
                pl.len().alias("cells"),
                pl.col("sd_over_se").min().alias("min"),
                pl.col("sd_over_se").median().alias("median"),
                pl.col("sd_over_se").max().alias("max"),
            )
            .sort("section", "trait")
        )
        print("Left out (r or se withheld in some rep):")
        print(dropped.select("scenario", "trait", "category", "n_reps", "sd_over_se"))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
