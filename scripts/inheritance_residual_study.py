r"""Calibrate the sibling inheritance cells of ADR 0029 on existing reps (simACE #67).

For every complete rep of each configured scenario in the named folders,
compute the cells the ``mendelian_residual`` and ``household_draw`` tests
score, from the rep's recorded pedigree and the parameters in its
``report.yaml``:

- per trait with A > 0, the Pearson correlation of the Mendelian residual
  A − midparent over FS, MHS and PHS pairs (expected 0), and its variance
  over the non-founders with one member of each MZ pair (expected A/2), with
  the asymptotic iid SE √((m₄ − s⁴)/n) and the exact χ² z;
- per trait with C > 0, the Pearson correlation of the household draw
  C − b_m·C_mother − b_f·C_father over PHS pairs (expected 0), with the
  share of PHS pairs whose two households are linked by more than one father.

The correlations take pg-phenotype's MRCA-or-shared-member sandwich SE
(``relative_pairs.correlation(..., se=True)``). One row per rep, trait and
cell goes to a TSV, and the pooled summary (mean z, SD z, share of
|z| > 1.96, per-scenario SD z range) is printed.

Two scenarios whose rep seed ranges overlap reuse pedigrees: with random
mating and no cross-trait correlation, the same seed gives the same pedigree
and the same standardized draws whatever the variances (``se_calibration``
spaces its 100-rep scenarios 10 seeds apart). Each row carries a
``pedigree_hash`` of the mother and father columns, and the summary counts
each hash once, keeping its first scenario in name order.

Example::

    pixi run python scripts/inheritance_residual_study.py se_calibration onset_tau --jobs 8
"""

import argparse
import hashlib
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from simace.cli.layout import RepArtifact, add_root_args, resolve_roots
from simace.cli.manifest import RepState
from simace.cli.stages import ResolvedRep
from simace.cli.status import resolve_all, status_on_disk
from simace.core.cohort import read_pedigree
from simace.core.yaml_io import load_yaml

if TYPE_CHECKING:
    from simace.cli.layout import Layout

#: One thread per worker, set before a spawned worker imports polars or pg-phenotype.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "POLARS_MAX_THREADS")

#: The per-scenario SD z range covers scenario × trait cells with at least this many distinct reps.
_MIN_SCENARIO_REPS = 50

_COLUMNS = [
    "folder",
    "scenario",
    "rep",
    "trait",
    "cell",
    "n",
    "observed",
    "expected",
    "se",
    "z",
    "z_exact",
    "linked",
    "pedigree_hash",
]


def _parent_rows(pedigree: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each row's mother and father row, -1 where the parent is unknown or not a row."""
    ids = pedigree["id"].to_numpy().astype(np.int64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    def lookup(parent: np.ndarray) -> np.ndarray:
        parent = parent.astype(np.int64)
        at = np.clip(np.searchsorted(sorted_ids, parent), 0, len(ids) - 1)
        return np.where((parent >= 0) & (sorted_ids[at] == parent), order[at], -1)

    return lookup(pedigree["mother"].to_numpy()), lookup(pedigree["father"].to_numpy())


def _linked_household_share(pedigree: pl.DataFrame) -> float:
    """Share of PHS pairs whose two households have children by more than one common father."""
    per_father = pedigree.filter(pl.col("father") >= 0).group_by("father", "household_id").agg(k=pl.len())
    pairs = (
        per_father.join(per_father, on="father")
        .filter(pl.col("household_id") < pl.col("household_id_right"))
        .with_columns(pairs=pl.col("k") * pl.col("k_right"))
        .group_by("household_id", "household_id_right")
        .agg(fathers=pl.len(), pairs=pl.col("pairs").sum())
    )
    total = pairs["pairs"].sum()
    return float(pairs.filter(pl.col("fathers") > 1)["pairs"].sum() / total) if total else 0.0


def _row(key: tuple, trait: int, cell: str, n: int, observed: float, expected: float, se: float | None, **extra: Any):
    z = None if se is None or observed is None else (observed - expected) / se
    folder, scenario, rep = key
    row = dict.fromkeys(_COLUMNS)
    row.update(folder=folder, scenario=scenario, rep=rep, trait=trait, cell=cell, n=n)
    row.update(observed=observed, expected=expected, se=se, z=z, **extra)
    return row


def rep_cells(key: tuple[str, str, int], pedigree_path: Path, report_path: Path) -> list[dict[str, Any]]:
    """One rep's cells, as rows of the TSV."""
    from pg_phenotype import Trait
    from pg_phenotype.relative_pairs import correlation

    pedigree = read_pedigree(pedigree_path)
    params = load_yaml(report_path)["inputs"]["parameters"]
    mother, father = _parent_rows(pedigree)
    both = (mother >= 0) & (father >= 0)
    ids, twin = pedigree["id"].to_numpy(), pedigree["twin"].to_numpy()
    one_per_mz = both & ((twin < 0) | (ids < twin))
    digest = hashlib.blake2b(digest_size=16)
    for column in ("mother", "father"):
        digest.update(pedigree[column].to_numpy().astype(np.int64).tobytes())
    rows = []
    for t in (1, 2):
        if params[f"A{t}"] > 0:
            a = pedigree[f"A{t}"].to_numpy().astype(np.float64)
            residual = np.full(len(a), np.nan)
            residual[both] = a[both] - (a[mother[both]] + a[father[both]]) / 2
            cells = correlation(
                pedigree, Trait(residual, kind="continuous"), categories=["FS", "MHS", "PHS"], se=True
            ).cells
            for cell in cells:
                est = cell.crude[0]
                rows.append(_row(key, t, f"A_{cell.category}", cell.n, est.value, 0.0, est.se))
            x = residual[one_per_mz]
            n, s2, target = len(x), float(x.var(ddof=1)), params[f"A{t}"] / 2
            se = float(np.sqrt((np.mean((x - x.mean()) ** 4) - s2**2) / n))
            rows.append(_row(key, t, "A_variance", n, s2, target, se, z_exact=(s2 / target - 1) * np.sqrt((n - 1) / 2)))
        if params[f"C{t}"] > 0:
            c = pedigree[f"C{t}"].to_numpy().astype(np.float64)
            b_m, b_f = params.get(f"Ctm_maternal{t}", 0.0), params.get(f"Ctm_paternal{t}", 0.0)
            draw = np.full(len(c), np.nan)
            draw[both] = c[both] - b_m * c[mother[both]] - b_f * c[father[both]]
            cell = correlation(pedigree, Trait(draw, kind="continuous"), categories=["PHS"], se=True).cells[0]
            est = cell.crude[0]
            rows.append(_row(key, t, "C_PHS", cell.n, est.value, 0.0, est.se, linked=_linked_household_share(pedigree)))
    for row in rows:
        row["pedigree_hash"] = digest.hexdigest()
    return rows


def _complete_reps(folder: str, scenario: str, params: dict[str, Any], layout: Layout) -> list[ResolvedRep]:
    """The scenario's reps that are complete under the current config, naming the others."""
    reps = []
    for r in range(1, int(params["replicates"]) + 1):
        rep = ResolvedRep(folder, scenario, r, params)
        state = status_on_disk(rep, layout).state
        if state is RepState.COMPLETE:
            reps.append(rep)
        else:
            print(f"inheritance_residual_study: skipping {rep.dir(layout)} ({state})", file=sys.stderr)
    return reps


def main(argv: list[str] | None = None) -> None:
    """Compute every rep's cells over the folders, write one TSV, and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="Folders under the results root, as in config/{folder}.yaml")
    parser.add_argument("--jobs", type=int, default=1, help="Reps computed at once")
    parser.add_argument(
        "--output", type=Path, default=None, help="TSV path (default: results/inheritance_residual_study.tsv)"
    )
    add_root_args(parser)
    args = parser.parse_args(argv)
    config_dir, layout = resolve_roots(args)

    configured = resolve_all(config_dir)
    jobs = [
        ((folder, name, rep.rep), rep.path(layout, RepArtifact.PEDIGREE), rep.path(layout, RepArtifact.REPORT))
        for folder in args.folders
        for name, p in sorted(configured.items())
        if p["folder"] == folder
        for rep in _complete_reps(folder, name, p, layout)
    ]
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = [row for rep_rows in pool.map(rep_cells, *zip(*jobs, strict=True)) for row in rep_rows]
    table = pl.DataFrame(rows, infer_schema_length=None).select(_COLUMNS)
    output = args.output or layout.root / "inheritance_residual_study.tsv"
    table.write_csv(output, separator="\t")

    first = table.unique("pedigree_hash", keep="first", maintain_order=True).select("pedigree_hash", "scenario")
    distinct = table.join(first, on=["pedigree_hash", "scenario"])
    print(
        f"{distinct['pedigree_hash'].n_unique()} distinct pedigrees over {table.select('folder', 'scenario', 'rep').n_unique()} reps"
    )
    table = distinct
    per_scenario = (
        table.group_by("folder", "scenario", "trait", "cell")
        .agg(reps=pl.col("z").count(), sd_z=pl.col("z").std())
        .filter(pl.col("reps") >= _MIN_SCENARIO_REPS)
    )
    summary = (
        table.group_by("cell")
        .agg(
            n=pl.col("z").count(),
            unscored=pl.col("z").null_count(),
            mean_z=pl.col("z").mean(),
            sd_z=pl.col("z").std(),
            over_1_96=(pl.col("z").abs() > 1.96).mean(),
            mean_z_exact=pl.col("z_exact").mean(),
            max_linked=pl.col("linked").max(),
        )
        .join(
            per_scenario.group_by("cell").agg(sd_z_min=pl.col("sd_z").min(), sd_z_max=pl.col("sd_z").max()),
            on="cell",
        )
        .sort("cell")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(summary)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
