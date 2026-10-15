"""Write a rep's pedigree as PLINK files: ``.fam``, plus ``.pheno`` and ``.twins`` sidecars.

``<stem>.fam`` is PLINK's six-column pedigree file (``FID IID PAT MAT SEX
PHENO``, tab-separated, no header). Everyone shares one FID, because PLINK
links a child only to parents with the same FID. An IID is ``id_<id>`` with
the simACE id unchanged, so a missing parent (simACE ``-1``) is PLINK's
``0`` and never collides with a real person. Sex ``0``/``1`` (female/male)
becomes PLINK's ``2``/``1``.

``<stem>.pheno`` (``FID IID affected1 affected2``, with a header) holds both
traits, and the ``.fam`` PHENO is trait 1. Given the rep's cohort, the files
hold the analysis pedigree (ADR 0021): cases are ``2``, controls ``1``, and
cohort members outside the analysis sample ``-9`` (ancestors, whether never
phenotyped or phenotyped but not drawn). Given only the recorded pedigree,
the files hold all of it and every phenotype is ``-9``.

PLINK has no twin field, so ``<stem>.twins`` lists each MZ pair once
(``IID1 IID2``, with a header).
"""

from __future__ import annotations

__all__ = ["PLINK_FID", "plink_iid", "write_plink"]

import logging
from pathlib import Path

import polars as pl

from simace.core.cohort import read_cohort, read_pedigree, selected_views
from simace.core.publish import publish

logger = logging.getLogger(__name__)

PLINK_FID = "1"
_MISSING_PHENO = -9
_OUTPUT_SUFFIXES = (".fam", ".pheno", ".twins")


def plink_iid(ids: pl.Expr) -> pl.Expr:
    """PLINK IIDs of simACE ids: ``id_<id>``, or ``0`` for a missing link (a negative id)."""
    return pl.when(ids >= 0).then(pl.format("id_{}", ids)).otherwise(pl.lit("0"))


def _pheno(affected: pl.Expr) -> pl.Expr:
    return pl.when(affected.is_null()).then(_MISSING_PHENO).when(affected).then(2).otherwise(1)


def write_plink(pedigree: pl.DataFrame, stem: str | Path, cohort: pl.DataFrame | None = None) -> list[Path]:
    """Write ``<stem>.fam``, ``<stem>.pheno`` and ``<stem>.twins``.

    Args:
        pedigree: The recorded pedigree (``pedigree.parquet``).
        stem: Output path without an extension.
        cohort: The rep's ``cohort.parquet``. When given, the analysis pedigree
            is exported with its outcomes.

    Returns:
        The paths written.
    """
    if cohort is None:
        missing = pl.lit(None, pl.Boolean)
        people = pedigree.with_columns(missing.alias("affected1"), missing.alias("affected2"))
    else:
        # selected_views checks that the cohort is the analysis pedigree, row for row.
        people = selected_views(pedigree, cohort).pedigree.with_columns(cohort["affected1"], cohort["affected2"])
    fid = pl.lit(PLINK_FID).alias("FID")
    iid = plink_iid(pl.col("id")).alias("IID")

    fam = people.select(
        fid,
        iid,
        plink_iid(pl.col("father")).alias("PAT"),
        plink_iid(pl.col("mother")).alias("MAT"),
        pl.when(pl.col("sex") == 1).then(1).when(pl.col("sex") == 0).then(2).otherwise(0).alias("SEX"),
        _pheno(pl.col("affected1")).alias("PHENO"),
    )
    twins = people.filter(pl.col("twin") > pl.col("id")).select(
        iid.alias("IID1"), plink_iid(pl.col("twin")).alias("IID2")
    )
    pheno = people.select(
        fid, iid, _pheno(pl.col("affected1")).alias("affected1"), _pheno(pl.col("affected2")).alias("affected2")
    )
    stem = str(stem)
    outputs: dict[Path, tuple[pl.DataFrame, bool]] = {
        Path(f"{stem}.fam"): (fam, False),
        Path(f"{stem}.pheno"): (pheno, True),
        Path(f"{stem}.twins"): (twins, True),
    }

    with publish(*outputs) as tmps:
        for tmp, (frame, header) in zip(tmps, outputs.values(), strict=True):
            frame.write_csv(tmp, separator="\t", include_header=header)
    logger.info("Wrote %s (%d people, %d MZ pairs)", ", ".join(map(str, outputs)), fam.height, twins.height)
    return list(outputs)


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line interface: export-plink."""
    import argparse

    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging

    parser = argparse.ArgumentParser(prog=prog, description="Write a rep's pedigree as PLINK .fam, .pheno and .twins")
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree", required=True, help="Recorded pedigree parquet (pedigree.parquet)")
    parser.add_argument(
        "--cohort", default=None, help="The rep's cohort.parquet: export the analysis pedigree with outcomes"
    )
    parser.add_argument("-o", "--output", required=True, help="Output stem; writes <stem>.fam, .twins, .pheno")
    args = parser.parse_args(argv)
    if args.output.endswith(_OUTPUT_SUFFIXES):
        parser.error(f"-o is a stem without an extension; got {args.output}")
    init_logging(args)

    pedigree = read_pedigree(args.pedigree, columns=["id", "sex", "mother", "father", "twin"])
    cohort = read_cohort(args.cohort) if args.cohort is not None else None
    write_plink(pedigree, args.output, cohort)


if __name__ == "__main__":
    cli()
