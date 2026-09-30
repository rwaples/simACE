"""The ``simace cohort`` stage: phenotype, censor, and ascertain one recorded pedigree in one process.

Runs :func:`~simace.phenotype.runner.run_phenotype`,
:func:`~simace.censoring.censor.run_censor`, and
:func:`~simace.ascertainment.runner.run_ascertainment` on in-memory frames
and writes only the two durable outputs (ADR 0021): ``cohort.parquet`` and
``phenotyped_population.yaml``, the phenotyped-population summary Analyze
reports. No trait file touches disk.

Each phase's output is put through
:func:`~simace.core.parquet.normalize_for_parquet`, so the next phase sees
exactly the values it would have read back from a file (float32 onsets and
death ages decide censoring outcomes). Each phase frees its input before the
next and logs its wall time and the process's peak RSS so far.
"""

from __future__ import annotations

__all__ = ["cli", "run_cohort"]

import argparse
import gc
import logging
import resource
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import polars as pl

from simace.analysis.prevalence import compute_prevalence
from simace.ascertainment.runner import run_ascertainment
from simace.censoring.censor import run_censor
from simace.core.cohort import build_cohort, read_pedigree, write_cohort
from simace.core.parquet import normalize_for_parquet
from simace.core.pedigree_arrays import PedigreeArrays
from simace.phenotype.runner import run_phenotype

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


logger = logging.getLogger(__name__)


@contextmanager
def _phase(name: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    logger.info("cohort phase %s: %.2fs, peak RSS so far %.0f MB", name, time.perf_counter() - start, peak_mb)


def run_cohort(
    pedigree: pl.DataFrame,
    *,
    seed: int,
    phenotype: Mapping[str, Any],
    censor: Mapping[str, Any],
    ascertain: Mapping[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return the cohort of ``pedigree`` and the summary of its phenotyped population.

    Args:
        pedigree: The recorded pedigree, as read from ``pedigree.parquet``.
        seed: The rep seed, passed to all three phases.
        phenotype: ``run_phenotype`` keyword arguments other than ``seed``.
        censor: ``run_censor`` keyword arguments other than ``seed``.
        ascertain: ``run_ascertainment`` keyword arguments other than ``seed``.

    Returns:
        The cohort frame, and ``{n_individuals, n_generations, prevalence}``
        over the censored phenotyped rows before ascertainment.
    """
    with _phase("phenotype"):
        raw = normalize_for_parquet(run_phenotype(pedigree, seed=seed, **phenotype))

    with _phase("censor"):
        censored = normalize_for_parquet(run_censor(raw, pedigree, seed=seed, **censor))
        del raw
        gc.collect()
        generation = PedigreeArrays.from_frame(pedigree.select("id", "generation")).gather(
            "generation", censored["id"].to_numpy()
        )
        phenotyped = censored.select("affected1", "affected2").with_columns(pl.Series("generation", generation))
        phenotyped_population = {
            "n_individuals": len(phenotyped),
            "n_generations": phenotyped["generation"].n_unique(),
            "prevalence": compute_prevalence(phenotyped),
        }
        del generation, phenotyped

    with _phase("ascertain"):
        analysis_pedigree, sample_trait = run_ascertainment(pedigree, censored, seed=seed, **ascertain)
        del censored
        gc.collect()
        cohort = build_cohort(analysis_pedigree, sample_trait)

    return cohort, phenotyped_population


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line entry point for the ``cohort`` stage."""
    from simace.core.cli_base import add_logging_args, add_version_arg, generation_map, init_logging, yaml_mapping
    from simace.core.publish import publish
    from simace.core.yaml_io import dump_yaml
    from simace.phenotype.hazards import STANDARDIZE_CHOICES
    from simace.phenotype.models import MODELS

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Phenotype, censor, and ascertain a recorded pedigree; write its cohort",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree", required=True, help="Input recorded pedigree parquet (layout 2)")
    parser.add_argument("--output-cohort", required=True, help="Output cohort parquet")
    parser.add_argument(
        "--output-phenotyped-population", required=True, help="Output phenotyped-population summary YAML"
    )
    parser.add_argument("--seed", type=int, default=42, help="Rep seed for all three phases")

    pheno = parser.add_argument_group("phenotype")
    pheno.add_argument("--G-pheno", type=int, default=3, help="Trailing generations to phenotype")
    pheno.add_argument("--standardize", choices=list(STANDARDIZE_CHOICES), default="global")
    for trait in (1, 2):
        pheno.add_argument(f"--phenotype-model{trait}", choices=sorted(MODELS), required=True)
        pheno.add_argument(f"--beta{trait}", type=float, default=1.0)
        pheno.add_argument(f"--beta-sex{trait}", type=float, default=0.0)
        pheno.add_argument(
            f"--phenotype-params{trait}",
            type=yaml_mapping,
            required=True,
            help=f"Trait {trait} model parameters as a YAML flow mapping",
        )

    censor = parser.add_argument_group("censor")
    censor.add_argument("--censor-age", type=float, default=100, help="Maximum follow-up age")
    censor.add_argument("--death-scale", type=float, default=79.433, help="Competing death hazard scale")
    censor.add_argument("--death-rho", type=float, default=10, help="Competing death hazard shape")
    censor.add_argument(
        "--gen-censoring", type=generation_map, default={}, help="Per-generation censoring windows as JSON dict"
    )

    ascertain = parser.add_argument_group("ascertain")
    ascertain.add_argument("--dropout-rate", type=float, default=0.0, help="Fraction of pedigree to drop uniformly")
    ascertain.add_argument("--case-ascertainment-ratio", type=float, default=1.0, help="Case weight vs controls")
    ascertain.add_argument("--N-sample", type=int, default=0, help="Target sample size (0 = pass-through)")

    args = parser.parse_args(argv)
    init_logging(args)

    cohort, phenotyped_population = run_cohort(
        read_pedigree(args.pedigree),
        seed=args.seed,
        phenotype={
            "G_pheno": args.G_pheno,
            "standardize": args.standardize,
            **{
                f"{key}{trait}": getattr(args, f"{key}{trait}")
                for trait in (1, 2)
                for key in ("phenotype_model", "beta", "beta_sex", "phenotype_params")
            },
        },
        censor={
            "censor_age": args.censor_age,
            "gen_censoring": args.gen_censoring,
            "death_scale": args.death_scale,
            "death_rho": args.death_rho,
        },
        ascertain={
            "dropout_rate": args.dropout_rate,
            "case_ascertainment_ratio": args.case_ascertainment_ratio,
            "N_sample": args.N_sample,
        },
    )
    with publish(args.output_cohort, args.output_phenotyped_population) as (tmp_cohort, tmp_population):
        write_cohort(cohort, tmp_cohort)
        dump_yaml(phenotyped_population, tmp_population)
