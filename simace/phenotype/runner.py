"""Two-trait phenotype dispatcher and CLI.

``run_phenotype`` trims the pedigree to the trailing ``G_pheno`` generations
and dispatches each of the two traits to its configured model family (see the
package docstring in :mod:`simace.phenotype` for the family list). ``cli`` is
the ``simace phenotype`` command.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import TYPE_CHECKING, Any

import polars as pl

from simace.core.parquet import load_parquet, save_parquet
from simace.core.schema import PEDIGREE
from simace.core.stage import stage
from simace.core.trait_schema import RAW_TRAIT, strip_trait_to_outcomes
from simace.phenotype.hazards import STANDARDIZE_CHOICES
from simace.phenotype.models import MODELS

if TYPE_CHECKING:
    import numpy as np

    from simace.phenotype.hazards import StandardizeMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _simulate_one_trait(
    pedigree: pl.DataFrame,
    *,
    trait_num: int,
    model_name: str,
    phenotype_params: dict,
    beta: float,
    beta_sex: float,
    seed: int,
    standardize: StandardizeMode | bool,
    sex: np.ndarray,
    generation: np.ndarray,
):
    """Dispatch a single trait to the appropriate phenotype model class."""
    model_cls = MODELS[model_name]
    config = {
        f"phenotype_params{trait_num}": phenotype_params,
        f"beta{trait_num}": beta,
        f"beta_sex{trait_num}": beta_sex,
    }
    model = model_cls.from_config(config, trait_num)
    return model.simulate(
        liability=pedigree[f"liability{trait_num}"].to_numpy(),
        seed=seed,
        standardize=standardize,
        sex=sex,
        generation=generation,
    )


@stage(reads=PEDIGREE, writes=RAW_TRAIT)
def run_phenotype(
    pedigree: pl.DataFrame,
    *,
    G_pheno: int,
    seed: int,
    standardize: StandardizeMode | bool,
    phenotype_model1: str,
    phenotype_model2: str,
    beta1: float,
    beta_sex1: float,
    phenotype_params1: dict,
    beta2: float,
    beta_sex2: float,
    phenotype_params2: dict,
) -> pl.DataFrame:
    """Simulate phenotype event times for two correlated traits.

    Per-trait prevalence (for adult / cure_frailty) lives inside
    ``phenotype_params{N}``; frailty / first_passage do not carry one.

    Args:
        pedigree: DataFrame with ``liability1``, ``liability2``, ``generation``,
            ``sex``, plus the genealogy columns preserved on output.
        G_pheno: number of trailing generations to phenotype.
        seed: RNG seed (trait 2 uses ``seed + 100``).
        standardize: global liability-standardization mode applied to
            threshold-style consumers (``threshold``, ``adult.ltm``,
            ``cure_frailty``'s threshold step). One of ``"none" | "global" |
            "per_generation"``; legacy bools accepted (``True`` → ``"global"``,
            ``False`` → ``"none"``). Hazard-bearing models (``frailty``,
            ``cure_frailty``, ``first_passage``, ``adult.cox``) inherit this
            for their hazard step unless the per-trait
            ``phenotype_params{N}["standardize_hazard"]`` overrides it.
        phenotype_model1: trait-1 model family (``frailty``, ``cure_frailty``,
            ``adult``, ``first_passage``, ``simple_ltm``).
        phenotype_model2: trait-2 model family (same options).
        beta1: trait-1 liability → log-hazard slope.
        beta_sex1: trait-1 sex → log-hazard slope.
        phenotype_params1: trait-1 model-specific parameter dict (e.g.
            ``{"distribution": "weibull", "scale": ..., "rho": ...}``;
            optionally ``"standardize_hazard": "..."``).
        beta2: trait-2 liability → log-hazard slope.
        beta_sex2: trait-2 sex → log-hazard slope.
        phenotype_params2: trait-2 model-specific parameter dict.

    Returns:
        Phenotype DataFrame with columns ``t1``, ``t2`` (raw event times)
        plus the preserved pedigree columns.
    """
    logger.info("Running phenotype simulation for %d individuals", len(pedigree))
    t0 = time.perf_counter()

    max_gen = int(pedigree["generation"].to_numpy().max())
    min_gen = max_gen - G_pheno + 1
    if min_gen < 0:
        raise ValueError(f"G_pheno ({G_pheno}) exceeds available generations ({max_gen + 1})")
    pedigree = pedigree.filter(pl.col("generation") >= min_gen)

    sex = pedigree["sex"].to_numpy()
    generation = pedigree["generation"].to_numpy()
    t1 = _simulate_one_trait(
        pedigree,
        trait_num=1,
        model_name=phenotype_model1,
        phenotype_params=phenotype_params1,
        beta=beta1,
        beta_sex=beta_sex1,
        seed=seed,
        standardize=standardize,
        sex=sex,
        generation=generation,
    )
    t2 = _simulate_one_trait(
        pedigree,
        trait_num=2,
        model_name=phenotype_model2,
        phenotype_params=phenotype_params2,
        beta=beta2,
        beta_sex=beta_sex2,
        seed=seed + 100,
        standardize=standardize,
        sex=sex,
        generation=generation,
    )

    # Null contract (ADR 0015 §3): models return numpy arrays where missing
    # onset is NaN; normalize to null as the arrays re-enter a frame.
    with_times = pedigree.with_columns(pl.Series("t1", t1).fill_nan(None), pl.Series("t2", t2).fill_nan(None))
    phenotype = strip_trait_to_outcomes(with_times, "raw")

    logger.info(
        "Phenotype simulation complete in %.1fs: %d individuals",
        time.perf_counter() - t0,
        len(phenotype),
    )
    return phenotype


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_phenotype_args(group: argparse._ArgumentGroup) -> None:
    """The phenotype-stage flags of a command that phenotypes a recorded pedigree (``phenotype``, ``cohort``, ``validate``)."""
    from simace.core.cli_base import yaml_mapping

    group.add_argument("--G-pheno", type=int, default=3, help="Trailing generations to phenotype")
    group.add_argument("--standardize", choices=list(STANDARDIZE_CHOICES), default="global")
    for trait in (1, 2):
        group.add_argument(f"--phenotype-model{trait}", choices=sorted(MODELS), required=True)
        group.add_argument(f"--beta{trait}", type=float, default=1.0)
        group.add_argument(f"--beta-sex{trait}", type=float, default=0.0)
        group.add_argument(
            f"--phenotype-params{trait}",
            type=yaml_mapping,
            required=True,
            help=f"Trait {trait} model parameters as a YAML flow mapping",
        )


def phenotype_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``run_phenotype``'s keyword arguments other than ``seed``, from :func:`add_phenotype_args`' flags."""
    return {
        "G_pheno": args.G_pheno,
        "standardize": args.standardize,
        **{
            f"{key}{trait}": getattr(args, f"{key}{trait}")
            for trait in (1, 2)
            for key in ("phenotype_model", "beta", "beta_sex", "phenotype_params")
        },
    }


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line entry point for phenotype simulation.

    Takes each trait's model parameters as ``--phenotype-params{N}``, a YAML
    flow mapping of the ``phenotype_params{N}`` config dict, the same flags
    the ``cohort`` stage takes.
    """
    from simace.core.cli_base import add_logging_args, add_version_arg, init_logging
    from simace.core.publish import publish

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Simulate phenotype event times for two correlated traits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_logging_args(parser)
    add_version_arg(parser, "simace")
    parser.add_argument("--pedigree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    add_phenotype_args(parser.add_argument_group("phenotype"))

    args = parser.parse_args(argv)
    init_logging(args)

    kwargs = {"seed": args.seed, **phenotype_kwargs(args)}

    pedigree = load_parquet(args.pedigree)
    phenotype = run_phenotype(pedigree, **kwargs)
    with publish(args.output) as (tmp,):
        save_parquet(phenotype, tmp)
