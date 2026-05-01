"""Phenotype simulation for two correlated traits.

Each trait is simulated by one of the four model families registered in
``simace.phenotyping.models``:

  * ``frailty``       — proportional hazards frailty (baseline hazards live
                        in ``simace.phenotyping.hazards``).
  * ``cure_frailty``  — mixture cure model: threshold determines case
                        status, frailty determines onset time among cases.
  * ``adult``         — ADuLT age-dependent liability threshold.
  * ``first_passage`` — inverse-Gaussian first-passage time.

Adding a new model is a single new file under
``simace/phenotyping/models/`` plus one entry in
``simace/phenotyping/models/__init__.py``'s ``MODELS`` dict.
"""

__all__ = ["run_phenotype"]

import argparse
import logging
import time
from typing import Any

import pandas as pd

from simace.core.parquet import save_parquet
from simace.core.schema import PEDIGREE, PHENOTYPE
from simace.core.stage import stage
from simace.phenotyping.hazards import StandardizeMode
from simace.phenotyping.models import MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _simulate_one_trait(
    pedigree: pd.DataFrame,
    params: dict[str, Any],
    trait_num: int,
    seed_offset: int,
):
    """Dispatch a single trait to the appropriate phenotype model class."""
    model_cls = MODELS[params[f"phenotype_model{trait_num}"]]
    model = model_cls.from_config(params, trait_num)
    sex = pedigree["sex"].to_numpy() if "sex" in pedigree.columns else None
    return model.simulate(
        liability=pedigree[f"liability{trait_num}"].to_numpy(),
        seed=params["seed"] + seed_offset,
        standardize=params["standardize"],
        sex=sex,
        generation=pedigree["generation"].to_numpy(),
    )


@stage(reads=PEDIGREE, writes=PHENOTYPE)
def run_phenotype(
    pedigree: pd.DataFrame,
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
) -> pd.DataFrame:
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
            ``adult``, ``first_passage``).
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

    max_gen = pedigree["generation"].max()
    min_gen = max_gen - G_pheno + 1
    if min_gen < 0:
        raise ValueError(f"G_pheno ({G_pheno}) exceeds available generations ({max_gen + 1})")
    pedigree = pedigree[pedigree["generation"] >= min_gen].reset_index(drop=True)

    helper_params: dict[str, Any] = {
        "G_pheno": G_pheno,
        "seed": seed,
        "standardize": standardize,
        "phenotype_model1": phenotype_model1,
        "phenotype_model2": phenotype_model2,
        "beta1": beta1,
        "beta_sex1": beta_sex1,
        "phenotype_params1": phenotype_params1,
        "beta2": beta2,
        "beta_sex2": beta_sex2,
        "phenotype_params2": phenotype_params2,
    }
    t1 = _simulate_one_trait(pedigree, helper_params, trait_num=1, seed_offset=0)
    t2 = _simulate_one_trait(pedigree, helper_params, trait_num=2, seed_offset=100)

    phenotype = pedigree.assign(t1=t1, t2=t2)

    logger.info(
        "Phenotype simulation complete in %.1fs: %d individuals",
        time.perf_counter() - t0,
        len(phenotype),
    )
    return phenotype


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    """Command-line entry point for phenotype simulation.

    Eager-registration scheme: every model's flag set is registered up
    front so ``--help`` shows them all in clearly-labeled per-family
    argument groups. Each model's ``from_cli`` rejects flags belonging to
    a different family when invoked alongside that model's selection.
    """
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(
        description="Simulate phenotype event times for two correlated traits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--G-pheno", type=int, default=3)
    parser.add_argument(
        "--standardize",
        choices=["none", "global", "per_generation"],
        default="global",
        help="Liability standardization mode (default: global)",
    )

    for trait in (1, 2):
        shared = parser.add_argument_group(f"Trait {trait} — shared")
        shared.add_argument(
            f"--phenotype-model{trait}",
            choices=sorted(MODELS),
            required=True,
            help=f"Phenotype model family for trait {trait}",
        )
        shared.add_argument(f"--beta{trait}", type=float, default=1.0)
        shared.add_argument(f"--beta-sex{trait}", type=float, default=0.0)
        for model_cls in MODELS.values():
            model_cls.add_cli_args(parser, trait)

    args = parser.parse_args()
    init_logging(args)

    kwargs: dict[str, Any] = {
        "G_pheno": args.G_pheno,
        "seed": args.seed,
        "standardize": args.standardize,
    }
    for trait in (1, 2):
        model_name = getattr(args, f"phenotype_model{trait}")
        model_cls = MODELS[model_name]
        instance = model_cls.from_cli(args, trait)
        kwargs[f"phenotype_model{trait}"] = model_name
        kwargs[f"phenotype_params{trait}"] = instance.to_params_dict()
        kwargs[f"beta{trait}"] = getattr(args, f"beta{trait}")
        kwargs[f"beta_sex{trait}"] = getattr(args, f"beta_sex{trait}")

    pedigree = pd.read_parquet(args.pedigree)
    phenotype = run_phenotype(pedigree, **kwargs)
    save_parquet(phenotype, args.output)
