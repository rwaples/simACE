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

from __future__ import annotations

__all__ = ["run_phenotype"]

import argparse
import logging
import time
from typing import Any

import pandas as pd

from simace.core.schema import PEDIGREE, PHENOTYPE, assert_schema
from simace.core.utils import save_parquet
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
    sex = pedigree["sex"].values if "sex" in pedigree.columns else None
    return model.simulate(
        liability=pedigree[f"liability{trait_num}"].values,
        seed=params["seed"] + seed_offset,
        standardize=params["standardize"],
        sex=sex,
        generation=pedigree["generation"].values,
    )


def run_phenotype(pedigree: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Simulate phenotype from pedigree and parameter dict.

    Expected keys in ``params``:
        ``G_pheno``, ``seed``, ``standardize``,
        ``phenotype_model1``, ``phenotype_params1``, ``beta1``, ``beta_sex1``,
        ``phenotype_model2``, ``phenotype_params2``, ``beta2``, ``beta_sex2``.

    Adult / cure_frailty additionally require ``prevalence{N}`` (PR3 will
    move this inside ``phenotype_params{N}``).

    Args:
        pedigree: DataFrame with ``liability1``, ``liability2``, ``generation``,
                  ``sex``, plus the genealogy columns preserved on output.
        params:   simulation parameter dict (see above).

    Returns:
        Phenotype DataFrame with columns ``t1``, ``t2`` (raw event times)
        plus the preserved pedigree columns.
    """
    assert_schema(pedigree, PEDIGREE, where="phenotype input")
    logger.info("Running phenotype simulation for %d individuals", len(pedigree))
    t0 = time.perf_counter()

    max_gen = pedigree["generation"].max()
    min_gen = max_gen - params["G_pheno"] + 1
    if min_gen < 0:
        raise ValueError(f"G_pheno ({params['G_pheno']}) exceeds available generations ({max_gen + 1})")
    pedigree = pedigree[pedigree["generation"] >= min_gen].reset_index(drop=True)

    t1 = _simulate_one_trait(pedigree, params, trait_num=1, seed_offset=0)
    t2 = _simulate_one_trait(pedigree, params, trait_num=2, seed_offset=100)

    phenotype = pd.DataFrame(
        {
            "id": pedigree["id"].values,
            "generation": pedigree["generation"].values,
            "sex": pedigree["sex"].values,
            "household_id": pedigree["household_id"].values,
            "mother": pedigree["mother"].values,
            "father": pedigree["father"].values,
            "twin": pedigree["twin"].values,
            "A1": pedigree["A1"].values,
            "C1": pedigree["C1"].values,
            "E1": pedigree["E1"].values,
            "liability1": pedigree["liability1"].values,
            "A2": pedigree["A2"].values,
            "C2": pedigree["C2"].values,
            "E2": pedigree["E2"].values,
            "liability2": pedigree["liability2"].values,
            "t1": t1,
            "t2": t2,
        }
    )

    assert_schema(phenotype, PHENOTYPE, where="phenotype output")
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
    parser.add_argument("--standardize", action="store_true", default=True)

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

    params: dict[str, Any] = {
        "G_pheno": args.G_pheno,
        "seed": args.seed,
        "standardize": args.standardize,
    }
    for trait in (1, 2):
        model_name = getattr(args, f"phenotype_model{trait}")
        model_cls = MODELS[model_name]
        instance = model_cls.from_cli(args, trait)
        params[f"phenotype_model{trait}"] = model_name
        params[f"phenotype_params{trait}"] = instance.to_params_dict()
        params[f"beta{trait}"] = getattr(args, f"beta{trait}")
        params[f"beta_sex{trait}"] = getattr(args, f"beta_sex{trait}")

    pedigree = pd.read_parquet(args.pedigree)
    phenotype = run_phenotype(pedigree, params)
    save_parquet(phenotype, args.output)
