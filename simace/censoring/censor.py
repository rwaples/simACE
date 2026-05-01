"""Observation censoring for simulated phenotypes.

Applies age-window censoring and competing-risk death censoring
to raw event times produced by the Weibull frailty phenotype model.
"""

__all__ = ["age_censor", "death_censor", "run_censor"]

import argparse
import logging
import time

import numpy as np
import pandas as pd

from simace.core.parquet import save_parquet
from simace.core.schema import CENSORED, PHENOTYPE
from simace.core.stage import stage

logger = logging.getLogger(__name__)


def age_censor(t: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-individual [left, right] age censoring.

    - t < left: left-censored (onset before observation window), t set to left
    - t > right: right-censored (onset after observation window), t set to right

    A zero-width window (left == right, e.g. ``[80, 80]``) fully censors the
    generation: every individual is flagged as censored because no continuous
    onset time can fall strictly within a zero-length interval.

    Args:
        t: array of time-to-onset values
        left: array of left-censoring ages (per individual)
        right: array of right-censoring ages (per individual)

    Returns:
        (t_censored, age_censored): tuple of arrays
    """
    left_trunc = t < left
    right_cens = t > right
    censored = left_trunc | right_cens
    t_out = np.clip(t, left, right)

    return t_out, censored


def death_censor(
    t: np.ndarray, seed: int, scale: float = 79.43282347242817, rho: float = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Apply competing risk death censoring with Weibull hazard.

    Args:
        t: array of time-to-onset values
        seed: random seed
        scale: Weibull scale parameter for death hazard
        rho: Weibull shape parameter for death hazard

    Returns:
        (t_censored, death_censored): tuple of arrays
    """
    rng = np.random.default_rng(seed)
    u = 1.0 - rng.uniform(size=len(t))
    dt = scale * (-np.log(u)) ** (1 / rho)
    censored = t > dt

    t[censored] = dt[censored]

    return t, censored


@stage(reads=PHENOTYPE, writes=CENSORED)
def run_censor(
    phenotype: pd.DataFrame,
    *,
    censor_age: float,
    seed: int,
    gen_censoring: dict[int, list[float]],
    death_scale: float,
    death_rho: float,
) -> pd.DataFrame:
    """Apply censoring to raw phenotype event times.

    Args:
        phenotype: DataFrame with raw event times (t1, t2) from run_phenotype.
        censor_age: maximum follow-up age (right boundary of the default
            observation window).
        seed: RNG seed for the competing-risk death draw.
        gen_censoring: per-generation ``{gen: [left, right]}`` observation
            windows.  Generations not listed use ``[0, censor_age]``.
        death_scale: Weibull scale for the competing-risk death hazard.
        death_rho: Weibull shape for the competing-risk death hazard.

    Returns:
        DataFrame with original columns plus censoring columns:
        death_age, age_censored1/2, t_observed1/2, death_censored1/2, affected1/2
    """
    logger.info("Running censoring for %d individuals", len(phenotype))
    t0 = time.perf_counter()

    generations = phenotype["generation"].values
    left_censor = np.zeros(len(phenotype))
    right_censor = np.full(len(phenotype), float(censor_age))
    for gen, (lo, hi) in gen_censoring.items():
        mask = generations == int(gen)
        left_censor[mask] = lo
        right_censor[mask] = hi

    rng_death = np.random.default_rng(seed + 1000)
    u_death = 1.0 - rng_death.uniform(size=len(phenotype))
    death_age = death_scale * (-np.log(u_death)) ** (1 / death_rho)

    t1_after_age, age_censored1 = age_censor(phenotype["t1"].values.copy(), left_censor, right_censor)
    death_censored1 = t1_after_age > death_age
    t_observed1 = np.where(death_censored1, death_age, t1_after_age)

    t2_after_age, age_censored2 = age_censor(phenotype["t2"].values.copy(), left_censor, right_censor)
    death_censored2 = t2_after_age > death_age
    t_observed2 = np.where(death_censored2, death_age, t2_after_age)

    result = phenotype.copy()
    result["death_age"] = death_age
    result["age_censored1"] = age_censored1
    result["t_observed1"] = t_observed1
    result["death_censored1"] = death_censored1
    result["affected1"] = ~age_censored1 & ~death_censored1
    result["age_censored2"] = age_censored2
    result["t_observed2"] = t_observed2
    result["death_censored2"] = death_censored2
    result["affected2"] = ~age_censored2 & ~death_censored2

    prev1 = result["affected1"].mean()
    prev2 = result["affected2"].mean()
    logger.info("Prevalence after censoring: trait1=%.3f, trait2=%.3f", prev1, prev2)

    elapsed = time.perf_counter() - t0
    logger.info("Censoring complete in %.1fs: %d individuals", elapsed, len(result))

    return result


def cli() -> None:
    """Command-line interface for censoring phenotype data."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Apply observation censoring to phenotype data")
    add_logging_args(parser)
    parser.add_argument("--phenotype", required=True, help="Input raw phenotype parquet")
    parser.add_argument("--output", required=True, help="Output censored phenotype parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--censor-age", type=float, default=100, help="Maximum follow-up age")
    parser.add_argument("--death-scale", type=float, default=79.433, help="Competing death hazard scale")
    parser.add_argument("--death-rho", type=float, default=10, help="Competing death hazard shape")
    parser.add_argument(
        "--gen-censoring",
        type=str,
        default=None,
        help='Per-generation censoring windows as JSON dict, e.g. \'{"0": [40, 80], "3": [0, 45]}\'',
    )

    args = parser.parse_args()

    init_logging(args)

    import json

    phenotype = pd.read_parquet(args.phenotype)
    gen_censoring = json.loads(args.gen_censoring) if args.gen_censoring else {}
    gen_censoring = {int(k): v for k, v in gen_censoring.items()}
    result = run_censor(
        phenotype,
        censor_age=args.censor_age,
        seed=args.seed,
        gen_censoring=gen_censoring,
        death_scale=args.death_scale,
        death_rho=args.death_rho,
    )
    save_parquet(result, args.output)
