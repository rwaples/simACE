"""Observation censoring for simulated phenotypes.

Applies age-window censoring and competing-risk death censoring
to raw event times produced by the Weibull frailty phenotype model.
"""

from __future__ import annotations

__all__ = ["add_censor_args", "age_censor", "run_censor"]

import argparse
import logging
import time

import numpy as np
import polars as pl

from simace.core.cli_base import CENSOR_KEYS
from simace.core.parquet import load_parquet, save_parquet
from simace.core.schema import PEDIGREE, assert_schema
from simace.core.stage import stage
from simace.core.trait_schema import CENSORED_TRAIT, RAW_TRAIT, hydrate_trait, strip_trait_to_outcomes

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


def _censor_trait(
    trait: int, onset: np.ndarray, left: np.ndarray, right: np.ndarray, death_age: np.ndarray
) -> list[pl.Series]:
    """Censor one trait's onsets by the observation window, then by the shared death age."""
    after_age, age_censored = age_censor(onset, left, right)
    death_censored = after_age > death_age
    return [
        pl.Series(f"age_censored{trait}", age_censored),
        pl.Series(f"t_observed{trait}", np.where(death_censored, death_age, after_age)),
        pl.Series(f"death_censored{trait}", death_censored),
        pl.Series(f"affected{trait}", ~age_censored & ~death_censored),
    ]


@stage(reads=RAW_TRAIT, writes=CENSORED_TRAIT)
def run_censor(
    phenotype: pl.DataFrame,
    pedigree: pl.DataFrame,
    *,
    censor_age: float,
    seed: int,
    gen_censoring: dict[int, list[float]],
    death_scale: float,
    death_rho: float,
) -> pl.DataFrame:
    """Apply censoring to raw phenotype event times.

    Args:
        phenotype: Outcomes-only DataFrame with raw event times (id, t1, t2)
            from run_phenotype. A null raw onset means the individual never
            onsets and is censored at the applicable finite boundary.
        pedigree: Pedigree DataFrame for the same IDs, used to hydrate
            generation-specific censoring windows.
        censor_age: maximum follow-up age (right boundary of the default
            observation window).
        seed: RNG seed for the competing-risk death draw.
        gen_censoring: per-generation ``{gen: [left, right]}`` observation
            windows.  Generations not listed use ``[0, censor_age]``.
        death_scale: Weibull scale for the competing-risk death hazard.
        death_rho: Weibull shape for the competing-risk death hazard.

    Returns:
        Outcomes-only DataFrame with id, raw event times, and censoring columns:
        death_age, age_censored1/2, t_observed1/2, death_censored1/2, affected1/2.
    """
    logger.info("Running censoring for %d individuals", len(phenotype))
    t0 = time.perf_counter()

    assert_schema(pedigree, PEDIGREE, where="censor pedigree input")
    hydrated = hydrate_trait(phenotype, pedigree, kind="raw", columns=["generation"])
    onset1 = hydrated["t1"].fill_null(float("inf")).to_numpy()
    onset2 = hydrated["t2"].fill_null(float("inf")).to_numpy()
    generations = hydrated["generation"].to_numpy()
    left_censor = np.zeros(len(phenotype))
    right_censor = np.full(len(phenotype), float(censor_age))
    for gen, (lo, hi) in gen_censoring.items():
        mask = generations == int(gen)
        left_censor[mask] = lo
        right_censor[mask] = hi

    rng_death = np.random.default_rng(seed + 1000)
    u_death = 1.0 - rng_death.uniform(size=len(phenotype))
    death_age = death_scale * (-np.log(u_death)) ** (1 / death_rho)

    columns = [pl.Series("death_age", death_age)]
    for trait, onset in ((1, onset1), (2, onset2)):
        columns += _censor_trait(trait, onset, left_censor, right_censor, death_age)
    result = phenotype.with_columns(columns)

    logger.info(
        "Prevalence after censoring: trait1=%.3f, trait2=%.3f", result["affected1"].mean(), result["affected2"].mean()
    )

    result = strip_trait_to_outcomes(result, "censored")

    elapsed = time.perf_counter() - t0
    logger.info("Censoring complete in %.1fs: %d individuals", elapsed, len(result))

    return result


def add_censor_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    """Add the censoring flags.

    The scalar defaults match ``censoring`` in ``config/_default.yaml``. ``--gen-censoring``
    defaults to no per-generation windows (every generation observed over
    ``[0, censor_age]``), not to that file's ``gen_censoring`` map.
    """
    from simace.core.cli_base import generation_map

    parser.add_argument("--censor-age", type=float, default=80, help="Maximum follow-up age")
    parser.add_argument("--death-scale", type=float, default=164, help="Competing death hazard scale")
    parser.add_argument("--death-rho", type=float, default=2.73, help="Competing death hazard shape")
    parser.add_argument(
        "--gen-censoring",
        type=generation_map,
        default={},
        help='Per-generation censoring windows as JSON dict, e.g. \'{"0": [40, 80], "3": [0, 45]}\'',
    )


def cli(argv: list[str] | None = None, prog: str | None = None) -> None:
    """Command-line interface for censoring phenotype data."""
    from simace.core.cli_base import add_logging_args, init_logging
    from simace.core.publish import publish

    parser = argparse.ArgumentParser(prog=prog, description="Apply observation censoring to phenotype data")
    add_logging_args(parser)
    parser.add_argument("--phenotype", required=True, help="Input raw trait parquet")
    parser.add_argument(
        "--pedigree", required=True, help="Input pedigree parquet used for generation-specific censoring"
    )
    parser.add_argument("--output", required=True, help="Output censored trait parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    add_censor_args(parser)

    args = parser.parse_args(argv)

    init_logging(args)

    phenotype = load_parquet(args.phenotype)
    pedigree = load_parquet(args.pedigree)
    result = run_censor(phenotype, pedigree, seed=args.seed, **{k: getattr(args, k) for k in CENSOR_KEYS})
    with publish(args.output) as (tmp,):
        save_parquet(result, tmp)
