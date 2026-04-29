"""Liability threshold phenotype model for two correlated traits.

Converts liability to binary affected status using a probit threshold
derived from prevalence: ``threshold = ndtri(1 - K)``.  When
``standardize=True`` (default), liability is standardized within each
generation before thresholding, preserving exact prevalence.  When
``standardize=False``, raw liability is compared to the N(0,1)-scale
threshold, so realised prevalence drifts with the liability variance.

No time-to-event or censoring -- purely binary outcome.
"""

__all__ = ["apply_threshold", "run_threshold"]

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from simace.core._numba_utils import _ndtri_approx
from simace.core.parquet import save_parquet
from simace.core.relationships import SEX_LEVELS
from simace.phenotyping.hazards import standardize_liability

logger = logging.getLogger(__name__)


def apply_threshold(
    liability: np.ndarray,
    generation: np.ndarray,
    prevalence: float | dict[int, float],
    standardize: bool = True,
) -> np.ndarray:
    """Apply liability threshold model per generation.

    The threshold is ``ndtri(1 - K)`` where *K* is the prevalence.  When
    *standardize* is True, liability is standardized within each generation
    to N(0,1) before comparison, so realised prevalence matches *K*.
    When False, raw liability is used and prevalence drifts with variance.

    Args:
        liability: array of liability values
        generation: array of generation labels (same length as liability)
        prevalence: fraction affected per generation — either a single float
            applied to all generations, or a dict mapping generation number
            to prevalence (e.g. {0: 0.05, 1: 0.08, 2: 0.10})
        standardize: if True, standardize liability per-generation before
            thresholding (preserves exact prevalence)

    Returns:
        affected: boolean array (True = affected)

    Raises:
        ValueError: if any prevalence value is not in (0, 1), or if a dict
            is provided but is missing entries for observed generations
    """
    unique_gens = np.unique(generation)

    if isinstance(prevalence, dict):
        # Validate: every phenotyped generation must have an entry
        missing = [int(g) for g in unique_gens if int(g) not in prevalence]
        if missing:
            raise ValueError(
                f"prevalence dict is missing entries for generations: {missing}. "
                f"Dict has keys {sorted(prevalence.keys())}, "
                f"but data contains generations {sorted(int(g) for g in unique_gens)}"
            )
        for gen_key, prev_val in prevalence.items():
            if not (0 < prev_val < 1):
                raise ValueError(
                    f"prevalence must be between 0 and 1 (exclusive), got {prev_val} for generation {gen_key}"
                )
    else:
        if not (0 < prevalence < 1):
            raise ValueError(f"prevalence must be between 0 and 1 (exclusive), got {prevalence}")

    affected = np.zeros(len(liability), dtype=bool)
    for gen in unique_gens:
        mask = generation == gen
        L = standardize_liability(liability[mask], standardize)
        prev = prevalence[int(gen)] if isinstance(prevalence, dict) else prevalence
        threshold = _ndtri_approx(1.0 - prev)
        affected[mask] = threshold <= L
    return affected


def _apply_threshold_sex_aware(
    liability: np.ndarray,
    generation: np.ndarray,
    sex: np.ndarray,
    params: dict[str, Any],
    trait_num: int,
) -> np.ndarray:
    """Apply threshold with optional sex-specific prevalence.

    When ``prevalence{N}`` is a dict with ``"female"`` and ``"male"`` keys,
    thresholds are applied separately per sex.  Each sex value may itself
    be a scalar or a per-generation dict (composing naturally with
    ``apply_threshold``).  Otherwise falls back to the standard
    ``apply_threshold`` with the scalar/dict prevalence directly.
    """
    prev = params[f"prevalence{trait_num}"]
    standardize = params.get("standardize", True)
    if isinstance(prev, dict) and "female" in prev and "male" in prev:
        affected = np.zeros(len(liability), dtype=bool)
        for sex_val, key in SEX_LEVELS:
            mask = sex == sex_val
            affected[mask] = apply_threshold(liability[mask], generation[mask], prev[key], standardize=standardize)
        return affected
    return apply_threshold(liability, generation, prev, standardize=standardize)


_DEFAULT_THRESHOLD_PREVALENCE: tuple[float, float] = (0.10, 0.20)


def run_threshold(
    pedigree: pd.DataFrame,
    *,
    phenotype_params1: dict | None,
    phenotype_params2: dict | None,
    G_pheno: int,
    standardize: bool = True,
) -> pd.DataFrame:
    """Orchestrate threshold phenotype from pedigree and per-trait params.

    Prevalence is extracted from the per-trait ``phenotype_params{N}`` dicts
    (the canonical home for adult / cure_frailty model prevalence after PR3).
    Traits whose primary model doesn't carry one (frailty / first_passage)
    fall back to a documented default so the model-agnostic threshold path
    still has a prevalence to use.

    Args:
        pedigree: DataFrame with pedigree data.
        phenotype_params1: trait-1 model-specific param dict.  Its
            ``"prevalence"`` value (scalar, per-generation dict, or
            sex-specific dict; see :func:`apply_threshold`) is used as the
            threshold target.
        phenotype_params2: trait-2 model-specific param dict (same shape
            options as ``phenotype_params1``).
        G_pheno: number of trailing generations to phenotype.
        standardize: if True, standardize liability per-generation before
            thresholding.

    Returns:
        phenotype DataFrame
    """
    pp1 = dict(phenotype_params1 or {})
    pp2 = dict(phenotype_params2 or {})
    prevalence1 = pp1.get("prevalence", _DEFAULT_THRESHOLD_PREVALENCE[0])
    prevalence2 = pp2.get("prevalence", _DEFAULT_THRESHOLD_PREVALENCE[1])

    if isinstance(prevalence1, dict) or isinstance(prevalence2, dict):
        logger.info("Applying threshold model: prevalence1=%s, prevalence2=%s", prevalence1, prevalence2)
    else:
        logger.info("Applying threshold model: prevalence1=%.3f, prevalence2=%.3f", prevalence1, prevalence2)
    t0 = time.perf_counter()
    # Filter to last G_pheno generations
    max_gen = pedigree["generation"].max()
    min_pheno_gen = max_gen - G_pheno + 1
    assert min_pheno_gen >= 0, f"G_pheno ({G_pheno}) > available generations ({max_gen + 1})"
    pedigree = pedigree[pedigree["generation"] >= min_pheno_gen].reset_index(drop=True)

    generation = pedigree["generation"].values
    sex = pedigree["sex"].values

    helper_params = {"prevalence1": prevalence1, "prevalence2": prevalence2, "standardize": standardize}
    affected1 = _apply_threshold_sex_aware(
        pedigree["liability1"].values,
        generation,
        sex,
        helper_params,
        trait_num=1,
    )
    affected2 = _apply_threshold_sex_aware(
        pedigree["liability2"].values,
        generation,
        sex,
        helper_params,
        trait_num=2,
    )

    phenotype = pedigree.assign(affected1=affected1, affected2=affected2)

    elapsed = time.perf_counter() - t0
    logger.info("Threshold model complete in %.1fs: %d individuals", elapsed, len(phenotype))

    return phenotype


def _parse_prevalence_arg(scalar: float | None, by_gen_json: str | None) -> float | dict[int, float] | None:
    """Resolve prevalence from scalar flag or JSON by-gen flag."""
    import json

    if by_gen_json is not None:
        raw = json.loads(by_gen_json)
        return {int(k): float(v) for k, v in raw.items()}
    return scalar


def cli() -> None:
    """Command-line interface for threshold phenotype simulation."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Apply liability threshold model")
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True, help="Input pedigree parquet")
    parser.add_argument("--output", required=True, help="Output phenotype parquet")
    parser.add_argument("--G-pheno", type=int, default=3, help="Number of generations to assign phenotypes")
    parser.add_argument("--prevalence1", type=float, default=0.1, help="Disease prevalence for trait 1")
    parser.add_argument("--prevalence2", type=float, default=0.1, help="Disease prevalence for trait 2")
    parser.add_argument(
        "--prevalence1-by-gen",
        type=str,
        default=None,
        help='Per-gen prevalence for trait 1 as JSON, e.g. \'{"0":0.05,"1":0.10}\'',
    )
    parser.add_argument(
        "--prevalence2-by-gen",
        type=str,
        default=None,
        help='Per-gen prevalence for trait 2 as JSON, e.g. \'{"0":0.05,"1":0.10}\'',
    )
    args = parser.parse_args()

    init_logging(args)

    pedigree = pd.read_parquet(args.pedigree)
    phenotype = run_threshold(
        pedigree,
        G_pheno=args.G_pheno,
        phenotype_params1={"prevalence": _parse_prevalence_arg(args.prevalence1, args.prevalence1_by_gen)},
        phenotype_params2={"prevalence": _parse_prevalence_arg(args.prevalence2, args.prevalence2_by_gen)},
    )
    save_parquet(phenotype, args.output)
