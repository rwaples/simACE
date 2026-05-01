"""Liability threshold phenotype model for two correlated traits.

Converts liability to binary affected status using a probit threshold
derived from prevalence: ``threshold = ndtri(1 - K)``.  Standardization
is controlled by the global ``standardize`` flag with three modes:

* ``"global"`` (default) — z-score across the whole cohort once, then
  compare to ``ndtri(1 - K)``.  Preserves prevalence at *K* only when the
  cohort liability is already centered with unit variance.
* ``"per_generation"`` — z-score within each generation independently
  before comparison; preserves *K* per generation regardless of how
  liability variance drifts across generations.
* ``"none"`` — compare raw liability to the N(0,1)-scale threshold;
  realised prevalence drifts with the liability variance.

For sex-specific prevalence dicts (``{"female": K_f, "male": K_m}``),
liability is standardized once across both sexes (per the selected mode);
sex-shifted liability means therefore translate into sex-specific
realised prevalences within each generation.

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
from simace.core.schema import PEDIGREE, PHENOTYPE
from simace.core.stage import stage
from simace.phenotyping.hazards import StandardizeMode, coerce_standardize_mode, standardize_liability

logger = logging.getLogger(__name__)


def _validate_prevalence_value(value: float, where: str) -> None:
    if not (0 < value < 1):
        raise ValueError(f"prevalence must be between 0 and 1 (exclusive), got {value}{where}")


def _validate_per_gen_prevalence(prev_dict: dict[int, float], unique_gens: np.ndarray) -> None:
    missing = [int(g) for g in unique_gens if int(g) not in prev_dict]
    if missing:
        raise ValueError(
            f"prevalence dict is missing entries for generations: {missing}. "
            f"Dict has keys {sorted(prev_dict.keys())}, "
            f"but data contains generations {sorted(int(g) for g in unique_gens)}"
        )
    for gen_key, prev_val in prev_dict.items():
        _validate_per_gen_value(prev_val, f" for generation {gen_key}")


def _validate_per_gen_value(prev_val: float, where: str) -> None:
    _validate_prevalence_value(prev_val, where)


def _resolve_prev_for_gen(prev_spec: float | dict[int, float], gen: int) -> float:
    if isinstance(prev_spec, dict):
        return prev_spec[int(gen)]
    return prev_spec


def _apply_thresholds_to_standardized(
    L_all: np.ndarray,
    generation: np.ndarray,
    sex: np.ndarray | None,
    prev_spec: float | dict,
) -> np.ndarray:
    """Apply per-cell thresholds to an already-standardized liability array.

    ``prev_spec`` may be a scalar, a per-generation dict, or a
    ``{"female": ..., "male": ...}`` dict whose values are themselves
    scalars or per-generation dicts. Loops over generations (and sex when
    sex-keyed) and writes ``threshold[g] <= L_all`` into the output.
    """
    unique_gens = np.unique(generation)
    affected = np.zeros(len(L_all), dtype=bool)

    if isinstance(prev_spec, dict) and "female" in prev_spec and "male" in prev_spec:
        if sex is None:
            raise ValueError("sex array required for sex-keyed prevalence dict")
        for sex_val, key in SEX_LEVELS:
            sub_spec = prev_spec[key]
            if isinstance(sub_spec, dict):
                _validate_per_gen_prevalence(sub_spec, unique_gens)
            else:
                _validate_prevalence_value(sub_spec, "")
            sex_mask = sex == sex_val
            for gen in unique_gens:
                cell = sex_mask & (generation == gen)
                if not cell.any():
                    continue
                prev = _resolve_prev_for_gen(sub_spec, int(gen))
                threshold = _ndtri_approx(1.0 - prev)
                affected[cell] = threshold <= L_all[cell]
        return affected

    if isinstance(prev_spec, dict):
        _validate_per_gen_prevalence(prev_spec, unique_gens)
    else:
        _validate_prevalence_value(prev_spec, "")
    for gen in unique_gens:
        mask = generation == gen
        prev = _resolve_prev_for_gen(prev_spec, int(gen))
        threshold = _ndtri_approx(1.0 - prev)
        affected[mask] = threshold <= L_all[mask]
    return affected


def apply_threshold(
    liability: np.ndarray,
    generation: np.ndarray,
    prevalence: float | dict[int, float],
    standardize: StandardizeMode | bool = "global",
) -> np.ndarray:
    """Apply liability threshold model per generation.

    The threshold is ``ndtri(1 - K)`` where *K* is the prevalence.  Liability
    is standardized once according to *standardize* (``"none"``, ``"global"``,
    or ``"per_generation"``); the per-generation comparison then uses the
    already-standardized values.  Bools accepted for back-compat
    (``True`` → ``"global"``, ``False`` → ``"none"``).

    Args:
        liability: array of liability values
        generation: array of generation labels (same length as liability)
        prevalence: fraction affected per generation — either a single float
            applied to all generations, or a dict mapping generation number
            to prevalence (e.g. {0: 0.05, 1: 0.08, 2: 0.10})
        standardize: standardization mode for liability before thresholding

    Returns:
        affected: boolean array (True = affected)

    Raises:
        ValueError: if any prevalence value is not in (0, 1), or if a dict
            is provided but is missing entries for observed generations
    """
    mode = coerce_standardize_mode(standardize)
    L_all = standardize_liability(liability, mode, generation)
    return _apply_thresholds_to_standardized(L_all, generation, sex=None, prev_spec=prevalence)


def _apply_threshold_sex_aware(
    liability: np.ndarray,
    generation: np.ndarray,
    sex: np.ndarray,
    params: dict[str, Any],
    trait_num: int,
) -> np.ndarray:
    """Apply threshold with optional sex-specific prevalence.

    When ``prevalence{N}`` is a dict with ``"female"`` and ``"male"`` keys,
    liability is standardized once across both sexes (so sex-shifted means
    yield sex-specific realised prevalences) and then thresholded
    per-(sex, gen) cell.  Otherwise behaves identically to
    :func:`apply_threshold` with the scalar/dict prevalence directly.
    """
    prev = params[f"prevalence{trait_num}"]
    mode = coerce_standardize_mode(params.get("standardize", "global"))
    L_all = standardize_liability(liability, mode, generation)
    if isinstance(prev, dict) and "female" in prev and "male" in prev:
        return _apply_thresholds_to_standardized(L_all, generation, sex, prev)
    return _apply_thresholds_to_standardized(L_all, generation, sex=None, prev_spec=prev)


_DEFAULT_THRESHOLD_PREVALENCE: tuple[float, float] = (0.10, 0.20)


@stage(reads=PEDIGREE, writes=PHENOTYPE)
def run_threshold(
    pedigree: pd.DataFrame,
    *,
    phenotype_params1: dict | None,
    phenotype_params2: dict | None,
    G_pheno: int,
    standardize: StandardizeMode | bool = "global",
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
        standardize: standardization mode for liability before thresholding;
            one of ``"none"``, ``"global"``, ``"per_generation"``.  Bools
            accepted for back-compat (``True`` → ``"global"``,
            ``False`` → ``"none"``).

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

    nan_t = np.full(len(pedigree), np.nan, dtype=np.float64)
    phenotype = pedigree.assign(affected1=affected1, affected2=affected2, t1=nan_t, t2=nan_t)

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
    parser.add_argument(
        "--standardize",
        choices=["none", "global", "per_generation"],
        default="global",
        help="Liability standardization mode (default: global)",
    )
    args = parser.parse_args()

    init_logging(args)

    pedigree = pd.read_parquet(args.pedigree)
    phenotype = run_threshold(
        pedigree,
        G_pheno=args.G_pheno,
        phenotype_params1={"prevalence": _parse_prevalence_arg(args.prevalence1, args.prevalence1_by_gen)},
        phenotype_params2={"prevalence": _parse_prevalence_arg(args.prevalence2, args.prevalence2_by_gen)},
        standardize=args.standardize,
    )
    save_parquet(phenotype, args.output)
