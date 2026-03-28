"""Subsample phenotyped individuals before stats/plotting."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from sim_ace.core.utils import save_parquet

logger = logging.getLogger(__name__)


def run_sample(phenotype: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Optionally subsample phenotyped individuals.

    Args:
        phenotype: DataFrame of phenotyped individuals.
        params: dict with keys ``N_sample`` (int), ``seed`` (int), and
            optionally ``case_ascertainment_ratio`` (float, default 1.0).
            When ratio != 1, cases (``affected1 == True``) are sampled
            with weight=ratio relative to controls (weight=1).

    Returns:
        DataFrame with at most ``N_sample`` rows (or all rows if
        ``N_sample <= 0`` or ``N_sample >= len(phenotype)``).
    """
    n_total = len(phenotype)
    n_sample = int(params.get("N_sample", 0))
    seed = int(params.get("seed", 42))
    ratio = float(params.get("case_ascertainment_ratio", 1.0))

    if ratio < 0:
        raise ValueError(f"case_ascertainment_ratio must be >= 0, got {ratio}")

    if n_sample <= 0 or n_sample >= n_total:
        if ratio != 1.0:
            logger.warning(
                "case_ascertainment_ratio=%.2f has no effect when N_sample=%d (all %d individuals passed through)",
                ratio,
                n_sample,
                n_total,
            )
        logger.info("Sample pass-through: keeping all %d individuals (N_sample=%d)", n_total, n_sample)
        return phenotype

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    if ratio == 1.0:
        # Fast path: uniform sampling
        indices = np.sort(rng.choice(n_total, n_sample, replace=False))
    else:
        # Weighted sampling: cases get weight=ratio, controls get weight=1
        is_case = phenotype["affected1"].values.astype(bool)
        n_cases = int(is_case.sum())
        n_controls = n_total - n_cases

        if n_cases == 0:
            logger.warning(
                "No cases found (affected1); ignoring case_ascertainment_ratio=%.2f, falling back to uniform sampling",
                ratio,
            )
            indices = np.sort(rng.choice(n_total, n_sample, replace=False))
        elif n_cases == n_total:
            logger.warning(
                "All individuals are cases (affected1); ignoring "
                "case_ascertainment_ratio=%.2f, falling back to uniform sampling",
                ratio,
            )
            indices = np.sort(rng.choice(n_total, n_sample, replace=False))
        elif ratio == 0:
            # Only controls sampled
            actual_n = min(n_sample, n_controls)
            if actual_n < n_sample:
                logger.warning(
                    "case_ascertainment_ratio=0: clamping N_sample from %d to %d (only %d controls available)",
                    n_sample,
                    actual_n,
                    n_controls,
                )
            control_indices = np.where(~is_case)[0]
            indices = np.sort(rng.choice(control_indices, actual_n, replace=False))
        else:
            weights = np.where(is_case, ratio, 1.0)
            probabilities = weights / weights.sum()

            # Warn if >90% of cases would be expected in sample
            expected_case_prob = (ratio * n_cases) / (ratio * n_cases + n_controls)
            expected_cases_drawn = expected_case_prob * n_sample
            if expected_cases_drawn > 0.9 * n_cases:
                logger.warning(
                    "Extreme ascertainment: expected to draw %.0f of %d total cases (ratio=%.1f, N_sample=%d)",
                    expected_cases_drawn,
                    n_cases,
                    ratio,
                    n_sample,
                )

            indices = np.sort(rng.choice(n_total, n_sample, replace=False, p=probabilities))

    result = phenotype.iloc[indices].reset_index(drop=True)
    elapsed = time.perf_counter() - t0

    if ratio != 1.0:
        n_cases_sampled = int(result["affected1"].sum())
        logger.info(
            "Sampled %d → %d individuals (cases: %d, %.1f%%) in %.1fs (seed=%d, ratio=%.1f)",
            n_total,
            len(result),
            n_cases_sampled,
            100.0 * n_cases_sampled / len(result) if len(result) > 0 else 0,
            elapsed,
            seed,
            ratio,
        )
    else:
        logger.info("Sampled %d → %d individuals in %.1fs (seed=%d)", n_total, n_sample, elapsed, seed)
    return result


def cli() -> None:
    """Command-line interface for subsampling phenotype data."""
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Subsample phenotyped individuals")
    add_logging_args(parser)
    parser.add_argument("--phenotype", required=True, help="Input phenotype parquet")
    parser.add_argument("--output", required=True, help="Output sampled phenotype parquet")
    parser.add_argument("--N-sample", type=int, default=0, help="Number of individuals to retain (0 = keep all)")
    parser.add_argument(
        "--case-ascertainment-ratio",
        type=float,
        default=1.0,
        help="Sampling weight for cases relative to controls (1.0 = uniform)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    init_logging(args)

    phenotype = pd.read_parquet(args.phenotype)
    params = {
        "N_sample": args.N_sample,
        "case_ascertainment_ratio": args.case_ascertainment_ratio,
        "seed": args.seed,
    }
    result = run_sample(phenotype, params)
    save_parquet(result, args.output)
