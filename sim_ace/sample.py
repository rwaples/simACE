"""Subsample phenotyped individuals before stats/plotting."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from sim_ace.utils import save_parquet

logger = logging.getLogger(__name__)


def run_sample(phenotype: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Optionally subsample phenotyped individuals.

    Args:
        phenotype: DataFrame of phenotyped individuals.
        params: dict with keys ``N_sample`` (int) and ``seed`` (int).

    Returns:
        DataFrame with at most ``N_sample`` rows (or all rows if
        ``N_sample <= 0`` or ``N_sample >= len(phenotype)``).
    """
    n_total = len(phenotype)
    n_sample = int(params.get("N_sample", 0))
    seed = int(params.get("seed", 42))

    if n_sample <= 0 or n_sample >= n_total:
        logger.info("Sample pass-through: keeping all %d individuals (N_sample=%d)", n_total, n_sample)
        return phenotype

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_total, n_sample, replace=False))
    result = phenotype.iloc[indices].reset_index(drop=True)
    elapsed = time.perf_counter() - t0

    logger.info("Sampled %d → %d individuals in %.1fs (seed=%d)", n_total, n_sample, elapsed, seed)
    return result


def cli() -> None:
    """Command-line interface for subsampling phenotype data."""
    from sim_ace.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Subsample phenotyped individuals")
    add_logging_args(parser)
    parser.add_argument("--phenotype", required=True, help="Input phenotype parquet")
    parser.add_argument("--output", required=True, help="Output sampled phenotype parquet")
    parser.add_argument("--N-sample", type=int, default=0, help="Number of individuals to retain (0 = keep all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    init_logging(args)

    phenotype = pd.read_parquet(args.phenotype)
    params = {"N_sample": args.N_sample, "seed": args.seed}
    result = run_sample(phenotype, params)
    save_parquet(result, args.output)
