"""Randomly drop individuals from a pedigree to simulate incomplete observation."""

from __future__ import annotations

__all__ = ["run_dropout"]

import argparse
import logging
import time

import numpy as np
import pandas as pd

from sim_ace.core.utils import save_parquet

logger = logging.getLogger(__name__)


def run_dropout(pedigree: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Remove a random fraction of individuals from the pedigree.

    Dropped individuals are deleted entirely. Any mother/father/twin links
    that reference a dropped individual are set to -1 (unknown).

    Args:
        pedigree: DataFrame with columns id, mother, father, twin, etc.
        params: dict with keys ``pedigree_dropout_rate`` (float 0-1) and
                ``seed`` (int).

    Returns:
        DataFrame with dropped rows removed and dangling links severed.
    """
    rate = float(params.get("pedigree_dropout_rate", 0))
    seed = int(params.get("seed", 42))

    if rate <= 0:
        logger.info("Dropout pass-through: rate=%.3f, keeping all %d individuals", rate, len(pedigree))
        return pedigree

    n_total = len(pedigree)
    n_drop = round(n_total * rate)

    if n_drop <= 0:
        logger.info("Dropout pass-through: n_drop=0 for rate=%.4f, N=%d", rate, n_total)
        return pedigree

    if n_drop >= n_total:
        raise ValueError(f"Dropout rate {rate} would remove all {n_total} individuals")

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    drop_indices = rng.choice(n_total, n_drop, replace=False)

    # Collect IDs to drop
    all_ids = pedigree["id"].values
    drop_ids = set(all_ids[drop_indices].tolist())

    # Keep non-dropped rows
    keep_mask = np.ones(n_total, dtype=bool)
    keep_mask[drop_indices] = False
    result = pedigree.loc[keep_mask].copy()

    # Sever parent links pointing to dropped individuals
    for col in ("mother", "father", "twin"):
        if col in result.columns:
            vals = result[col].values
            dangling = np.isin(vals, list(drop_ids)) & (vals >= 0)
            if dangling.any():
                result.loc[result.index[dangling], col] = -1

    result = result.reset_index(drop=True)
    elapsed = time.perf_counter() - t0

    logger.info(
        "Dropout: %d → %d individuals (dropped %d, rate=%.3f) in %.2fs (seed=%d)",
        n_total,
        len(result),
        n_drop,
        rate,
        elapsed,
        seed,
    )
    return result


def cli() -> None:
    """Command-line interface for pedigree dropout."""
    from sim_ace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Randomly drop individuals from a pedigree")
    add_logging_args(parser)
    parser.add_argument("--pedigree", required=True, help="Input pedigree parquet")
    parser.add_argument("--output", required=True, help="Output pedigree parquet")
    parser.add_argument("--dropout-rate", type=float, default=0, help="Fraction of individuals to drop (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    init_logging(args)

    pedigree = pd.read_parquet(args.pedigree)
    params = {"pedigree_dropout_rate": args.dropout_rate, "seed": args.seed}
    result = run_dropout(pedigree, params)
    save_parquet(result, args.output)
