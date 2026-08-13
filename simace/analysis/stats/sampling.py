"""Per-rep downsampling for scatter/histogram plot inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd


def create_sample(
    df: pd.DataFrame | pl.DataFrame,
    seed: int = 42,
    n_per_gen: int = 50_000,
) -> pd.DataFrame | pl.DataFrame:
    """Downsample for scatter/histogram plots, preserving parent rows.

    Same-type dual-frame API (transitional, ADR 0015): returns the frame
    library it was given. All random selection runs on NumPy row positions, so
    the fixed-seed sampled rows are identical under either library
    (decision 14). Cross-repo consumer: fitACE.
    """
    rng = np.random.default_rng(seed)
    is_polars = isinstance(df, pl.DataFrame)
    generations = df["generation"].to_numpy()
    unique_gens = sorted(np.unique(generations))
    if all(int((generations == g).sum()) <= n_per_gen for g in unique_gens):
        return df.clone() if is_polars else df.copy()
    ids = df["id"].to_numpy()
    max_id = int(ids.max()) + 1
    id_to_row = np.full(max_id, -1, dtype=np.int32)
    id_to_row[ids] = np.arange(len(df), dtype=np.int32)
    sampled_chunks = []
    for gen in unique_gens:
        gen_idx = np.where(generations == gen)[0]
        sampled_chunks.append(rng.choice(gen_idx, min(len(gen_idx), n_per_gen), replace=False))
    sampled_rows = np.concatenate(sampled_chunks)
    parent_rows = []
    for pid_arr in (df["mother"].to_numpy()[sampled_rows], df["father"].to_numpy()[sampled_rows]):
        valid = (pid_arr >= 0) & (pid_arr < max_id)
        rows = id_to_row[pid_arr[valid]]
        parent_rows.append(rows[rows >= 0])
    final_rows = np.unique(np.concatenate([sampled_rows, *parent_rows]))
    if is_polars:
        return df[final_rows]
    return df.iloc[final_rows].copy()
