"""Per-rep downsampling for scatter/histogram plot inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def create_sample(
    df: pd.DataFrame,
    seed: int = 42,
    n_per_gen: int = 50_000,
) -> pd.DataFrame:
    """Downsample for scatter/histogram plots, preserving parent rows."""
    rng = np.random.default_rng(seed)
    generations = df["generation"].values
    unique_gens = sorted(np.unique(generations))
    if all(int((generations == g).sum()) <= n_per_gen for g in unique_gens):
        return df.copy()
    ids = df["id"].values
    max_id = int(ids.max()) + 1
    id_to_row = np.full(max_id, -1, dtype=np.int32)
    id_to_row[ids] = np.arange(len(df), dtype=np.int32)
    selected = set()
    for gen in unique_gens:
        gen_idx = np.where(generations == gen)[0]
        chosen = rng.choice(gen_idx, min(len(gen_idx), n_per_gen), replace=False)
        selected.update(chosen.tolist())
    tmp = np.array(list(selected), dtype=np.intp)
    for pid_arr in [df["mother"].values[tmp], df["father"].values[tmp]]:
        valid = (pid_arr >= 0) & (pid_arr < max_id)
        rows = id_to_row[pid_arr[valid]]
        selected.update(rows[rows >= 0].tolist())
    return df.iloc[np.sort(np.array(list(selected), dtype=np.intp))].copy()
