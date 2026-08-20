"""Per-rep downsampling for scatter/histogram plot inputs."""

from __future__ import annotations

import numpy as np
import polars as pl


def create_sample(
    df: pl.DataFrame,
    seed: int = 42,
    n_per_gen: int = 50_000,
) -> pl.DataFrame:
    """Downsample for scatter/histogram plots, preserving parent rows.

    All random selection runs on NumPy row positions, so the fixed-seed
    sampled rows are unchanged by the polars migration (ADR 0015 decision
    14). Cross-repo consumer: fitACE.

    Raises:
        TypeError: If ``df`` is not a polars DataFrame (ADR 0015).
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            "create_sample requires a polars DataFrame since the polars migration "
            f"(ADR 0015); got {type(df).__name__}. Convert with pl.from_pandas(...) at the call site."
        )
    rng = np.random.default_rng(seed)
    generations = df["generation"].to_numpy()
    unique_gens = sorted(np.unique(generations))
    if all(int((generations == g).sum()) <= n_per_gen for g in unique_gens):
        return df.clone()
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
    return df[final_rows]
