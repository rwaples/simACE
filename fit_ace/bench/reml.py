"""Helpers for benchmarking external REML tools (sparseREML, MPH) on ACE pedigrees.

The CLI wrapper lives at ``scripts/bench_sparse_reml.py``. This module keeps
the small reusable pieces — kinship subsetting, tool-output parsers, and a
timed-subprocess helper — in importable form so they can be unit-tested
independently of R and C++ toolchains.
"""

from __future__ import annotations

__all__ = [
    "build_kinship_for_subset",
    "parse_mph_vc_csv",
    "parse_sparse_reml_tsv",
    "run_timed_subprocess",
]

import logging
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sim_ace.core.pedigree_graph import PedigreeGraph

if TYPE_CHECKING:
    import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kinship subset construction
# ---------------------------------------------------------------------------


def build_kinship_for_subset(
    full_ped: pd.DataFrame,
    subset_ped: pd.DataFrame,
) -> sp.csr_matrix:
    """Compute kinship over the full pedigree, then return the subset submatrix.

    Uses :func:`sim_ace.core.pedigree_graph.PedigreeGraph.compute_inbreeding`,
    which walks generations and stores the resulting sparse K as
    ``pg._kinship_matrix``.

    Args:
        full_ped: full pedigree DataFrame (must include *all* ancestors of
            subset_ped) with columns ``id``, ``mother``, ``father``, ``twin``,
            ``sex``, ``generation``.
        subset_ped: rows whose ids we want the kinship submatrix for.
            Typically the phenotyped subset after twin collapse.

    Returns:
        Sparse (n_subset, n_subset) kinship matrix in CSR format, rows/cols
        ordered to match *subset_ped*.
    """
    pg = PedigreeGraph(full_ped)
    t0 = time.perf_counter()
    pg.compute_inbreeding()
    dt = time.perf_counter() - t0
    K_full = pg._kinship_matrix
    logger.info(
        "build_kinship_for_subset: full kinship n=%d built in %.2fs (nnz=%d)",
        K_full.shape[0],
        dt,
        K_full.nnz,
    )

    ids_full = full_ped["id"].to_numpy()
    max_id = int(ids_full.max())
    id_to_row = np.full(max_id + 1, -1, dtype=np.int64)
    id_to_row[ids_full] = np.arange(len(full_ped), dtype=np.int64)

    subset_ids = subset_ped["id"].to_numpy()
    if subset_ids.max() > max_id:
        raise ValueError("subset_ped contains ids absent from full_ped")
    rows = id_to_row[subset_ids]
    if np.any(rows < 0):
        raise ValueError("subset_ped contains ids absent from full_ped")

    K_csr = K_full.tocsr()
    K_sub = K_csr[rows, :][:, rows]
    logger.info(
        "build_kinship_for_subset: subset n=%d, nnz=%d (%.4f%% sparse)",
        K_sub.shape[0],
        K_sub.nnz,
        100 * K_sub.nnz / max(K_sub.shape[0] ** 2, 1),
    )
    return K_sub


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


def parse_sparse_reml_tsv(vc_path: Path, meta_path: Path | None = None) -> dict[str, Any]:
    """Parse the TSV written by ``scripts/run_sparse_reml_r.R``.

    Args:
        vc_path: path to the main TSV with columns ``vc_name``, ``estimate``,
            ``se``.
        meta_path: optional path to the sidecar ``<vc_path>.meta`` file; if
            omitted, defaults to ``Path(str(vc_path) + ".meta")``.

    Returns:
        Flat dict with ``est_<name>``/``se_<name>`` keys per variance component
        plus ``converged``, ``n_iter``, ``logLik``, ``reml_wall_s`` (the last
        comes from the meta sidecar; not the same as the outer wall time).
    """
    df = pd.read_csv(vc_path, sep="\t")
    out: dict[str, Any] = {}
    for _, row in df.iterrows():
        name = str(row["vc_name"])
        out[f"est_{name}"] = float(row["estimate"])
        out[f"se_{name}"] = float(row["se"])

    meta = Path(str(vc_path) + ".meta") if meta_path is None else Path(meta_path)
    if meta.exists():
        for line in meta.read_text().strip().splitlines():
            key, _, value = line.partition("\t")
            if key == "wall_s":
                out["reml_wall_s"] = float(value)
            elif key == "n_iter":
                out["n_iter"] = int(value)
            elif key == "converged":
                out["converged"] = int(value)
            elif key == "logLik":
                out["logLik"] = float(value)
    return out


def parse_mph_vc_csv(vc_csv: Path) -> dict[str, Any]:
    """Parse ``<output_file>.mq.vc.csv`` produced by MPH's REML run.

    MPH writes one row per variance component (plus one "err" row for
    residual), with columns ``trait_x, trait_y, vc_name, m, var, seV, pve,
    seP, ...``. In the ACE setup there are 2 GRM rows (A, then C) followed
    by the "err" row. We match positionally — the first non-"err" row is
    V(A), the second is V(C), and "err" becomes Ve.

    Args:
        vc_csv: path to the CSV.

    Returns:
        Flat dict with ``est_<name>``/``se_<name>`` keys for
        V(A), V(C), Ve, Vp (sum), h2, c2.
    """
    df = pd.read_csv(vc_csv)
    out: dict[str, Any] = {}

    mask_err = df["vc_name"].astype(str) == "err"
    non_err = df.loc[~mask_err].reset_index(drop=True)
    canonical = ["V(A)", "V(C)"]
    for i, canon in enumerate(canonical):
        if i >= len(non_err):
            break
        row = non_err.iloc[i]
        out[f"est_{canon}"] = float(row["var"])
        out[f"se_{canon}"] = float(row["seV"])
        out[f"est_{canon}_pve"] = float(row["pve"])
        out[f"se_{canon}_pve"] = float(row["seP"])

    err_rows = df.loc[mask_err]
    if len(err_rows) > 0:
        row = err_rows.iloc[0]
        out["est_Ve"] = float(row["var"])
        out["se_Ve"] = float(row["seV"])

    # Total Vp = sum of component vars (SE unavailable positionally, leave NaN).
    if {"est_V(A)", "est_V(C)", "est_Ve"}.issubset(out):
        out["est_Vp"] = out["est_V(A)"] + out["est_V(C)"] + out["est_Ve"]
        out["se_Vp"] = math.nan
    # Map pve to h2 / c2 on our canonical schema.
    if "est_V(A)_pve" in out:
        out["est_h2"] = out.pop("est_V(A)_pve")
        out["se_h2"] = out.pop("se_V(A)_pve")
    if "est_V(C)_pve" in out:
        out["est_c2"] = out.pop("est_V(C)_pve")
        out["se_c2"] = out.pop("se_V(C)_pve")
    return out


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def run_timed_subprocess(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> tuple[float, float, int]:
    """Run a subprocess with stdout/stderr captured; return wall, peak RSS, rc.

    ``ru_maxrss`` is reported in KB on Linux (POSIX 2001) and in bytes on
    macOS.  We convert both to MB.  Peak RSS is the *post*-run value for all
    child processes in aggregate — good enough for single-call tool runs.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w") as log_fh:
        proc = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            check=False,
        )
    wall = time.perf_counter() - t0
    rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    maxrss = rusage.ru_maxrss
    # Linux: KB → MB.  macOS: bytes → MB.
    if sys.platform == "darwin":
        peak_rss_mb = maxrss / (1024 * 1024)
    else:
        peak_rss_mb = maxrss / 1024
    return wall, peak_rss_mb, proc.returncode
