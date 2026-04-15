"""Subprocess wrapper around the ``ace_sreml`` C++ binary.

Writes GRMs, phenotype, and covariates to a temp directory in the formats
the binary reads (ace_sreml's own binary CSC for GRMs, PLINK-style text for
pheno/covar), invokes the binary with the caller's tunings, and parses the
TSV output back into a :class:`SparseREMLResult`.

Typical usage:

    from fit_ace.sparse_reml import fit_sparse_reml
    from sim_ace.core.pedigree_graph import PedigreeGraph

    pg = PedigreeGraph(ped_df)
    pg.compute_inbreeding()
    result = fit_sparse_reml(
        y=phenotype_y,
        kinship=pg._kinship_matrix,
        household_id=ped_df["household_id"].to_numpy(),
        grm_threshold=0.05,
        ordering="metis",
    )
    print(result.vc)
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipy.sparse as sp

from sim_ace.analysis.export_grm import (
    build_household_matrix,
    export_pheno_plink,
    export_sparse_grm_binary,
)

logger = logging.getLogger(__name__)


# Binary discovery: ACE_SREML_BIN env var wins, else a repo-relative default.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BINARY = _REPO_ROOT / "external" / "ace_sreml" / "build" / "ace_sreml"


def default_binary() -> Path:
    """Return the path to the ace_sreml binary.

    Honors the ``ACE_SREML_BIN`` environment variable first, then falls back
    to the repo-relative build output.
    """
    env = os.environ.get("ACE_SREML_BIN")
    if env:
        return Path(env)
    return _DEFAULT_BINARY


@dataclass
class SparseREMLResult:
    """Outcome of one ace_sreml fit.

    Attributes:
        vc: DataFrame with columns ``vc_name``, ``estimate``, ``se``.  Rows
            include one per GRM (e.g. ``V(A)``, ``V(C)``), then ``Ve``,
            ``Vp``, ``h2``, ``c2``.
        cov: (K+1, K+1) covariance matrix of the variance-component
            estimates (excluding the derived Vp/h²/c² rows).  Indexed by
            ``vc_name``.
        iter_log: per-iteration log.  Columns ``iter``, ``logLik``,
            ``dLLpred``, ``tau_nr``, ``grad_norm``, then one column per VC
            tracking its current value.
        wall_s: total wall-clock seconds reported by the binary.
        n_iter: iterations actually run.
        converged: whether the convergence criterion fired (rather than
            max_iter being reached).
        logLik: final REML log-likelihood.
        command: argv list that was executed (useful for debugging).
        bench: optional per-stage bench timings read from
            ``<out>.bench.tsv``; a DataFrame with ``stage``, ``wall_s``,
            ``calls``, ``mean_ms``.
    """

    vc: pd.DataFrame
    cov: pd.DataFrame
    iter_log: pd.DataFrame
    wall_s: float
    n_iter: int
    converged: bool
    logLik: float
    command: list[str]
    bench: pd.DataFrame | None = None


def fit_sparse_reml(
    y: np.ndarray,
    kinship: sp.spmatrix,
    *,
    household_id: np.ndarray | None = None,
    covariates: np.ndarray | None = None,
    iids: Sequence[str] | np.ndarray | None = None,
    grm_threshold: float = 0.0,
    n_rand_vec: int = 100,
    max_iter: int = 50,
    tol: float = 1e-3,
    seed: int = 42,
    threads: int = 8,
    ordering: str = "auto",
    log_level: str = "info",
    binary: Path | str | None = None,
    work_dir: Path | str | None = None,
    cleanup: bool = True,
) -> SparseREMLResult:
    """Fit an A(+C)+E sparse REML model via the ace_sreml binary.

    Args:
        y: (n,) phenotype vector.  NaNs are rejected — drop missing rows
            before calling.
        kinship: (n, n) sparse kinship matrix.  The wrapper doubles it to a
            GRM (A = 2·K) during export.  Any scipy.sparse format is OK;
            it's canonicalized to symmetric CSC on the write side.
        household_id: optional (n,) array mapping each individual to a
            household (int).  When given, a household C GRM is added to the
            fit so the model is A+C+E.  Pass ``None`` for an AE-only fit.
        covariates: optional (n, p) matrix of fixed-effect covariates.  An
            intercept is added automatically by the binary; do not include
            one here.
        iids: optional length-n iid labels.  If omitted, ``"id0", "id1", …``
            is generated.  Only matters internally (used to cross-reference
            rows between GRM and phenotype files).
        grm_threshold: drop off-diagonal GRM entries with |value| below this
            (after the ×2 kinship-to-GRM rescale).  ``0.05`` matches
            sparseREML's default and keeps Cholesky fill tractable at
            n ≳ 10⁴; ``0`` keeps everything.
        n_rand_vec, max_iter, tol, seed, threads, ordering, log_level:
            forwarded to ``ace_sreml``.
        binary: path to the compiled binary; defaults to
            ``default_binary()`` (env var ``ACE_SREML_BIN`` or the
            repo-relative build output).
        work_dir: directory in which to stage input files and collect
            outputs.  ``None`` uses a ``tempfile.TemporaryDirectory``.
        cleanup: when ``work_dir`` is ``None`` and this is ``True``, the
            temp directory is removed after parsing the result.  Pass
            ``False`` to inspect the intermediate files for debugging.

    Returns:
        :class:`SparseREMLResult`.

    Raises:
        FileNotFoundError: if the binary is missing.
        RuntimeError: if the binary exits non-zero.  The stderr log is
            re-attached to the exception message.
        ValueError: for malformed inputs (shape mismatch, NaNs, …).
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    if kinship.shape != (n, n):
        raise ValueError(f"kinship shape {kinship.shape} does not match len(y)={n}")
    if np.any(~np.isfinite(y)):
        raise ValueError("y contains NaN or inf; drop missing rows before calling")
    if household_id is not None:
        household_id = np.asarray(household_id)
        if household_id.shape != (n,):
            raise ValueError(f"household_id shape {household_id.shape} != ({n},)")
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if covariates.shape[0] != n:
            raise ValueError(f"covariates.shape[0]={covariates.shape[0]} != n={n}")
    if iids is None:
        iids = np.array([f"id{i}" for i in range(n)])
    else:
        iids = np.asarray(iids)
        if iids.shape != (n,):
            raise ValueError(f"iids shape {iids.shape} != ({n},)")

    bin_path = Path(binary) if binary is not None else default_binary()
    if not bin_path.exists():
        raise FileNotFoundError(
            f"ace_sreml binary not found at {bin_path}. "
            "Build it with `cmake -S external/ace_sreml -B external/ace_sreml/build && "
            "cmake --build external/ace_sreml/build`, or set the ACE_SREML_BIN env var."
        )

    # ------------------------------------------------------------------
    # Prepare a working directory and write the binary's inputs
    # ------------------------------------------------------------------
    cm: TemporaryDirectory | None = None
    if work_dir is None:
        cm = TemporaryDirectory(prefix="ace_sreml_")
        work = Path(cm.name)
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        inputs = work / "inputs"
        inputs.mkdir(exist_ok=True)
        out_prefix = work / "out"

        # y + covariates as a single DataFrame, routed through the PLINK exporter.
        df = pd.DataFrame({"id": iids, "y": y})
        covar_cols: tuple[str, ...] = ()
        if covariates is not None:
            covar_cols = tuple(f"cov{i + 1}" for i in range(covariates.shape[1]))
            for i, name in enumerate(covar_cols):
                df[name] = covariates[:, i]

        export_sparse_grm_binary(kinship, iids, inputs / "A", to_grm=True, threshold=grm_threshold)
        grm_prefixes = [inputs / "A"]
        if household_id is not None:
            export_sparse_grm_binary(
                build_household_matrix(household_id),
                iids,
                inputs / "C",
                to_grm=False,
                threshold=0.0,
            )
            grm_prefixes.append(inputs / "C")
        export_pheno_plink(df, inputs / "plink", pheno_col="y", covar_cols=covar_cols)

        grm_list_path = work / "grm_list.txt"
        grm_list_path.write_text("\n".join(str(p) for p in grm_prefixes) + "\n")

        cmd = [
            str(bin_path),
            "--grm_list",
            str(grm_list_path),
            "--phen_file",
            str(inputs / "plink.pheno.txt"),
            "--covar_file",
            str(inputs / "plink.covar.txt"),
            "--output",
            str(out_prefix),
            "--num_random_vectors",
            str(n_rand_vec),
            "--max_iter",
            str(max_iter),
            "--tol",
            str(tol),
            "--seed",
            str(seed),
            "--threads",
            str(threads),
            "--ordering",
            str(ordering),
            "--log-level",
            str(log_level),
        ]

        log_path = work / "ace_sreml.log"
        logger.info("ace_sreml: %s", " ".join(cmd))
        with log_path.open("w") as log_fh:
            proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            tail = log_path.read_text().splitlines()[-30:]
            raise RuntimeError(f"ace_sreml exited {proc.returncode}; last lines of log:\n" + "\n".join(tail))

        return _parse_result(out_prefix, cmd)
    finally:
        if cm is not None and cleanup:
            cm.cleanup()


def _parse_result(out_prefix: Path, command: list[str]) -> SparseREMLResult:
    """Read ``<out_prefix>.{vc,cov,iter}.tsv`` + meta sidecar into a result."""
    vc = pd.read_csv(f"{out_prefix}.vc.tsv", sep="\t")
    cov_raw = pd.read_csv(f"{out_prefix}.cov.tsv", sep="\t")
    cov = cov_raw.set_index("name")
    iter_log = pd.read_csv(f"{out_prefix}.iter.tsv", sep="\t")

    meta_path = Path(f"{out_prefix}.vc.tsv.meta")
    wall_s = float("nan")
    n_iter = 0
    converged = False
    logLik = float("nan")
    if meta_path.exists():
        for line in meta_path.read_text().strip().splitlines():
            key, _, value = line.partition("\t")
            if key == "wall_s":
                wall_s = float(value)
            elif key == "n_iter":
                n_iter = int(value)
            elif key == "converged":
                converged = bool(int(value))
            elif key == "logLik":
                logLik = float(value)

    bench_path = Path(f"{out_prefix}.bench.tsv")
    bench = pd.read_csv(bench_path, sep="\t") if bench_path.exists() else None

    return SparseREMLResult(
        vc=vc,
        cov=cov,
        iter_log=iter_log,
        wall_s=wall_s,
        n_iter=n_iter,
        converged=converged,
        logLik=logLik,
        command=command,
        bench=bench,
    )
