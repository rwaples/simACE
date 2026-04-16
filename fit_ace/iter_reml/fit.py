"""Subprocess wrapper around the ``ace_iter_reml`` C++ binary.

Mirrors :mod:`fit_ace.sparse_reml.fit` so callers can swap the two
fitters with no other code changes.  The two-phase fit (Phase 1 RHE-mc
warm-start → Phase 2 PCG-AI-REML) outputs the same TSV schemas as
``ace_sreml`` so the existing ``_parse_result`` parser reads our outputs
unchanged.

Typical usage:

    from fit_ace.iter_reml import fit_iter_reml
    from sim_ace.core.pedigree_graph import PedigreeGraph

    pg = PedigreeGraph(ped_df)
    pg.compute_inbreeding()
    result = fit_iter_reml(
        y=phenotype_y,
        kinship=pg._kinship_matrix,
        household_id=ped_df["household_id"].to_numpy(),
        grm_threshold=0.05,
        pc_type="deflation",
        deflation_k=200,
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
from typing import TYPE_CHECKING, Literal, get_args

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


PcType = Literal["jacobi", "deflation"]


def _check_choice(value: str, choices: tuple[str, ...], arg_name: str) -> None:
    if value not in choices:
        raise ValueError(f"{arg_name}={value!r} not in {list(choices)}")


# Binary discovery.
# Default is the fp32 build (build-fp32/) — ~25% faster wall and 1.5× less
# memory than fp64 on pedigree VC fits; σ² estimates match fp64 to 4 dp
# (within AI-SE).  The fp64 build (build/) is still available as a
# fallback via ACE_ITER_REML_BIN env var; use it only when (a) PCG
# relative residuals < 1e-6 are required or (b) the fp32 env isn't built.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BINARY_FP32 = _REPO_ROOT / "fit_ace" / "ace_iter_reml" / "build-fp32" / "ace_iter_reml"
_DEFAULT_BINARY_FP64 = _REPO_ROOT / "fit_ace" / "ace_iter_reml" / "build-fp64" / "ace_iter_reml"


def default_binary() -> Path:
    """Return the path to the ace_iter_reml binary.

    Resolution order:
      1. ``ACE_ITER_REML_BIN`` env var if set (explicit override).
      2. The fp32 build (``build-fp32/ace_iter_reml``) if it exists.
      3. The fp64 build (``build-fp64/ace_iter_reml``) as fallback.

    Each build has RPATH baked into its own conda env's lib dir
    (``ace_iter_reml`` for fp64, ``ace_iter_reml_fp32`` for fp32), so
    subprocess invocation does not need ``conda run -n ...``.
    """
    env = os.environ.get("ACE_ITER_REML_BIN")
    if env:
        return Path(env)
    if _DEFAULT_BINARY_FP32.exists():
        return _DEFAULT_BINARY_FP32
    return _DEFAULT_BINARY_FP64


@dataclass
class IterREMLResult:
    """Outcome of one ace_iter_reml fit.

    Attributes mirror :class:`fit_ace.sparse_reml.SparseREMLResult` so the
    two fitters are interchangeable for downstream consumers.

    Attributes:
        vc: DataFrame with columns ``vc_name``, ``estimate``, ``se``;
            rows ``V(A)``, ``V(C)``, ``Ve``, ``Vp``, ``h2``, ``c2``.
        cov: 3×3 covariance matrix of the variance-component estimates,
            indexed by ``name`` (``V(A)``, ``V(C)``, ``Ve``).  Computed
            as the inverse of the AI matrix at convergence.
        iter_log: per-iteration log.  Columns ``iter``, ``logLik`` (NA in
            v1, no log|V|), ``dLLpred``, ``tau_nr``, ``grad_norm``,
            ``VC_A``, ``VC_C``, ``VC_E``, ``pcg_iters_avg``.
        wall_s: total wall-clock seconds (Phase 1 + Phase 2).
        n_iter: AI-REML outer iterations actually run.
        converged: whether the convergence criterion fired.
        logLik: NaN in v1 (no log|V|; SLQ deferred to a later milestone).
        command: argv list executed (useful for debugging).
        bench: per-stage bench timings from ``<out>.bench.tsv`` —
            ``stage``, ``wall_s``, ``calls``, ``mean_ms``.
        phase1_vc: optional RHE-mc warm-start estimates if Phase 1 ran;
            same schema as ``vc`` but no SE/cov layer (jackknife only).
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
    phase1_vc: pd.DataFrame | None = None


def fit_iter_reml(
    y: np.ndarray,
    kinship: sp.spmatrix | None = None,
    *,
    household_id: np.ndarray | None = None,
    iids: Sequence[str] | np.ndarray | None = None,
    pedigree_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    phen_ids_in_pedigree: np.ndarray | None = None,
    grm_threshold: float = 0.0,
    phase1_probes: int = 100,
    phase1_blocks: int = 10,
    phase2_probes: int = 60,
    max_iter: int = 50,
    tol: float = 1e-3,
    pcg_tol: float = 1e-5,
    pcg_max_iter: int = 500,
    pc_type: PcType = "jacobi",
    deflation_k: int = 100,
    seed: int = 42,
    threads: int = 8,
    no_center: bool = False,
    skip_phase1: bool = False,
    skip_phase2: bool = False,
    log_level: str = "info",
    binary: Path | str | None = None,
    work_dir: Path | str | None = None,
    cleanup: bool = True,
) -> IterREMLResult:
    """Fit an A+C+E iterative REML model via the ace_iter_reml binary.

    Args:
        y: (n,) phenotype vector.  NaNs are rejected — drop missing rows
            before calling.
        kinship: (n, n) sparse kinship matrix.  Doubled to a GRM
            (A = 2·K) on export.  Any scipy.sparse format is OK.
        household_id: (n,) integer household labels.  A household
            indicator C matrix is built from these.  Required: an A+C+E
            model is the only thing v1 fits (drop the column for AE-only
            and pass an array of unique ids if you want C ≈ I — but a
            single-person-per-household C contributes nothing).
        iids: optional length-n iid labels.  If omitted,
            ``"id00000", "id00001", …`` is generated.  Used only to
            cross-reference rows between GRM and phenotype files.
        grm_threshold: drop off-diagonal A entries with |value| below
            this (after the ×2 kinship-to-GRM rescale).  ``0.05`` is a
            sensible default for pedigree GRMs at large n; the binary's
            RHE-mc and PCG do not require thresholding for correctness
            (it only controls memory).
        phase1_probes, phase1_blocks: RHE-mc tunings.  Defaults match the
            warm-start jackknife configuration.
        phase2_probes: Hutchinson trace probes per AI-REML iter.
        max_iter, tol: AI-REML outer loop cap and |dLLpred| convergence
            threshold.
        pcg_tol, pcg_max_iter: PCG inner solver controls.
        pc_type: ``"jacobi"`` (cheap; works on small/well-conditioned V)
            or ``"deflation"`` (top-k SVD of A + Jacobi on complement;
            required at large connected pedigrees).
        deflation_k: top-k eigenvectors of A to deflate; ignored when
            pc_type=jacobi.
        seed: RNG seed for Rademacher probes (Phase 1 + Phase 2).
        threads: OpenMP threads.
        no_center: by default y is mean-centered before the fit
            (intercept-only fixed effect).  Pass True to skip.
        skip_phase1: skip RHE-mc; Phase 2 starts from σ²=var(y)/3.
        skip_phase2: only run Phase 1; emits ``<out>.phase1.tsv``.
        binary: path to the compiled ``ace_iter_reml``; defaults to
            :func:`default_binary`.
        work_dir: directory in which to stage input files and collect
            outputs.  ``None`` uses a ``tempfile.TemporaryDirectory``.
        cleanup: when ``work_dir`` is ``None`` and this is ``True``, the
            temp directory is removed after parsing.  Pass ``False`` to
            inspect intermediate files for debugging.

    Returns:
        :class:`IterREMLResult`.

    Raises:
        FileNotFoundError: if the binary is missing.
        RuntimeError: if the binary exits non-zero.  The stderr log is
            re-attached to the exception message.
        ValueError: for malformed inputs (shape mismatch, NaNs, ...).
    """
    _check_choice(pc_type, get_args(PcType), "pc_type")

    if (kinship is None) == (pedigree_arrays is None):
        raise ValueError("pass exactly one of kinship= (scipy sparse) "
                         "or pedigree_arrays= (ids, mothers, fathers, twins)")

    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    if kinship is not None and kinship.shape != (n, n):
        raise ValueError(f"kinship shape {kinship.shape} does not match len(y)={n}")
    if np.any(~np.isfinite(y)):
        raise ValueError("y contains NaN or inf; drop missing rows before calling")
    if household_id is None:
        raise ValueError("household_id is required for the A+C+E model in v1")
    household_id = np.asarray(household_id)
    if household_id.shape != (n,):
        raise ValueError(f"household_id shape {household_id.shape} != ({n},)")
    if iids is None:
        iids = np.array([f"id{i:08d}" for i in range(n)])
    else:
        iids = np.asarray(iids)
        if iids.shape != (n,):
            raise ValueError(f"iids shape {iids.shape} != ({n},)")

    bin_path = Path(binary) if binary is not None else default_binary()
    if not bin_path.exists():
        raise FileNotFoundError(
            f"ace_iter_reml binary not found at {bin_path}. "
            "Build it with `conda env create -f fit_ace/ace_iter_reml/environment.yml` "
            "(one-time) then `conda run -n ace_iter_reml cmake -S fit_ace/ace_iter_reml "
            "-B fit_ace/ace_iter_reml/build-fp64 && conda run -n ace_iter_reml "
            "cmake --build fit_ace/ace_iter_reml/build-fp64`. "
            "For the default (fp32) build, create the ace_iter_reml_fp32 env "
            "and build into fit_ace/ace_iter_reml/build-fp32.  Or set the "
            "ACE_ITER_REML_BIN env var to a specific binary path."
        )

    cm: TemporaryDirectory | None = None
    if work_dir is None:
        cm = TemporaryDirectory(prefix="ace_iter_reml_")
        work = Path(cm.name)
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        inputs = work / "inputs"
        inputs.mkdir(exist_ok=True)
        out_prefix = work / "out"

        df = pd.DataFrame({"id": iids, "y": y})
        if pedigree_arrays is not None:
            # Fast path: build kinship directly via numba + write ACEGRM
            # bytes in one pass, skipping the scipy sparse intermediate.
            from fit_ace.pafgrs.kinship_fast import build_kinship_to_acegrm

            ped_ids_arr, mothers_arr, fathers_arr, twins_arr = pedigree_arrays
            if phen_ids_in_pedigree is None:
                phen_ids_arr = np.asarray(iids)
            else:
                phen_ids_arr = np.asarray(phen_ids_in_pedigree)
            build_kinship_to_acegrm(
                ped_ids=np.asarray(ped_ids_arr),
                mothers=np.asarray(mothers_arr),
                fathers=np.asarray(fathers_arr),
                twins=None if twins_arr is None else np.asarray(twins_arr),
                out_prefix=inputs / "A",
                phen_ids=phen_ids_arr,
                to_grm=True,
                threshold=grm_threshold,
            )
        else:
            export_sparse_grm_binary(kinship, iids, inputs / "A", to_grm=True, threshold=grm_threshold)
        export_sparse_grm_binary(
            build_household_matrix(household_id),
            iids,
            inputs / "C",
            to_grm=False,
            threshold=0.0,
        )
        export_pheno_plink(df, inputs / "plink", pheno_col="y")

        cmd = [
            str(bin_path),
            "--grm_a", str(inputs / "A"),
            "--grm_c", str(inputs / "C"),
            "--phen_file", str(inputs / "plink.pheno.txt"),
            "--output", str(out_prefix),
            "--phase1_probes", str(phase1_probes),
            "--phase1_blocks", str(phase1_blocks),
            "--phase2_probes", str(phase2_probes),
            "--max_iter", str(max_iter),
            "--tol", str(tol),
            "--pcg_tol", str(pcg_tol),
            "--pcg_max_iter", str(pcg_max_iter),
            "--pc_type", pc_type,
            "--deflation_k", str(deflation_k),
            "--seed", str(seed),
            "--threads", str(threads),
            "--log-level", log_level,
        ]
        if no_center:
            cmd.append("--no-center")
        if skip_phase1:
            cmd.append("--skip_phase1")
        if skip_phase2:
            cmd.append("--skip_phase2")

        log_path = work / "ace_iter_reml.log"
        logger.info("ace_iter_reml: %s", " ".join(cmd))
        with log_path.open("w") as log_fh:
            proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            tail = log_path.read_text().splitlines()[-30:]
            raise RuntimeError(
                f"ace_iter_reml exited {proc.returncode}; last lines of log:\n"
                + "\n".join(tail)
            )

        return _parse_result(out_prefix, cmd, skip_phase1=skip_phase1, skip_phase2=skip_phase2)
    finally:
        if cm is not None and cleanup:
            cm.cleanup()


def _parse_result(
    out_prefix: Path,
    command: list[str],
    *,
    skip_phase1: bool = False,
    skip_phase2: bool = False,
) -> IterREMLResult:
    """Read ``<out_prefix>.{vc,cov,iter}.tsv`` + meta sidecar into a result."""
    if skip_phase2:
        # Phase 1 only — vc.tsv comes from .phase1.tsv; no cov, no iter log.
        vc = pd.read_csv(f"{out_prefix}.phase1.tsv", sep="\t")
        cov = pd.DataFrame()
        iter_log = pd.DataFrame()
    else:
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
                try:
                    logLik = float(value)
                except ValueError:
                    pass

    bench_path = Path(f"{out_prefix}.bench.tsv")
    bench = pd.read_csv(bench_path, sep="\t") if bench_path.exists() else None

    phase1_path = Path(f"{out_prefix}.phase1.tsv")
    phase1_vc = (
        pd.read_csv(phase1_path, sep="\t") if (phase1_path.exists() and not skip_phase1) else None
    )

    return IterREMLResult(
        vc=vc,
        cov=cov,
        iter_log=iter_log,
        wall_s=wall_s,
        n_iter=n_iter,
        converged=converged,
        logLik=logLik,
        command=command,
        bench=bench,
        phase1_vc=phase1_vc,
    )
