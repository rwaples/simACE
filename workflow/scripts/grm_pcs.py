"""Snakemake wrapper: sparse GRM binary → top-n_pcs eigenvectors + eigenvalues TSV."""

from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))

    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from sim_ace.analysis.export_grm import read_sparse_grm_binary
    from sim_ace.core.kinship_pcs import compute_kinship_pcs
    from sim_ace.core.utils import save_parquet

    logger = logging.getLogger(__name__)

    n_pcs = int(snakemake.params.n_pcs)
    seed = int(snakemake.params.seed)

    bin_path = Path(snakemake.input.grm_bin)
    if not bin_path.name.endswith(".grm.sp.bin"):
        raise ValueError(f"expected input to end with .grm.sp.bin, got {bin_path}")
    prefix = bin_path.with_name(bin_path.name[: -len(".grm.sp.bin")])

    # Binary stores 2φ (GRM); divide by 2 to recover kinship scale for PCA.
    # (Scaling doesn't change eigenvectors, only eigenvalues — dividing keeps
    # them interpretable on the kinship scale that upstream code expects.)
    grm_matrix, iids_str = read_sparse_grm_binary(prefix)
    kinship_matrix = grm_matrix / 2.0
    logger.info("grm_pcs: loaded N=%d, nnz=%d", kinship_matrix.shape[0], kinship_matrix.nnz)

    ids = np.asarray(iids_str).astype(np.int64)
    ids, pcs, eigenvalues, trace = compute_kinship_pcs(
        matrix=kinship_matrix,
        ids=ids,
        n_pcs=n_pcs,
        seed=seed,
    )

    pc_cols = {f"PC{i + 1}": pcs[:, i] for i in range(n_pcs)}
    pcs_df = pd.DataFrame({"id": ids, **pc_cols})
    save_parquet(pcs_df, snakemake.output.pcs)
    logger.info("Wrote PCs: %s (%d rows, %d columns)", snakemake.output.pcs, len(pcs_df), pcs_df.shape[1])

    ranks = np.arange(1, n_pcs + 1, dtype=np.int32)
    ve = eigenvalues / trace if trace > 0 else np.full(n_pcs, np.nan)
    ve_filled = np.where(np.isnan(ve), 0.0, ve)
    cum_ve = np.cumsum(ve_filled)
    cum_ve = np.where(np.isnan(ve), np.nan, cum_ve)

    eig_df = pd.DataFrame(
        {
            "rank": ranks,
            "eigenvalue": eigenvalues,
            "variance_explained": ve,
            "cumulative_variance_explained": cum_ve,
        }
    )
    eig_df.to_csv(snakemake.output.eigenvalues, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote eigenvalues: %s (trace=%.4f)", snakemake.output.eigenvalues, trace)


if __name__ == "__main__":
    try:
        snakemake  # type: ignore[name-defined]
    except NameError as exc:
        raise SystemExit("grm_pcs.py is a Snakemake script wrapper; run via Snakemake.") from exc
    else:
        _run_snakemake()
