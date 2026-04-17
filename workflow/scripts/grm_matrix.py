"""Snakemake wrapper: pedigree → sparse GRM binary (ACEGRM format)."""

from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))

    import logging
    from pathlib import Path

    import pandas as pd

    from sim_ace.analysis.export_grm import export_sparse_grm_binary
    from sim_ace.core.kinship import build_sparse_kinship

    logger = logging.getLogger(__name__)

    min_kinship = float(snakemake.params.min_kinship)
    pedigree = pd.read_parquet(snakemake.input.pedigree)
    logger.info(
        "grm_matrix: N=%d, min_kinship=%g, pedigree=%s",
        len(pedigree),
        min_kinship,
        snakemake.input.pedigree,
    )

    K = build_sparse_kinship(
        pedigree["id"].to_numpy(),
        pedigree["mother"].to_numpy(),
        pedigree["father"].to_numpy(),
        pedigree["twin"].to_numpy(),
        min_kinship=min_kinship,
    )

    # Output prefix is the path minus the ".grm.sp.bin" suffix
    bin_path = Path(snakemake.output.grm_bin)
    if not bin_path.name.endswith(".grm.sp.bin"):
        raise ValueError(f"expected output to end with .grm.sp.bin, got {bin_path}")
    prefix = bin_path.with_name(bin_path.name[: -len(".grm.sp.bin")])

    export_sparse_grm_binary(
        K=K,
        iids=pedigree["id"].to_numpy(),
        prefix=prefix,
        to_grm=True,
        threshold=0.0,
    )


if __name__ == "__main__":
    try:
        snakemake  # type: ignore[name-defined]
    except NameError as exc:
        raise SystemExit("grm_matrix.py is a Snakemake script wrapper; run via Snakemake.") from exc
    else:
        _run_snakemake()
