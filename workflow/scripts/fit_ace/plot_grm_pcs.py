"""Snakemake wrapper for the GRM PC plot."""

from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake() -> None:
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))

    import pandas as pd

    from fit_ace.plotting.plot_grm_pcs import plot_grm_pcs

    pcs_df = pd.read_parquet(snakemake.input.pcs)
    eigenvalues_df = pd.read_csv(snakemake.input.eigenvalues, sep="\t")
    phenotype_df = pd.read_parquet(snakemake.input.phenotype)

    plot_grm_pcs(
        pcs_df=pcs_df,
        eigenvalues_df=eigenvalues_df,
        phenotype_df=phenotype_df,
        output_path=snakemake.output.plot,
        scenario=str(snakemake.wildcards.scenario),
        rep=str(snakemake.wildcards.rep),
    )


if __name__ == "__main__":
    try:
        snakemake  # type: ignore[name-defined]
    except NameError as exc:
        raise SystemExit("plot_grm_pcs.py is a Snakemake script wrapper; run via Snakemake.") from exc
    else:
        _run_snakemake()
