"""PA-FGRS scoring - Snakemake wrapper with CLI fallback."""

from sim_ace import _snakemake_tag, setup_logging
from sim_ace.pafgrs import cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from sim_ace.ltm_falconer import compute_ltm_falconer
    from sim_ace.pafgrs import (
        build_sparse_kinship,
        compute_empirical_cip,
        compute_true_cip_weibull,
        score_probands,
    )
    from sim_ace.pafgrs_metrics import compute_pafgrs_metrics, write_metrics_tsv
    from sim_ace.utils import save_parquet

    logger = logging.getLogger(__name__)
    p = snakemake.params

    pedigree_df = pd.read_parquet(snakemake.input.pedigree)
    phenotype_df = pd.read_parquet(snakemake.input.phenotype)

    base_dir = str(Path(snakemake.output.done).parent)

    ndegree = int(p.ndegree)
    censor_age = float(p.censor_age)

    # Build kinship once for all variants
    kmat = build_sparse_kinship(
        pedigree_df["id"].values,
        pedigree_df["mother"].values,
        pedigree_df["father"].values,
        pedigree_df["twin"].values if "twin" in pedigree_df.columns else None,
    )

    for trait_num in p.trait_nums:
        trait_key = f"trait{trait_num}"
        h2_true = float(getattr(p, f"A{trait_num}"))
        prevalence = float(getattr(p, f"prevalence{trait_num}"))
        beta = float(getattr(p, f"beta{trait_num}"))
        model = getattr(p, f"phenotype_model{trait_num}")
        pheno_params = getattr(p, f"phenotype_params{trait_num}") or {}

        cip_tables = {}
        cip_ages_emp, cip_vals_emp, prev_emp = compute_empirical_cip(
            phenotype_df, trait_num, max_age=censor_age,
        )
        cip_tables["empirical"] = (cip_ages_emp, cip_vals_emp, prev_emp)

        if model == "weibull":
            cip_ages_true, cip_vals_true = compute_true_cip_weibull(
                scale=pheno_params["scale"],
                rho=pheno_params["rho"],
                beta=beta,
                max_age=censor_age,
            )
            cip_tables["true"] = (cip_ages_true, cip_vals_true, prevalence)
        else:
            logger.warning("True CIP for model %s not implemented; using empirical", model)
            cip_tables["true"] = cip_tables["empirical"]

        try:
            falconer = compute_ltm_falconer(
                phenotype_df, kinds=["FS"], trait_num=trait_num,
                seed=int(p.seed), pedigree=pedigree_df,
            )
            h2_estimated = falconer.get("FS", {}).get("h2_falconer")
            if h2_estimated is None or np.isnan(h2_estimated):
                h2_estimated = h2_true
                logger.warning("Falconer h2 estimation failed; using true h2")
            else:
                h2_estimated = float(np.clip(h2_estimated, 0.01, 0.99))
        except Exception:
            logger.warning("Falconer h2 estimation failed; using true h2", exc_info=True)
            h2_estimated = h2_true

        h2_values = {"true": h2_true, "estimated": h2_estimated}

        for cip_source in p.cip_sources:
            cip_ages, cip_vals, lifetime_prev = cip_tables[cip_source]

            for h2_source in p.h2_sources:
                h2 = h2_values[h2_source]
                tag = f"{trait_key}_{cip_source}_{h2_source}"
                logger.info("Scoring: %s (h2=%.3f, prev=%.3f)", tag, h2, lifetime_prev)

                scores = score_probands(
                    pedigree_df,
                    phenotype_df,
                    h2=h2,
                    cip_ages=cip_ages,
                    cip_values=cip_vals,
                    lifetime_prevalence=lifetime_prev,
                    trait_num=trait_num,
                    ndegree=ndegree,
                    kmat=kmat,
                )

                scores_path = f"{base_dir}/scores_{tag}.parquet"
                save_parquet(scores, scores_path)
                logger.info("Wrote %s", scores_path)

                metrics = compute_pafgrs_metrics(scores)
                metrics_path = f"{base_dir}/metrics_{tag}.tsv"
                write_metrics_tsv(
                    metrics, metrics_path,
                    trait=trait_key, cip_source=cip_source, h2_source=h2_source,
                )


try:
    snakemake
    _run_snakemake()
except NameError:
    _cli()
