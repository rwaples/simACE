"""PA-FGRS scoring - Snakemake wrapper with CLI fallback."""

from fit_ace.pafgrs.pafgrs import cli as _cli
from sim_ace import _snakemake_tag, setup_logging


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from fit_ace.pafgrs.pafgrs import (
        build_kinship_from_pairs,
        build_pheno_lookups_univariate,
        compute_empirical_cip,
        compute_thresholds_and_w,
        compute_true_cip_weibull,
        prepare_univariate_scoring,
        score_univariate_variant,
    )
    from fit_ace.pafgrs.pafgrs_metrics import compute_pafgrs_metrics
    from sim_ace.analysis.ltm_falconer import compute_ltm_falconer
    from sim_ace.core.utils import save_parquet

    logger = logging.getLogger(__name__)
    p = snakemake.params

    pedigree_df = pd.read_parquet(
        snakemake.input.pedigree,
        columns=["id", "mother", "father", "twin", "sex", "generation"],
    )
    phenotype_df = pd.read_parquet(
        snakemake.input.phenotype,
        columns=["id", "generation", "affected1", "affected2", "A1", "A2", "t_observed1", "t_observed2"],
    )

    base_dir = str(Path(snakemake.output.done).parent)

    ndegree = int(p.ndegree)
    censor_age = float(p.censor_age)

    # Build kinship once
    kmat = build_kinship_from_pairs(pedigree_df, ndegree=ndegree)

    # Prep once: extract relatives, pre-extract kinship
    prep = prepare_univariate_scoring(pedigree_df, phenotype_df, ndegree, kmat)

    # Shared columns (written once)
    combined = pd.DataFrame({"id": phenotype_df["id"].values})
    all_metrics: list[dict] = []
    combined["generation"] = phenotype_df["generation"].values.astype(np.int32)

    for trait_num in p.trait_nums:
        trait_key = f"trait{trait_num}"
        h2_true = float(getattr(p, f"A{trait_num}"))
        prevalence = float(getattr(p, f"prevalence{trait_num}"))
        beta = float(getattr(p, f"beta{trait_num}"))
        model = getattr(p, f"phenotype_model{trait_num}")
        pheno_params = getattr(p, f"phenotype_params{trait_num}") or {}

        combined[f"affected{trait_num}"] = phenotype_df[f"affected{trait_num}"].values
        combined[f"true_A{trait_num}"] = phenotype_df[f"A{trait_num}"].values.astype(np.float32)

        # CIP tables
        cip_tables = {}
        cip_ages_emp, cip_vals_emp, prev_emp = compute_empirical_cip(
            phenotype_df,
            trait_num,
            max_age=censor_age,
        )
        cip_tables["empirical"] = (cip_ages_emp, cip_vals_emp, prev_emp)

        if model == "frailty" and pheno_params.get("distribution") == "weibull":
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

        # h2 estimation
        try:
            falconer = compute_ltm_falconer(
                phenotype_df,
                kinds=["FS"],
                trait_num=trait_num,
                seed=int(p.seed),
                pedigree=pedigree_df,
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

        affected = phenotype_df[f"affected{trait_num}"].values.astype(bool)
        t_observed = phenotype_df[f"t_observed{trait_num}"].values

        for cip_source in p.cip_sources:
            cip_ages, cip_vals, lifetime_prev = cip_tables[cip_source]

            # Compute thresholds/w once per trait × CIP source
            thresholds, w = compute_thresholds_and_w(
                affected,
                t_observed,
                cip_ages,
                cip_vals,
                lifetime_prev,
            )
            lookup_aff, lookup_thr, lookup_w = build_pheno_lookups_univariate(
                prep,
                affected,
                thresholds,
                w,
            )

            for h2_source in p.h2_sources:
                h2 = h2_values[h2_source]
                tag = f"{trait_key}_{cip_source}_{h2_source}"
                logger.info("Scoring: %s (h2=%.3f, prev=%.3f)", tag, h2, lifetime_prev)

                scores = score_univariate_variant(
                    prep,
                    h2,
                    trait_num,
                    lookup_aff,
                    lookup_thr,
                    lookup_w,
                    phenotype_df,
                )

                combined[f"est_{tag}"] = scores["est"].values.astype(np.float32)
                combined[f"var_{tag}"] = scores["var"].values.astype(np.float32)
                combined[f"nrel_{tag}"] = scores["n_relatives"].values.astype(np.int16)

                metrics = compute_pafgrs_metrics(scores)
                all_metrics.append({**metrics, "trait": trait_key, "cip_source": cip_source, "h2_source": h2_source})

    scores_path = f"{base_dir}/scores.parquet"
    save_parquet(combined, scores_path)
    logger.info("Wrote combined scores: %s (%d rows, %d cols)", scores_path, len(combined), len(combined.columns))

    metrics_path = f"{base_dir}/metrics.tsv"
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    logger.info("Wrote combined metrics: %s (%d rows)", metrics_path, len(metrics_df))


try:
    snakemake
    _run_snakemake()
except NameError:
    _cli()
